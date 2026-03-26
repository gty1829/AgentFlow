"""
Simplified Trajectory Sampler for RAG synthesis
"""
import os
import copy
import json
import hashlib
import traceback
from typing import Dict, List, Optional, Any
import uuid
import bdb
import logging
import asyncio
from sandbox import format_tool_result
from .models import TrajectoryNode
from .config import SynthesisConfig
from .worker import SandboxWorker
from .utils import parse_action_xml, async_chat_completion

from sandbox.tool_schemas import get_tool_schemas

from .prompts import *

class TrajectorySampler:
    """Samples trajectory trees by exploring with LLM + tools"""

    def __init__(self, worker: SandboxWorker, config: SynthesisConfig):
        """Initialize sampler"""
        self.worker = worker
        self.config = config    # SynthesisConfig

        # Tree storage
        self.nodes: Dict[str, TrajectoryNode] = {}
        self.root_id: Optional[str] = None

        # Available tools (will be populated after worker starts)
        self.available_tools: List[Dict[str, Any]] = []

        # Per-seed action de-duplication
        self._seed_used_action_signatures: set = set()
        self._seed_used_action_signatures_ordered: List[str] = []

        self.messages_jsonl_path = "/share/project/guotianyu/AgentFlow/logs/branch_messages.jsonl"
        os.makedirs(os.path.dirname(self.messages_jsonl_path), exist_ok=True)
        self._jsonl_lock = asyncio.Lock()

        self.node_messages: Dict[str, List[Dict[str, Any]]] = {}

    async def sample_trajectory_tree(
        self,
        seed_data: Dict[str, Any],
        seed_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, TrajectoryNode]:
        """Sample a trajectory tree starting from seed data (async)"""
        if seed_kwargs is None:
            seed_kwargs = {}

        print(f"\n{'='*60}")
        print(f"Starting Trajectory Sampling")
        print(f"Seed: video_id: {seed_data['video_id']} | start_time: {seed_data['start_time']} | end_time: {seed_data['end_time']} | caption: {seed_data['caption'][:100]}...")
        if seed_kwargs:
            print(f"Kwargs: {seed_kwargs}")
        print(f"{'='*60}\n")

        # Reset tree
        self.nodes = {}
        self.root_id = None
        self._seed_used_action_signatures.clear()
        self._seed_used_action_signatures_ordered.clear()

        # Store kwargs for use in action execution
        self.seed_kwargs = seed_kwargs

        # Load tool schemas from local tools module
        self.available_tools = get_tool_schemas(
            self.config.available_tools if self.config.available_tools else None
        )
        print(f"Available tools: {[t.get('name', 'unknown') for t in self.available_tools]}")

        # Create root node
        # 创建根节点，把 seed_data 包装成根节点
        root_id = self._generate_node_id()
        root_node = TrajectoryNode(
            node_id=root_id,
            observation={"text_info":f"Starting point: {seed_data}", "mm_info": []},
            intent="Initialize exploration",
            action=None,
            parent_id=None,
            children_ids=[],
            depth=0
        )
        self.nodes[root_id] = root_node
        self.root_id = root_id

        # Explore tree asynchronously
        await self._explore_node(root_node, seed_data)

        print(f"\n✅ Sampling complete. Total nodes: {len(self.nodes)}")
        return self.nodes

    async def _explore_node(self, node: TrajectoryNode, seed_data: Dict[str, Any]):
        """Explore from a node (async breadth-first for siblings)"""
        # 如果超过最大深度，则返回
        if node.depth >= self.config.max_depth:
            node.stop_reason = "max_depth"
            return

        # 根据当前深度决定分支数
        # Determine how many children to create
        num_children = self.config.branching_factor if node.depth < self.config.depth_threshold else 1

        # 并发创建所有子节点
        # Create tasks for all children at this level
        child_tasks = []
        for i in range(num_children):
            child_tasks.append(self._create_and_explore_child(node, seed_data))
        # Execute all children concurrently
        await asyncio.gather(*child_tasks, return_exceptions=True)

    async def _create_and_explore_child(self, parent_node: TrajectoryNode, seed_data: Dict[str, Any]):
        """Create and explore a single child node"""
        max_retries = 3
        retry_wait = 0.5
        intent = ""
        action: Optional[Dict[str, Any]] = None
        observation = {"text_info": "", "mm_info": []}
        for attempt in range(max_retries + 1):
            try:
                # Generate next action
                intent, action, messages = await self._generate_next_action(parent_node, seed_data)
                if not action:
                    return

                # Execute action asynchronously
                observation = await self._execute_action(action)
                break
            except Exception as e:
                if attempt >= max_retries:
                    print(f"  ⚠️ Error exploring node: {e}")
                    traceback.print_exc()
                    return
                await asyncio.sleep(retry_wait * (2 ** attempt))

        # Create child node
        child_id = self._generate_node_id()
        child_node = TrajectoryNode(
            node_id=child_id,
            observation=observation,
            intent=intent,
            action=action,
            parent_id=parent_node.node_id,
            children_ids=[],
            depth=parent_node.depth + 1
        )

        self.nodes[child_id] = child_node
        parent_node.children_ids.append(child_id)
        print(f"parent_node.node_id: {parent_node.node_id}, len(children_ids): {len(parent_node.children_ids)}")

        self.node_messages[child_id] = messages

        node_index = len(self.nodes)
        action_data = action or {}
        print(
            f"\033[36m[#={node_index} Id={child_node.node_id} Depth={child_node.depth} "
            f"\033[33m{action_data.get('tool_name', 'unknown')}\033[36m]\033[0m:\n"
            f"{intent}"
        )
        print(f"\033[36m[params]:\n\033[0m {json.dumps(action_data.get('parameters', {}), ensure_ascii=False)}")
        preview = observation["text_info"][:1000] + "...\n\n" if len(observation["text_info"]) > 1000 else observation["text_info"]
        print(f"\033[36m[output]:\n\033[0m {preview}")

        # Recursively explore this child
        await self._explore_node(child_node, seed_data)

        # 分支结束后写 jsonl
        if len(child_node.children_ids) == 0:
            if child_node.stop_reason is None:
                child_node.stop_reason = "no_further_expansion"
            await self._dump_branch_messages_jsonl(child_node, seed_data)
        return

    async def _generate_next_action(self, node: TrajectoryNode, seed_data: Dict[str, Any]) -> tuple[str, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate next action using LLM (async)"""
        # Build context
        context = self._build_context(node, seed_data)

        # Build used actions block
        used_actions_block = self._format_used_actions_for_prompt()

        # Build prompt
        messages = self._build_exploration_prompt(context, seed_data, used_actions_block)

        try:
            # 只要保证返回的content是一个列表就没问题了
            response = await async_chat_completion(
                config=self.config.completion_config,
                messages=messages,
            )
            result = parse_action_xml(response)
            intent = result.get("intent", "")
            action = result.get("action", {})

            # 复制一份 messages，并把当轮 response 拼到后面
            messages_with_response = copy.deepcopy(messages)
            messages_with_response.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            })

            # if action and isinstance(action, dict):
            #     # Check for duplicate action
            #     sig = self._action_signature(action, intent=intent)
            #     if sig in self._seed_used_action_signatures:
            #         print(f"  ⚠️ Duplicate action detected, skipping: {sig}")
            #         return "", None, messages_with_response
            #     # Record the action signature
            #     self._seed_used_action_signatures.add(sig)
            #     self._seed_used_action_signatures_ordered.append(sig)
            return intent, action, messages_with_response

        except Exception as e:
            if isinstance(e, bdb.BdbQuit):
                raise e
            print(f"  ⚠️ LLM generation failed: {e}")
            messages_with_response = copy.deepcopy(messages)
            messages_with_response.append({
                "role": "assistant",
                "content": [{"type": "text", "text": f"LLM generation failed:: {str(e)}"}],
                "is_error": True,
            })
            return "", None, messages_with_response

    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action via worker (async)"""
        tool_name = action.get("tool_name", "")
        parameters = action.get("parameters", {})
        try:
            # Execute tool asynchronously using worker's async method
            # Pass seed_kwargs to worker
            result = await self.worker.execute_tool(tool_name, parameters, **self.seed_kwargs)

            # Use format_tool_result to format the result for agent consumption
            # This handles the new sandbox response format automatically
            formatted_result = format_tool_result(result, verbose=False)  # {"text_info": str, "mm_info": List[Dict[str, Any]]}
            return formatted_result
        except Exception as e:
            print(f"Error executing {tool_name}: {str(e)}")
            error_dict = {"text_info": f"Error executing {tool_name}: {str(e)}", "mm_info": []}
            return error_dict

    def _build_context(self, node: TrajectoryNode, seed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build context from current path"""
        context = []
        context.append({"text_info": f"Starting point: {seed_data}", "mm_info": []})
        # Trace back to root
        path = []
        current = node
        while current.parent_id:
            path.append(current)
            current = self.nodes[current.parent_id]
        path.reverse()

        # Format path
        for i, n in enumerate(path, 1):
            info_dict = {}
            text_info = ""
            text_info += f"Step {i}:\n"
            text_info += f"  Intent: {n.intent}\n"
            if n.action:
                text_info += f"  Action: {n.action.get('tool_name', 'unknown')}\n"
                text_info += f"  Parameters: {json.dumps(n.action.get('parameters', {}), ensure_ascii=False)}\n"
                info_dict["mm_info"] = n.observation["mm_info"] # mm_info: List[List[Dict[str, Any]]]
            else:
                info_dict["mm_info"] = []  # 用于占位，text_info和mm_info对齐

            # TODO: 去掉observation的压缩
            if len(path) <= 2 or i > len(path) - 2:
                # 最近的两个加载完整信息，其他加载部分信息
                text_info += f"  Observation: {n.observation['text_info']}\n\n"
            else:
                obs_preview = n.observation["text_info"][:500] + "..." if len(n.observation["text_info"]) > 500 else n.observation["text_info"]
                text_info += f"  Observation: {obs_preview}\n\n"
            info_dict["text_info"] = text_info
            context.append(info_dict)

        return context

    def _build_exploration_prompt(self, context: List[Dict[str, Any]], seed_data: Dict[str, Any], used_actions_block: str = "") -> List[Dict[str, Any]]:
        """Build prompt for exploration, return a list of messages"""
        # Generate detailed tool descriptions with parameters
        tool_descriptions = []
        for tool in self.available_tools:
            desc = f"\n{len(tool_descriptions) + 1}. {tool['name']}: {tool['description']}\n"
            desc += "   Parameters:\n"
            for param in tool.get("parameters", []):
                param_type = param.get("type", "string")
                if param_type == "array":
                    param_type = f"array of {param.get('array_type', 'string')}"
                required_str = " (required)" if param.get("required", False) else " (optional)"
                desc += f"   - {param['name']} ({param_type}){required_str}: {param.get('description', '')}\n"
            tool_descriptions.append(desc)

        tool_descriptions_str = "\n".join(tool_descriptions)
        
        messages = []

        messages.append({"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}]})
        messages.append({"role": "user", "content": []})

        prompt = f"""
[Starting Point Information]
Content: {seed_data}"""

        # if self.config.seed_description:
        #     prompt += f"\nDescription: {SEED_DESCRIPTION}"

        # prompt += EXPLORATION_GOAL
        if used_actions_block:
            prompt += USED_ACTIONS_BLOCK_PREFIX + "\n" + used_actions_block + "\n\n"

        if self.config.sampling_tips:
            prompt += SAMPLING_TIPS

        # prompt += "[Current History Trajectory]\n"
        messages[1]['content'].append({"type": "text", "text": prompt})

        for idx, ctx in enumerate(context):
            text_info_ctx = ctx["text_info"]
            mm_info_ctx = ctx["mm_info"]
            
            for mm_info_dict in mm_info_ctx:
                b64, fmt, media_type = mm_info_dict["base64_data"], mm_info_dict["fmt"], mm_info_dict["media_type"]
                mm_info = {"type": f"{media_type}_url", f"{media_type}_url": {"url": f"data:{media_type}/{fmt};base64,{b64}"}}
                messages[1]['content'].append(mm_info)
            
            if idx != len(context) - 1:
                text_info = {"type": "text", "text": text_info_ctx}
            else:
    #             prompt = f"""[Current Turn]:
    # {text_info_ctx}

    # [Current Observation]:
    # {text_info_ctx.split('Observation:')[-1] if 'Observation:' in text_info_ctx else 'Starting exploration'}

    # [Available Tools]:
    # {tool_descriptions_str}
    # """                       
                prompt = f"""[Last Turn]: 
    {text_info_ctx}

    [Available Tools]:
    {tool_descriptions_str}
    """       
                text_info = {"type": "text", "text": prompt}           
            messages[1]['content'].append(text_info)


        prompt = PROMPT_SUFFIX
        messages[1]['content'].append({"type": "text", "text": prompt})
        return messages

    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        return f"node_{uuid.uuid4().hex[:8]}"

    def _action_signature(self, action: Dict[str, Any], intent: Optional[str] = None) -> str:
        """
        Canonical signature for action de-duplication within a seed.
        Format: tool_name + canonicalized JSON(parameters)
        """
        tool_name = ""
        parameters: Any = {}
        try:
            tool_name = str(action.get("tool_name", ""))
            parameters = action.get("parameters", {}) if isinstance(action, dict) else {}
        except Exception:
            tool_name = ""
            parameters = {}

        def _shrink_obj(obj: Any, max_str_len: int = 160) -> Any:
            try:
                if isinstance(obj, str):
                    if len(obj) <= max_str_len:
                        return obj
                    h = hashlib.md5(obj.encode("utf-8")).hexdigest()[:10]
                    return f"<str:{len(obj)}:md5:{h}>"
                if isinstance(obj, dict):
                    return {str(k): _shrink_obj(v, max_str_len=max_str_len) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_shrink_obj(v, max_str_len=max_str_len) for v in obj]
                return obj
            except Exception:
                return str(obj)

        try:
            safe_params = _shrink_obj(parameters)
            params_str = json.dumps(safe_params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            params_str = str(parameters)

        return f"{tool_name}({params_str})"

    def _format_used_actions_for_prompt(self, max_items: int = 40, max_line_chars: int = 300) -> str:
        """Format executed actions list for inclusion in prompt (truncate to keep prompt bounded)."""
        if not self._seed_used_action_signatures_ordered:
            return "None yet."

        items = self._seed_used_action_signatures_ordered[-max_items:]
        lines = []
        for i, sig in enumerate(items, 1):
            s = sig
            if len(s) > max_line_chars:
                s = s[:max_line_chars] + "...(truncated)"
            lines.append(f"- {i}. {s}")

        omitted = len(self._seed_used_action_signatures_ordered) - len(items)
        if omitted > 0:
            lines.insert(0, f"(Showing last {len(items)} actions; {omitted} earlier actions omitted but still forbidden.)")

        return "\n".join(lines)

    async def _dump_branch_messages_jsonl(self, leaf_node: TrajectoryNode, seed_data: Dict[str, Any]) -> None:
        """Dump all successful messages along one finished branch into a shared jsonl file."""
        branch_node_ids = []
        current = leaf_node

        while current is not None:
            branch_node_ids.append(current.node_id)
            if current.parent_id is None:
                break
            current = self.nodes[current.parent_id]

        branch_node_ids.reverse()

        branch_messages = []
        branch_actions = []

        for node_id in branch_node_ids:
            if node_id in self.node_messages:
                branch_messages.append({
                    "node_id": node_id,
                    "messages": self.node_messages[node_id]
                })

            node = self.nodes[node_id]
            if node.action:
                branch_actions.append({
                    "node_id": node_id,
                    "intent": node.intent,
                    "action": node.action,
                    "observation": node.observation["text_info"]
                })

        record = {
            "seed_data": seed_data,
            "leaf_node_id": leaf_node.node_id,
            "leaf_stop_reason": leaf_node.stop_reason,
            "depth": leaf_node.depth,
            "branch_node_ids": branch_node_ids,
            "branch_actions": branch_actions,
            "branch_messages": branch_messages
        }

        async with self._jsonl_lock:
            with open(self.messages_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

