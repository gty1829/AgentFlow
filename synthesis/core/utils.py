import asyncio
import json
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, Tuple, Type, List
import pdb

import random
import requests
import openai


def create_openai_client(api_key: str, base_url: str) -> openai.OpenAI:
    if not api_key:
        raise ValueError("Missing api_key in synthesis config")
    if not base_url:
        raise ValueError("Missing base_url in synthesis config")
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def extract_json_object(text: str) -> str:
    """Extract the first complete JSON object from text (forward scan)."""
    if not text:
        return text
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def extract_xml_block(text: str, root_tag: str = "response") -> str:
    """Extract the first XML block with the given root tag."""
    if not text:
        return text
    pattern = rf"<{root_tag}\b[^>]*>.*?</{root_tag}>"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(0) if match else text


def parse_action_xml(text: str) -> Dict[str, Any]:
    """Parse XML into {intent, action:{tool_name, parameters}}."""
    def _find_tag(raw: str, tag: str) -> str:
        match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", raw, flags=re.DOTALL)
        return (match.group(1).strip() if match else "")

    xml_text = extract_xml_block(text, root_tag="response")
    intent = ""
    tool_name = ""
    parameters: Dict[str, Any] = {}
    if "<response" in xml_text:
        root = ET.fromstring(xml_text)
        intent = (root.findtext("intent") or "").strip()
        action_el = root.find("action")
        if action_el is not None:
            tool_name = (action_el.findtext("tool_name") or "").strip()
            params_text = (action_el.findtext("parameters") or "").strip()
        else:
            tool_name = (root.findtext("tool_name") or "").strip()
            params_text = (root.findtext("parameters") or "").strip()
    else:
        intent = _find_tag(text, "intent")
        tool_name = _find_tag(text, "tool_name")
        params_text = _find_tag(text, "parameters")

    if params_text:
        try:
            parameters = json.loads(params_text)
        except Exception:
            parameters = {}
    return {
        "intent": intent,
        "action": {
            "tool_name": tool_name,
            "parameters": parameters
        }
    }


def chat_completion(
    client: openai.OpenAI,
    *,
    max_retries: int = 3,
    retry_wait: float = 0.5,
    retry_backoff: float = 2.0,
    retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    **kwargs: Any
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except retry_exceptions:
            if attempt >= max_retries:
                raise
            time.sleep(retry_wait * (retry_backoff ** attempt))


# async def async_chat_completion(
#     client: openai.OpenAI,
#     *,
#     max_retries: int = 3,
#     retry_wait: float = 0.5,
#     retry_backoff: float = 2.0,
#     retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
#     **kwargs: Any
# ) -> Any:
#     loop = asyncio.get_event_loop()
#     for attempt in range(max_retries + 1):
#         try:
#             return await loop.run_in_executor(
#                 None,
#                 lambda: client.chat.completions.create(**kwargs)
#             )
#         except retry_exceptions as e:
#             if attempt >= max_retries:
#                 raise
#             await asyncio.sleep(retry_wait * (retry_backoff ** attempt))


async def async_chat_completion(
    config: Any,
    messages: List[Dict[str, Any]],
) -> Any:
    loop = asyncio.get_running_loop()
    model = config.get("model", "")

    return await loop.run_in_executor(
        None,
        lambda: completion(
            messages=messages,
            model=model,
            completion_config=config,
        )
    )

class CompletionError(Exception):
    pass


def _is_gemini_model(model: str) -> bool:
    return "gemini" in model.lower() or "mgg" in model.lower() or "ep-20260317031559-av3m2" in model.lower()


def _messages_to_gemini_contents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI-style messages to Gemini-style contents.
    Supports:
      - string content
      - list content with text / image_url / video_url / audio_url
    """
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        raw_content = msg.get("content", "")

        gemini_role = "model" if role == "assistant" else "user"
        parts: List[Dict[str, Any]] = []

        if isinstance(raw_content, str):
            parts.append({"text": raw_content})

        elif isinstance(raw_content, list):
            for item in raw_content:
                if not isinstance(item, dict):
                    parts.append({"text": str(item)})
                    continue

                item_type = item.get("type")

                if item_type == "text":
                    parts.append({"text": item.get("text", "")})

                elif item_type in ("image_url", "video_url", "audio_url"):
                    media_obj = item.get(item_type, {}) or {}
                    url = media_obj.get("url", "")
                    if not isinstance(url, str) or not url.startswith("data:"):
                        parts.append({"text": f"[unsupported media url]"})
                        continue

                    try:
                        header, b64_data = url.split(",", 1)
                        mime_type = header[len("data:"):].split(";")[0]
                    except Exception:
                        parts.append({"text": "[invalid media data url]"})
                        continue

                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_data,
                            }
                        }
                    )

                else:
                    parts.append({"text": str(item)})

        else:
            parts.append({"text": str(raw_content)})

        contents.append(
            {
                "role": gemini_role,
                "parts": parts,
            }
        )

    return contents


def _build_openai_payload(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    #top_p: float,
    # top_k: int,
    # response_format: Dict[str, Any],
    # extra_body: Dict[str, Any],
) -> Dict[str, Any]:
    seed = random.randint(1, 100000)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        # **extra_body,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    # if top_p is not None:
    #     payload["top_p"] = top_p
    # if top_k is not None:
    #     payload["top_k"] = top_k
    # if response_format is not None:
    #     payload["response_format"] = response_format

    return payload


def _build_gemini_payload(
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    # top_p: float,
    # top_k: int,
    # extra_body: Dict[str, Any],
) -> Dict[str, Any]:
    generation_config: Dict[str, Any] = {
        "temperature": temperature,
    }

    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    # if top_p is not None:
    #     generation_config["topP"] = top_p
    # if top_k is not None:
    #     generation_config["topK"] = top_k

    payload: Dict[str, Any] = {
        "contents": _messages_to_gemini_contents(messages),
        "generationConfig": generation_config,
        # **extra_body,
    }
    return payload


def _parse_openai_response(data: Dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise CompletionError(f"Invalid OpenAI response format: {data}") from e

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts).strip()

    return str(content)


def _parse_gemini_response(data: Dict[str, Any]) -> str:
    try:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""

        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []

        text_parts = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])

        return "\n".join(text_parts).strip()
    except Exception as e:
        raise CompletionError(f"Invalid Gemini response format: {data}") from e


def completion(
    messages: List[Dict[str, Any]],
    model: str,
    completion_config: Dict,   # completion config
) -> str:
    api_url = completion_config.get("api_url", "")
    api_key = completion_config.get("api_key", "")
    connect_timeout = completion_config.get("connect_timeout", 300)
    read_timeout = completion_config.get("read_timeout", 180)
    temperature = completion_config.get("temperature", 0.7)
    max_tokens = completion_config.get("max_tokens", 32768)
    # top_p = completion_config.get("top_p", 0.95)
    # top_k = completion_config.get("top_k", 20)
    # response_format = completion_config.get("response_format", {})
    # extra_body = completion_config.get("extra_body", {})
    max_retry = completion_config.get("max_retry", 3)
    retry_interval = completion_config.get("retry_interval", 0.5)

    if not api_url:
        raise ValueError("api_url is required")

    is_gemini = _is_gemini_model(model)

    headers = {
        "Content-Type": "application/json",
    }

    # 默认先按 Bearer 兼容；如果你后面遇到 query ?key=... 的 Gemini 网关，再单独改
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if is_gemini:
        payload = _build_gemini_payload(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # top_p=top_p,
            # top_k=top_k,
            # extra_body=extra_body,
        )
    else:
        payload = _build_openai_payload(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            # top_p=top_p,
            # top_k=top_k,
            # response_format=response_format,
            # extra_body=extra_body,
        )

    last_error = None

    for attempt in range(max_retry):
        try:
            resp = requests.post(
                api_url,
                headers=headers,
                data=json.dumps(payload),
                # json=json.dumps(payload),
                timeout=(connect_timeout, read_timeout),
            )
            resp.raise_for_status()

            data = resp.json()

            if is_gemini:
                return _parse_gemini_response(data)
            return _parse_openai_response(data)

        except Exception as e:
            last_error = e
            if attempt < max_retry - 1:
                time.sleep(retry_interval)
            else:
                raise CompletionError(
                    f"Completion failed after {max_retry} retries: {e}"
                ) from e

    raise CompletionError(f"Completion failed: {last_error}")
