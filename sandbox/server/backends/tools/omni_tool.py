# sandbox/server/backends/tools/omni_tool.py
"""
OMNI API Tools

Provides tools for OMNI QA
"""

import base64
import json
import logging
import os
import time
import asyncio
import requests
from typing import Any, Dict, List, Tuple, Union

from . import register_api_tool
from ..error_codes import ErrorCode
from .base_tool import BaseApiTool, ToolBusinessError

logger = logging.getLogger("OmniTool")

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
    top_p: float,
    top_k: int,
    response_format: Dict[str, Any],
    extra_body: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        **extra_body,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if response_format is not None:
        payload["response_format"] = response_format

    return payload


def _build_gemini_payload(
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
    extra_body: Dict[str, Any],
) -> Dict[str, Any]:
    generation_config: Dict[str, Any] = {
        "temperature": temperature,
    }

    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    if top_p is not None:
        generation_config["topP"] = top_p
    if top_k is not None:
        generation_config["topK"] = top_k

    payload: Dict[str, Any] = {
        "contents": _messages_to_gemini_contents(messages),
        "generationConfig": generation_config,
        **extra_body,
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
    api_url = completion_config.api_url
    api_key = completion_config.api_key
    connect_timeout = completion_config.connect_timeout
    read_timeout = completion_config.read_timeout
    temperature = completion_config.temperature
    max_tokens = completion_config.max_tokens
    top_p = completion_config.top_p
    top_k = completion_config.top_k
    response_format = completion_config.response_format
    extra_body = completion_config.extra_body or {}
    max_retry = completion_config.max_retry
    retry_interval = completion_config.retry_interval

    if not api_url:
        raise ValueError("completion_config.api_url is required")

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
            top_p=top_p,
            top_k=top_k,
            extra_body=extra_body,
        )
    else:
        payload = _build_openai_payload(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            response_format=response_format,
            extra_body=extra_body,
        )

    last_error = None

    for attempt in range(max_retry):
        try:
            resp = requests.post(
                api_url,
                headers=headers,
                json=payload,
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

    
def _encode_file_to_base64(file_path: str) -> Tuple[str, str, str]:
    """
    Read a media file and encode it to base64.

    Returns:
        (base64_data, media_format, media_type)
    """
    ext = file_path.rsplit(".", 1)[-1].lower()

    format_map = {
        # image
        "jpg": ("jpeg", "image"),
        "jpeg": ("jpeg", "image"),
        "png": ("png", "image"),
        "webp": ("webp", "image"),

        # audio
        "mp3": ("mp3", "audio"),
        "mpeg": ("mp3", "audio"),
        "mpga": ("mp3", "audio"),
        "wav": ("wav", "audio"),
        "flac": ("flac", "audio"),
        "m4a": ("m4a", "audio"),
        "ogg": ("ogg", "audio"),

        # video
        "mp4": ("mp4", "video"),
        "mov": ("mov", "video"),
        "avi": ("avi", "video"),
        "mkv": ("mkv", "video"),
        "webm": ("webm", "video"),
    }

    if ext not in format_map:
        raise ValueError(f"Unsupported media format: {ext}")

    media_format, media_type = format_map[ext]

    with open(file_path, "rb") as f:
        raw = f.read()

    b64 = base64.b64encode(raw).decode("utf-8")
    return b64, media_format, media_type

class SearchTool(BaseApiTool):
    """
    用于读取所有包含待搜索关键词的caption
    """
    def __init__(self):
        super().__init__(tool_name="omni:search_caption", resource_type="omni")

    def single_search(
        self,
        key_word: str,
        video_clips_info: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        single_search_result = []
        for video_clip_info in video_clips_info:
            caption = video_clip_info.get('caption')
            if key_word.lower() in caption.lower():
                single_search_result.append(video_clip_info)

        return single_search_result

    async def execute(
        self,
        key_words: Union[str, List[str]],
        max_search_results: int,
        **kwargs,
    ):
        """
        Search for keywords in all captions
        
        Args:
            key_words: Keywords (string or list)
            max_search_results: Maximum number of search results
        """
        if isinstance(key_words, str):       
            key_words = [key_words]
        
        # jsonl中的每一个视频info: video_path, video_clips_info(instruction会去掉放到seeds.jsonl中), output_dir(视频输出目录)
        # 都会保存为一个独立的json文件
        video_info_path = kwargs.get('video_info_path')   

        if not video_info_path:
            raise ToolBusinessError("video_info_path must be provided in kwargs", ErrorCode.EXECUTION_ERROR)
        
        with open(video_info_path, 'r', encoding='utf-8') as f:
            original_video_info = json.load(f)
            video_clips_info = original_video_info.get('video_clips_info')
        
        search_result = ""
        for key_word in key_words:
            single_search_result = self.single_search(key_word, video_clips_info)
            result_num = len(single_search_result)
            if result_num > max_search_results:
                for subelement in single_search_result[max_search_results:]:
                    single_search_result.remove(subelement)

            if result_num > max_search_results:
                search_result += f"A Caption search for `{key_word}` found {result_num} results. To shorten response, the first {max_search_results} results are listed below:\n\n{single_search_result}"
            else:
                search_result += f"A Caption search for `{key_word}` found {result_num} results:\n\n{single_search_result}"
            search_result += "\n\n=============================\n\n"
        search_result = search_result.strip("\n\n=============================\n\n")

        return {"result": search_result}

class ReadTool(BaseApiTool):
    """
    Read multimodal data
    """
    def __init__(self):
        super().__init__(tool_name="omni:read", resource_type="omni")
    
    def _single_read(
        self,
        mm_path: str,
    ):
        base64_data, fmt, media_type = _encode_file_to_base64(mm_path)
        return (base64_data, fmt, media_type)
    
    async def execute(
        self,
        mm_paths: Union[str, List[str]],
        **kwargs,
    ):
        
        if isinstance(mm_paths, str):
            mm_paths = [mm_paths]
        
        mm_result = []
        read_result = ""
        for mm_path in mm_paths:
            base64_data, fmt, media_type = self._single_read(mm_path)
            mm_result.append({"base64_data": base64_data, "fmt": fmt, "media_type": media_type})
            read_result += f"A {media_type} read for `{mm_path}` found"
            read_result += "\n\n=============================\n\n"
        read_result = read_result.strip("\n\n=============================\n\n")
        return {
            "result": read_result,
            "mm_result": mm_result
        }
    
class ExtractClipTool(BaseApiTool):
    """
    Crop multimodal data: audio / video / audio-video
    """
    def __init__(self):
        super().__init__(tool_name="omni:extract_clip", resource_type="omni")

    @staticmethod
    def _normalize_time(value: Union[int, float, str]) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError(f"Invalid time value: {value}")
    
    @staticmethod
    def _safe_mkdir(path: str):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _build_output_path(
        output_dir: str,
        mm_path: str,
        idx: int,
        start_time: Union[int, float, str],
        end_time: Union[int, float, str],
        mode: str,
    ) -> str:
        base_name = os.path.splitext(os.path.basename(mm_path))[0]

        if mode == "audio":
            ext = ".wav"
        else:
            ext = ".mp4"

        return os.path.join(
            output_dir,
            f"{base_name}_{mode}_{idx}_{str(start_time).replace(':', '-')}_{str(end_time).replace(':', '-')}{ext}"
        )
    
    async def _run_ffmpeg(self, command: List[str]) -> Dict[str, Any]:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return {
            "returncode": process.returncode,
            "stdout": stdout.decode("utf-8", errors="ignore"),
            "stderr": stderr.decode("utf-8", errors="ignore"),
        }

    async def _extract_single_clip(
        self,
        mm_path: str,
        start_time: Union[int, float, str],
        end_time: Union[int, float, str],
        output_path: str,
        mode: str,
    ) -> Dict[str, Any]:
        if not os.path.exists(mm_path):
            raise ToolBusinessError(
                f"mm_path does not exist: {mm_path}",
                ErrorCode.EXECUTION_ERROR,
            )

        start_time = self._normalize_time(start_time)
        end_time = self._normalize_time(end_time)

        # 核心逻辑：
        # av    -> 默认输出音视频
        # audio -> 只映射音频流
        # video -> 禁掉音频，仅保留视频流
        if mode == "av":
            command = [
                "ffmpeg",
                "-y",
                "-ss", start_time,
                "-to", end_time,
                "-i", mm_path,
                "-c", "copy",
                output_path,
            ]
        elif mode == "audio":
            command = [
                "ffmpeg",
                "-y",
                "-ss", start_time,
                "-to", end_time,
                "-i", mm_path,
                "-vn",
                "-c:a", "aac",
                output_path,
            ]
        elif mode == "video":
            command = [
                "ffmpeg",
                "-y",
                "-ss", start_time,
                "-to", end_time,
                "-i", mm_path,
                "-an",
                "-c:v", "libx264",
                output_path,
            ]
        else:
            raise ToolBusinessError(
                f"Unsupported mode: {mode}. Choose from ['av', 'audio', 'video']",
                ErrorCode.PARAM_ERROR,
            )

        exec_result = await self._run_ffmpeg(command)

        if exec_result["returncode"] != 0:
            raise ToolBusinessError(
                f"ffmpeg crop failed: {exec_result['stderr']}",
                ErrorCode.EXECUTION_ERROR,
            )

        return {
            "mm_path": mm_path,
            "start_time": start_time,
            "end_time": end_time,
            "mode": mode,
            "output_path": output_path,
        }

    async def execute(
        self,
        mm_path: str,
        clips: List[Dict[str, Union[int, float, str]]],
        mode: str = "av",
        **kwargs,
    ):
        """
        Args:
            mm_path: 输入视频路径
            clips: [{"start_time": 1.2, "end_time": 5.8}, ...]
            mode: av / audio / video
        """
        output_dir = kwargs.get('output_dir')    # 输出目录
        if not output_dir:
            raise ToolBusinessError("output_dir must be provided in kwargs", ErrorCode.EXECUTION_ERROR)

        if not mm_path:
            raise ToolBusinessError("mm_path is required", ErrorCode.PARAM_ERROR)

        if not clips or not isinstance(clips, list):
            raise ToolBusinessError("clips must be a non-empty list", ErrorCode.PARAM_ERROR)

        self._safe_mkdir(output_dir)

        mm_result = []
        text_result = ""

        for idx, clip in enumerate(clips, start=1):
            start_time = clip.get("start_time")
            end_time = clip.get("end_time")

            if start_time is None or end_time is None:
                raise ToolBusinessError(
                    f"Each clip must contain start_time and end_time, got: {clip}",
                    ErrorCode.PARAM_ERROR,
                )

            output_path = self._build_output_path(
                output_dir=output_dir,
                mm_path=mm_path,
                idx=idx,
                start_time=start_time,
                end_time=end_time,
                mode=mode,
            )

            single_result = await self._extract_single_clip(
                mm_path=mm_path,
                start_time=start_time,
                end_time=end_time,
                output_path=output_path,
                mode=mode,
            )

            base64_data, fmt, media_type = _encode_file_to_base64(output_path)
            mm_result.append({"base64_data": base64_data, "fmt": fmt, "media_type": media_type})
            text_result += f"Clip {idx}: mode={mode}, `{start_time}` -> `{end_time}`, saved to `{output_path}`"
            text_result += "\n\n=============================\n\n"
        text_result = text_result.strip("\n\n=============================\n\n")

        return {
            "result": text_result,
            "mm_result": mm_result,
        }

class StatelessCodeExecutionTool(BaseApiTool):
    """
    Execute code
    """
    def __init__(self):
        super().__init__(tool_name="omni:stateless_code_execute", resource_type="omni")
    
    pass

# Register tools
search = register_api_tool(
    name="omni:search",
    config_key="omni",
    description="Search for keywords in all captions"
)(SearchTool())

read = register_api_tool(
    name="omni:read",
    config_key="omni",
    description="Read multimodal data"
)(ReadTool())

extract_clip = register_api_tool(
    name="omni:extract_clip",
    config_key="omni",
    description="Extract clip from multimodal data"
)(ExtractClipTool())

stateless_code_execute = register_api_tool(
    name="omni:stateless_code_execute",
    config_key="omni",
    description="Execute code"
)(StatelessCodeExecutionTool())