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
from .code_executor import PersistentPythonExecutor

logger = logging.getLogger("OmniTool")
    
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
    根据关键词搜索 caption，并返回距离指定 video_id 最近的若干条结果。
    注意：key_words 中的每个 key_word 会单独搜索。
    """
    def __init__(self):
        super().__init__(tool_name="omni:search", resource_type="omni")

    def _find_target_index(
        self,
        video_id: str,
        video_clips_info: List[Dict[str, Any]],
    ) -> int:
        for idx, clip in enumerate(video_clips_info):
            if str(clip.get("video_id")) == str(video_id):
                return idx
        return -1

    def _match_single_keyword(
        self,
        caption: str,
        key_word: str,
    ) -> bool:
        if not caption:
            return False
        return key_word.lower() in caption.lower()

    def search_nearest_matches_for_single_keyword(
        self,
        video_id: str,
        key_word: str,
        video_clips_info: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        target_index = self._find_target_index(video_id, video_clips_info)
        if target_index == -1:
            raise ToolBusinessError(
                f"video_id `{video_id}` not found",
                ErrorCode.EXECUTION_ERROR
            )

        matched_results = []
        for idx, clip in enumerate(video_clips_info):
            if idx == target_index:
                continue
            caption = clip.get("caption", "")
            if self._match_single_keyword(caption, key_word):
                matched_results.append({
                    "distance": abs(idx - target_index),
                    "index": idx,
                    "video_clip_info": clip,
                })

        # 按距离排序；距离相同则按索引排序
        matched_results.sort(key=lambda x: (x["distance"], x["index"]))

        # 这里只返回全部结果，不做截断
        result = [dict(item["video_clip_info"]) for item in matched_results]
        return result

    async def execute(
        self,
        key_word: str,
        max_search_results: int,
        video_id: str,
        **kwargs,
    ):
        """
        Search captions by keywords and return the nearest matched clips to the target video_id.

        Args:
            key_words: 关键词，可以是字符串或字符串列表。每个关键词会单独搜索。
            max_search_results: 每个关键词展示结果上限
            video_id: 目标视频片段id
        """
        if max_search_results <= 0:
            raise ToolBusinessError(
                "max_search_results must be greater than 0",
                ErrorCode.EXECUTION_ERROR
            )
        
        if not isinstance(key_word, str) or not key_word.strip():
            raise ToolBusinessError(
                "key_word must not be empty",
                ErrorCode.EXECUTION_ERROR
            )

        key_word = key_word.strip()
        key_word = key_word.lower()

        video_info_path = kwargs.get("video_info_path")
        if not video_info_path:
            raise ToolBusinessError(
                "video_info_path must be provided in kwargs",
                ErrorCode.EXECUTION_ERROR
            )

        with open(video_info_path, "r", encoding="utf-8") as f:
            original_video_info = json.load(f)
            video_clips_info = original_video_info.get("video_clips_info")

        if not video_clips_info:
            raise ToolBusinessError(
                "video_clips_info is empty or missing in json file",
                ErrorCode.EXECUTION_ERROR
            )

        search_result = ""

        single_search_result = self.search_nearest_matches_for_single_keyword(
            video_id=video_id,
            key_word=key_word,
            video_clips_info=video_clips_info,
        )

        result_num = len(single_search_result)

        if result_num > max_search_results:
            displayed_result = single_search_result[:max_search_results]
            search_result += (
                f"A Caption search for `{key_word}` found {result_num} results. "
                f"To shorten response, the first {max_search_results} results are listed below:\n\n"
                f"{displayed_result}"
            )
        else:
            search_result += (
                f"A Caption search for `{key_word}` found {result_num} results:\n\n"
                f"{single_search_result}"
            )

        search_result += "\n\n=============================\n\n"
        search_result = search_result.strip("\n\n=============================\n\n")

        return {
            "result": search_result,
            "mm_result": [],
        }


# class SearchTool(BaseApiTool):
#     """
#     用于读取所有包含待搜索关键词的caption
#     """
#     def __init__(self):
#         super().__init__(tool_name="omni:search", resource_type="omni")

#     def single_search(
#         self,
#         key_word: str,
#         video_clips_info: List[Dict[str, Any]],
#     ) -> List[Dict[str, Any]]:
#         single_search_result = []
#         for video_clip_info in video_clips_info:
#             caption = video_clip_info.get('caption')
#             if key_word.lower() in caption.lower():
#                 single_search_result.append(video_clip_info)

#         return single_search_result

#     async def execute(
#         self,
#         key_words: Union[str, List[str]],
#         max_search_results: int,
#         **kwargs,
#     ):
#         """
#         Search for keywords in all captions
        
#         Args:
#             key_words: Keywords (string or list)
#             max_search_results: Maximum number of search results
#         """
#         if isinstance(key_words, str):       
#             key_words = [key_words]
        
#         # jsonl中的每一个视频info: video_path, video_clips_info(instruction会去掉放到seeds.jsonl中), output_dir(视频输出目录)
#         # 都会保存为一个独立的json文件
#         video_info_path = kwargs.get('video_info_path')   

#         if not video_info_path:
#             raise ToolBusinessError("video_info_path must be provided in kwargs", ErrorCode.EXECUTION_ERROR)
        
#         with open(video_info_path, 'r', encoding='utf-8') as f:
#             original_video_info = json.load(f)
#             video_clips_info = original_video_info.get('video_clips_info')
        
#         search_result = ""
#         for key_word in key_words:
#             single_search_result = self.single_search(key_word, video_clips_info)
#             result_num = len(single_search_result)
#             if result_num > max_search_results:
#                 for subelement in single_search_result[max_search_results:]:
#                     single_search_result.remove(subelement)

#             if result_num > max_search_results:
#                 search_result += f"A Caption search for `{key_word}` found {result_num} results. To shorten response, the first {max_search_results} results are listed below:\n\n{single_search_result}"
#             else:
#                 search_result += f"A Caption search for `{key_word}` found {result_num} results:\n\n{single_search_result}"
#             search_result += "\n\n=============================\n\n"
#         search_result = search_result.strip("\n\n=============================\n\n")

#         return {"result": search_result, "mm_result": []}

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
            if not os.path.exists(mm_path):
                read_result += f"File `{mm_path}` does not exist"
                read_result += "\n\n=============================\n\n"
                continue
                
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
    async def _get_media_duration(mm_path: str) -> float:
        process = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            mm_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise ToolBusinessError(
                f"ffprobe failed: {stderr.decode('utf-8', errors='ignore')}",
                ErrorCode.EXECUTION_ERROR,
            )

        try:
            return float(stdout.decode("utf-8", errors="ignore").strip())
        except ValueError:
            raise ToolBusinessError(
                f"Unable to parse media duration for: {mm_path}",
                ErrorCode.EXECUTION_ERROR,
            )

    @staticmethod
    def _build_output_path(
        output_dir: str,
        mm_path: str,
        start_time: Union[int, float, str],
        end_time: Union[int, float, str],
        mode: str,
    ) -> str:
        base_name = os.path.splitext(os.path.basename(mm_path))[0]

        if mode == "audio":
            ext = ".wav"
        else:
            ext = ".mp4"

        start_str = str(start_time).replace(':', '-').replace('.', '_')
        end_str = str(end_time).replace(':', '-').replace('.', '_')

        return os.path.join(
            output_dir,
            f"{base_name}_{mode}_start_time_{start_str}-end_time_{end_str}{ext}"
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
        output_dir: str,
        mode: str,
    ) -> Dict[str, Any]:
        if not os.path.exists(mm_path):
            raise ToolBusinessError(
                f"mm_path does not exist: {mm_path}",
                ErrorCode.EXECUTION_ERROR,
            )

        start_time = self._normalize_time(start_time)
        end_time = self._normalize_time(end_time)

        try:
            start_time_float = float(start_time)
            end_time_float = float(end_time)
        except ValueError:
            raise ToolBusinessError(
                f"start_time and end_time must be numeric or numeric strings, got start_time={start_time}, end_time={end_time}",
                ErrorCode.PARAM_ERROR,
            )

        if start_time_float >= end_time_float:
            raise ToolBusinessError(
                f"start_time must be smaller than end_time, got start_time={start_time}, end_time={end_time}",
                ErrorCode.PARAM_ERROR,
            )

        duration = await self._get_media_duration(mm_path)

        if end_time_float > duration:
            end_time_float = duration
            end_time = str(end_time_float)

        if start_time_float >= end_time_float:
            raise ToolBusinessError(
                f"start_time must be smaller than effective end_time, got start_time={start_time}, end_time={end_time}",
                ErrorCode.PARAM_ERROR,
            )

        # 在时间校正之后再生成 output_path
        output_path = self._build_output_path(
            output_dir=output_dir,
            mm_path=mm_path,
            start_time=start_time,
            end_time=end_time,
            mode=mode,
        )

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

            # output_path = self._build_output_path(
            #     output_dir=output_dir,
            #     mm_path=mm_path,
            #     idx=idx,
            #     start_time=start_time,
            #     end_time=end_time,
            #     mode=mode,
            # )

            single_result = await self._extract_single_clip(
                mm_path=mm_path,
                start_time=start_time,
                end_time=end_time,
                output_dir=output_dir,
                mode=mode,
            )

            base64_data, fmt, media_type = _encode_file_to_base64(single_result['output_path'])
            mm_result.append({"base64_data": base64_data, "fmt": fmt, "media_type": media_type})
            # text_result += f"Clip {idx}: mode={mode}, `{start_time}` -> `{end_time}`, saved to `{output_path}`"
            text_result += (
                f"Clip {idx}: mode={mode}, "
                f"`{single_result['start_time']}` -> `{single_result['end_time']}`, "
                f"saved to `{single_result['output_path']}`"
            )
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
        super().__init__(tool_name="omni:run_python", resource_type="omni")

    async def execute(
        self,
        code: str,
        timeout: float = 5.0,
        **kwargs,
    ):
        executor = PersistentPythonExecutor()

        run_result = executor.run(code, timeout=timeout)
        ok = run_result.get("ok", False)
        stdout = run_result.get("stdout", "").strip()
        stderr = run_result.get("stderr", "").strip()
        error = run_result.get("error", "")

        executor.close()

        text_result = ""
        if ok:
            text_result += f"Code execution succeeded:\n\n{stdout}"
        else:
            text_result += f"Code execution failed:\n\n{stderr}\n\n{error}"
        text_result = text_result.strip("\n\n=============================\n\n")

        mm_result = []

        mm_exts = [
            'jpg', 'jpeg', 'png', 'webp',
            'mp3', 'mpeg', 'mpga', 'wav', 'flac', 'm4a', 'ogg',
            'mp4', 'mov', 'avi', 'mkv', 'webm',
        ]

        if os.path.exists(stdout):
            ext = stdout.rsplit(".", 1)[-1].lower()
            if ext in mm_exts:
                base64_data, fmt, media_type = _encode_file_to_base64(stdout)
                mm_result.append({"base64_data": base64_data, "fmt": fmt, "media_type": media_type})
        return {
            "result": text_result,
            "mm_result": mm_result,
        }

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

# stateless_code_execute = register_api_tool(
#     name="omni:run_python",
#     config_key="omni",
#     description="Execute code"
# )(StatelessCodeExecutionTool())