"""
Omni Tool Schemas

This module defines the tool schemas for omni operations.
These schemas are used to construct prompts for the omni agent.
"""

from typing import List, Dict, Any

def get_omni_tool_schemas() -> List[Dict[str, Any]]:
    """
    Get all omni tool schemas.

    Returns:
        List of tool schema dictionaries
    """
    return [
        get_omni_search_schema(),
        get_omni_read_schema(),
        get_omni_extract_clip_schema(),
        get_omni_run_python_schema(),
    ]

def get_omni_search_schema() -> Dict[str, Any]:
    """
    Schema for omni:search tool - search a keyword match in video captions
    and return the nearest matched clips to a target video_id.
    """
    return {
        "name": "omni:search",
        "description": (
            "Search for a keyword in video clip captions and return matched clip metadata "
            "nearest to a given target video_id. Matching is case-insensitive. "
            "The tool finds all matched clips, sorts them by distance to the target video_id, "
            "and returns up to max_search_results results in the response."
            "After calling omni:search, the model should examine the returned captions, select one caption "
            "whose video or audio content is likely to be relevant to the current trajectory exploration videos or captions, and then call "
            "omni:extract_clip to extract and inspect the corresponding video clip."
        ),
        "parameters": [
            {
                "name": "video_id",
                "type": "string",
                "description": (
                    "The target video clip id used as the anchor point for distance comparison."
                ),
                "required": True,
            },
            {
                "name": "key_word",
                "type": "string",
                "description": (
                    "A single keyword string to search for in video clip captions. "
                    "Matching is case-insensitive."
                ),
                "required": True,
            },
            {
                "name": "max_search_results",
                "type": "integer",
                "description": (
                    "Maximum number of matched results to display."
                ),
                "required": True,
            },
        ],
    }

# def get_omni_search_schema() -> Dict[str, Any]:
#     """
#     Schema for omni:search tool - search keyword matches in video captions.
#     """
#     return {
#         "name": "omni:search",
#         "description": (
#             "Search for one or more keywords in all video clip captions and return matched clip metadata. "
#             "Matching is case-insensitive. This tool is useful for locating relevant video segments based "
#             "on caption text. Note: video_info_path is automatically provided from kwargs, "
#             "so you do not need to specify it."
#         ),
#         "parameters": [
#             {
#                 "name": "key_words",
#                 "type": "array",
#                 "array_type": "string",
#                 "description": (
#                     "Array of keyword strings to search for in video clip captions. "
#                     "The tool will search each keyword independently and return matched clip information."
#                 ),
#                 "required": True,
#             },
#             {
#                 "name": "max_search_results",
#                 "type": "integer",
#                 "description": (
#                     "Maximum number of search results to keep for each keyword."
#                 ),
#                 "required": True,
#             },
#         ],
#     }

def get_omni_read_schema() -> Dict[str, Any]:
    """
    Schema for omni:read tool - read multimodal files.
    """
    return {
        "name": "omni:read",
        "description": (
            "Read one or more multimodal files and return their encoded contents. "
            "This tool loads local files, checks whether they exist, and returns "
            "their base64 content together with format and media type information."
        ),
        "parameters": [
            {
                "name": "mm_paths",
                "type": "array",
                "array_type": "string",
                "description": (
                    "List of local file paths to read. Each file will be checked "
                    "for existence before reading."
                ),
                "required": True,
            }
        ],
    }

def get_omni_extract_clip_schema() -> Dict[str, Any]:
    """
    Schema for omni:extract_clip tool - extract audio/video/av clips from multimodal files.
    """
    return {
        "name": "omni:extract_clip",
        "description": (
            "Extract one or more clips from a multimodal file such as a video or audio file. "
            "This tool supports extracting audio-video clips, audio-only clips, or video-only clips "
            "based on the provided time ranges. Note: output_dir is automatically provided in kwargs, "
            "so you do not need to specify it."
        ),
        "parameters": [
            {
                "name": "mm_path",
                "type": "string",
                "description": (
                    "Path to the input multimodal file to crop from."
                ),
                "required": True,
            },
            {
                "name": "clips",
                "type": "array",
                "array_type": "dict",
                "description": (
                    "List of clip ranges to extract. Each item must contain "
                    "'start_time' and 'end_time'. Example: "
                    "[{'start_time': 1.2, 'end_time': 5.8}]"
                ),
                # "items": {
                #     "type": "object",
                #     "properties": {
                #         "start_time": {
                #             "type": "float",
                #             "description": (
                #                 "Clip start time. Can be a float number."
                #             ),
                #         },
                #         "end_time": {
                #             "type": "float",
                #             "description": (
                #                 "Clip end time. Can be a float number."
                #             ),
                #         },
                #     },
                #     "required": ["start_time", "end_time"],
                # },
                "required": True,
            },
            {
                "name": "mode",
                "type": "string",
                "description": (
                    "Clip extraction mode. Supported values are 'av' for audio-video, "
                    "'audio' for audio only, and 'video' for video only. Default is 'av'."
                ),
                "required": False,
            },
        ],
    }

def get_omni_run_python_schema() -> Dict[str, Any]:
    """
    Schema for omni:run_python tool - execute a single Python code snippet.
    """
    return {
        "name": "omni:run_python",
        "description": (
            "Execute a single Python code snippet in a stateless way and capture its standard output. "
            "This tool only inspects the captured stdout to determine whether the code printed a file path. "
            "If the printed stdout is exactly one valid local file path, the tool will read that file and return it "
            "as multimodal output. Therefore, when the code generates an output file such as an image, figure, video frame, "
            "audio clip, or other processed artifact, the code must save the result to disk and print only that single file path. "
            "Do not print explanations, labels, extra text, multiple lines, or multiple paths together with the path, "
            "otherwise the tool cannot reliably detect the output file. "
            "If no output file is produced, the code may print normal textual results to stdout."
        ),
        "parameters": [
            {
                "name": "code",
                "type": "string",
                "description": (
                    "A single Python code string to execute. "
                    "If the code writes a result file to disk and wants the tool to return it as multimodal output, "
                    "the code must print exactly one file path to stdout and nothing else."
                ),
                "required": True,
            },
            {
                "name": "timeout",
                "type": "number",
                "description": (
                    "Execution timeout in seconds. Default is 5.0 seconds."
                ),
                "required": False,
            },
        ],
    }