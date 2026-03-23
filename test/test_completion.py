import os
import sys
import base64
from typing import Tuple
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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


from synthesis.core.utils import completion
path = "/share/project/guotianyu/AgentFlow/assets/intro.png"
response = completion(
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _encode_file_to_base64(path)[0]}}, 
        {"type": "text", "text": "What is the picture about?"}
    ]}],
    model="ep-20260317033626-qkz4b",
    completion_config={
        "api_url": "https://kspmas.ksyun.com/v1/models/ep-20260317031559-av3m2:generateContent",
        "api_key": "8407460c-9a3d-4a32-bb0d-43e91a74304f",
    }
)

print(response)