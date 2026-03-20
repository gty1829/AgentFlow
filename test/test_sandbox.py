import os
import sys
import json
import asyncio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sandbox import Sandbox

SERVER_URL = "http://127.0.0.1:18890"


def show_result(result, base64_preview_len: int = 120):
    def truncate_mm_result(mm_result):
        preview = []
        for item in mm_result:
            if isinstance(item, dict):
                item_copy = dict(item)
                base64_data = item_copy.get("base64_data")
                if isinstance(base64_data, str) and len(base64_data) > base64_preview_len:
                    item_copy["base64_data"] = (
                        base64_data[:base64_preview_len]
                        + f"...<truncated, total_len={len(base64_data)}>"
                    )
                preview.append(item_copy)
            else:
                preview.append(item)
        return preview

    data = result.get("data", {})
    result_preview = dict(result)
    data_preview = dict(data)
    data_preview["mm_result"] = truncate_mm_result(data.get("mm_result", []))
    result_preview["data"] = data_preview

    print(json.dumps(result_preview, ensure_ascii=False, indent=2))


async def test_tool(sandbox: Sandbox, action: str, params: dict):
    result = await sandbox.execute(action, params)
    print(f"\n===== {action} =====")
    show_result(result)
    return result


async def test_run_python_text(sandbox: Sandbox):
    return await test_tool(sandbox, "omni:run_python", {
        "code": "print('hello from sandbox')",
        "timeout": 5.0,
    })


async def test_run_python_file(sandbox: Sandbox):
    code = """
with open('/share/project/guotianyu/AgentFlow/test/test_data/tool_test.txt', 'w', encoding='utf-8') as f:
    f.write('hello file')
print('/share/project/guotianyu/AgentFlow/test/test_data/tool_test.txt')
"""
    return await test_tool(sandbox, "omni:run_python", {
        "code": code,
        "timeout": 5.0,
    })


async def test_read(sandbox: Sandbox):
    return await test_tool(sandbox, "omni:read", {
        "mm_paths": [
            "/share/project/guotianyu/AgentFlow/test/test_data/2FE4zeNa5no/2FE4zeNa5no.mp4",
            "/share/project/guotianyu/AgentFlow/test/test_data/jLGXZApNW2o/jLGXZApNW2o.mp4",
        ]
    })


async def test_search_caption(sandbox: Sandbox):
    return await test_tool(sandbox, "omni:search", {
        "key_words": ["A man with a beard", "dog", "beard"],
        "max_search_results": 3,
        "video_info_path": "/share/project/guotianyu/AgentFlow/test/test_data/video_info_jsons/_LzSK2l6Fkc.json",
    })


async def test_extract_clip_av(sandbox: Sandbox):
    return await test_tool(sandbox, "omni:extract_clip", {
        "mm_path": "/share/project/guotianyu/AgentFlow/test/test_data/_LzSK2l6Fkc/_LzSK2l6Fkc.mp4",
        "clips": [{"start_time": 0, "end_time": 100.0}],
        "mode": "av",
        "output_dir": "/share/project/guotianyu/AgentFlow/test/test_data/_LzSK2l6Fkc",
    })

async def test_extract_clip_video(sandbox: Sandbox):
    return await test_tool(sandbox, "omni:extract_clip", {
        "mm_path": "/share/project/guotianyu/AgentFlow/test/test_data/_LzSK2l6Fkc/_LzSK2l6Fkc.mp4",
        "clips": [{"start_time": 0, "end_time": 10.0}],
        "mode": "video",
        "output_dir": "/share/project/guotianyu/AgentFlow/test/test_data/_LzSK2l6Fkc",
    })

async def test_extract_clip_audio(sandbox: Sandbox):
    return await test_tool(sandbox, "omni:extract_clip", {
        "mm_path": "/share/project/guotianyu/AgentFlow/test/test_data/_LzSK2l6Fkc/_LzSK2l6Fkc.mp4",
        "clips": [{"start_time": 0, "end_time": 10.0}],
        "mode": "audio",
        "output_dir": "/share/project/guotianyu/AgentFlow/test/test_data/_LzSK2l6Fkc",
    })


async def main():
    async with Sandbox(server_url=SERVER_URL) as sandbox:
        await test_extract_clip_av(sandbox)
        # await test_extract_clip_video(sandbox)
        # await test_extract_clip_audio(sandbox)
        # await test_search_caption(sandbox)
        # await test_read(sandbox)
        # await test_run_python_text(sandbox)
        # await test_run_python_file(sandbox)


if __name__ == "__main__":
    asyncio.run(main())
