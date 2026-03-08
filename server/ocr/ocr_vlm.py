import openai
from pathlib import Path
from typing import List
from server.config import Config
from server.ingestion import ImgLike, to_urls

_PROMPT_CACHE: dict[str, tuple[int, str]] = {}
_CLIENT_CACHE: dict[tuple[str, str], openai.OpenAI] = {}

def _normalize_base_url(base_url: str) -> str:
    b = base_url.strip().rstrip("/")
    if b.endswith("/v1"):
        return b
    return b + "/v1"


def _read_prompt_md(path: Path) -> str:
    key = str(path)
    mtime = path.stat().st_mtime_ns
    cached = _PROMPT_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    prompt = path.read_text(encoding="utf-8").strip()
    _PROMPT_CACHE[key] = (mtime, prompt)
    return prompt


def vlm_ocr(images: List[ImgLike], cfg: Config) -> str:
    prompt_path = Path(cfg.prompt_md_path).expanduser().resolve()
    prompt = _read_prompt_md(prompt_path)

    urls = to_urls(images, max_side=cfg.max_image_side, jpeg_quality=cfg.jpeg_quality)

    content = [{"type": "text", "text": prompt}]
    for u in urls:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": u,
                    "detail": cfg.image_detail,
                },
            }
        )

    base_url = _normalize_base_url(cfg.vlm_base_url)
    api_key = cfg.vlm_api_key
    client = _CLIENT_CACHE.get((base_url, api_key))
    if client is None:
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
        _CLIENT_CACHE[(base_url, api_key)] = client
    response = client.chat.completions.create(
        model=cfg.vlm_model,
        messages=[{"role": "user", "content": content}],
        max_tokens=cfg.vlm_max_tokens,
    )

    return (response.choices[0].message.content or "").strip()
