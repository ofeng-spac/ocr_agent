import openai
from pathlib import Path
from typing import List
from config import Config
from server.ingestion import ImgLike, to_urls


def _normalize_base_url(base_url: str) -> str:
    b = base_url.strip().rstrip("/")
    if b.endswith("/v1"):
        return b
    return b + "/v1"


def _read_prompt_md(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def vlm_ocr(images: List[ImgLike], cfg: Config) -> str:
    prompt_path = Path(cfg.prompt_md_path).expanduser().resolve()
    prompt = _read_prompt_md(prompt_path)

    images = images[: cfg.max_images]
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

    client = openai.OpenAI(
        base_url=_normalize_base_url(cfg.vlm_base_url),
        api_key=cfg.vlm_api_key,
    )
    response = client.chat.completions.create(
        model=cfg.vlm_model,
        messages=[{"role": "user", "content": content}],
        max_tokens=cfg.vlm_max_tokens,
    )

    return (response.choices[0].message.content or "").strip()
