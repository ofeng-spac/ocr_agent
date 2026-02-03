import base64
import io
import mimetypes
from pathlib import Path
import openai
from PIL import Image, ImageOps

class Config:
    api_key = "sk-123456"
    base_url = "http://localhost:8000/v1"
    model = "qwen3-vl-8b-instruct"
    image_dir = "/home/vision/Kestrel/data/20250422T150435"
    prompt_md_path = str(Path(__file__).with_name("prompt.md"))
    max_images = 14
    image_detail = "auto"
    max_image_side = 1280
    jpeg_quality = 85
    max_tokens = 8192

def _to_data_url(path: Path, max_side: int, jpeg_quality: int) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime == "image/gif":
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")

        w, h = im.size
        if max_side and max(w, h) > max_side:
            scale = max_side / max(w, h)
            im = im.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def _images_in(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)

def _normalize_base_url(base_url: str) -> str:
    b = base_url.strip().rstrip("/")
    if b.endswith("/v1"):
        return b
    return b + "/v1"

def _read_prompt_md(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    return text

def main() -> int:
    cfg = Config()

    folder = Path(cfg.image_dir).expanduser().resolve()
    images = _images_in(folder)[: max(1, cfg.max_images)]

    prompt_path = Path(cfg.prompt_md_path).expanduser().resolve()
    prompt = _read_prompt_md(prompt_path)

    content = [{"type": "text", "text": prompt}]
    for img in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _to_data_url(img, cfg.max_image_side, cfg.jpeg_quality),
                    "detail": cfg.image_detail,
                },
            }
        )

    client = openai.OpenAI(base_url=_normalize_base_url(cfg.base_url), api_key=cfg.api_key)
    response = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": content}],
        max_tokens=cfg.max_tokens,
    )
    print(response.choices[0].message.content)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
