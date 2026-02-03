import base64
import mimetypes
from pathlib import Path
import openai

class Config:
    api_key = "sk-123456"
    base_url = "http://localhost:8000/v1"
    model = "qwen3-vl-8b-instruct"
    image_dir = "/home/vision/Kestrel/data/20250422T150435"
    prompt_md_path = str(Path(__file__).with_name("prompt.md"))
    max_images = 14
    max_tokens = 8192

def _to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime or 'application/octet-stream'};base64,{b64}"

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
    if not text:
        raise SystemExit(f"提示词文件为空：{path}")
    return text

def main() -> int:
    cfg = Config()

    folder = Path(cfg.image_dir).expanduser().resolve()
    images = _images_in(folder)[: max(1, cfg.max_images)]

    prompt_path = Path(cfg.prompt_md_path).expanduser().resolve()
    prompt = _read_prompt_md(prompt_path)

    content = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _to_data_url(img), "detail": "auto"}})

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
