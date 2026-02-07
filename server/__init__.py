from config import Config

__all__ = ["Config", "ImgLike", "crop_background", "to_urls", "video_downsample"]

def __getattr__(name: str):
    if name in {"ImgLike", "crop_background", "to_urls", "video_downsample"}:
        from . import ingestion as _ingestion

        return getattr(_ingestion, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
