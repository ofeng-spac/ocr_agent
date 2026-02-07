__all__ = ["vlm_ocr"]

def __getattr__(name: str):
    if name == "vlm_ocr":
        from .vlm_ocr import vlm_ocr

        return vlm_ocr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
