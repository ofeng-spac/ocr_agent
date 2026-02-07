from pathlib import Path

class Config:
    video_path = str(Path(__file__).resolve().parent / "video" / "video5.mp4")
    target_fps = 5

    ocr = "vlm"    # "vlm" "paddle" "tesseract"

    enable_crop_background = True
    crop_border = 10
    crop_k = 3.2
    crop_area_ratio = 0.02
    crop_pad_ratio = 0.06

    max_images = 14
    image_detail = "auto"
    max_image_side = 1280
    jpeg_quality = 85

    prompt_md_path = str(Path(__file__).resolve().parent / "server" / "prompt.md")

    vlm_api_key = "sk-67656f65a21a45bea7b6bfa0e206bcd4"
    vlm_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    vlm_model = "qwen3-vl-flash"
    vlm_max_tokens = 8192

    tesseract_lang = "chi_sim+eng"
    tesseract_psm = None
    tesseract_oem = None
    tesseract_cmd = None
    tesseract_extra_config = None
    tesseract_cmd = r"C:\Users\hanze\scoop\apps\tesseract\current\tesseract.exe"