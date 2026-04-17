from __future__ import annotations

from pydantic import BaseModel


class RecognizeRequest(BaseModel):
    video_name: str
    model: str = "qwen3-vl-8b-instruct-awq-4bit"
    knowledge: bool = True
    guide: bool = True
    expected_drug_name: str | None = None


class VerifyRequest(BaseModel):
    video_name: str
    expected_drug_name: str
    model: str = "qwen3-vl-8b-instruct-awq-4bit"
    knowledge: bool = True
    guide: bool = True


class AskRequest(BaseModel):
    canonical_name: str
    question: str


class VideoItem(BaseModel):
    name: str
    url: str
