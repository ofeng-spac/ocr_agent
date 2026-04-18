from __future__ import annotations

from unittest.mock import MagicMock, call, patch

from app.services.recognizer import recognize_video, run_recognition


@patch("app.services.recognizer.time.perf_counter", side_effect=[10.0, 12.345])
@patch("app.services.recognizer.parse_recognition_result")
@patch("app.services.recognizer.load_server_functions")
@patch("app.services.recognizer.load_config")
def test_run_recognition_with_mocked_dependencies(
    mock_load_config,
    mock_load_server_functions,
    mock_parse_result,
    mock_perf_counter,
):
    mock_load_config.return_value = {
        "video": {"fps": 5, "max_frames": 14},
        "image": {"max_side": 1280, "quality": 85},
        "api": {
            "mock-model": {
                "api_key": "sk-test",
                "base_url": "http://localhost:8000/v1",
                "model": "mock-model-id",
                "max_tokens": 512,
            }
        },
    }

    sample_frames = MagicMock(return_value=["frame-1", "frame-2"])
    encode_frame = MagicMock(side_effect=["url-1", "url-2"])
    load_prompt = MagicMock(return_value="PROMPT")
    call_vlm = MagicMock(return_value="RAW_RESULT")
    mock_load_server_functions.return_value = (sample_frames, encode_frame, load_prompt, call_vlm)

    mock_parse_result.return_value = {
        "raw_name": "注射用头孢噻呋钠",
        "evidence_text": "兽用;批号:20240723",
        "uncertainty_text": "低",
        "uncertainty_level": "low",
        "raw_result": "RAW_RESULT",
    }

    result = run_recognition(
        video_path="/tmp/video001.mp4",
        model="mock-model",
        knowledge=False,
        guide=True,
    )

    sample_frames.assert_called_once_with("/tmp/video001.mp4", 5, 14)
    assert encode_frame.call_args_list == [
        call("frame-1", max_side=1280, quality=85),
        call("frame-2", max_side=1280, quality=85),
    ]
    load_prompt.assert_called_once()
    _, kwargs = load_prompt.call_args
    assert kwargs["guide"] is True
    assert kwargs["kb"] is False

    call_vlm.assert_called_once_with(
        ["url-1", "url-2"],
        "PROMPT",
        api_key="sk-test",
        base_url="http://localhost:8000/v1",
        model="mock-model-id",
        max_tokens=512,
    )
    mock_parse_result.assert_called_once_with("RAW_RESULT")

    assert result["video_name"] == "video001.mp4"
    assert result["model"] == "mock-model"
    assert result["elapsed"] == 2.345
    assert result["raw_name"] == "注射用头孢噻呋钠"
    assert result["evidence_text"] == "兽用;批号:20240723"
    assert result["uncertainty_level"] == "low"


@patch("app.services.recognizer.apply_verification")
@patch("app.services.recognizer.run_recognition")
def test_recognize_video_composes_recognition_and_verification(
    mock_run_recognition,
    mock_apply_verification,
):
    recognition_result = {
        "video_name": "video001.mp4",
        "model": "mock-model",
        "elapsed": 1.234,
        "result": "RAW_RESULT",
        "raw_name": "注射用头孢噻呋钠",
        "evidence_text": "兽用;批号:20240723",
        "uncertainty_text": "低",
        "uncertainty_level": "low",
    }
    final_result = {
        **recognition_result,
        "canonical_name": "注射用头孢噻呋钠",
        "verify_status": "verified_exact",
        "verify_match_type": "canonical",
        "verify_reason": "已匹配到标准药名 注射用头孢噻呋钠。",
    }
    mock_run_recognition.return_value = recognition_result
    mock_apply_verification.return_value = final_result

    result = recognize_video(
        video_path="/tmp/video001.mp4",
        model="mock-model",
        knowledge=True,
        guide=False,
        expected_drug_name="注射用头孢噻呋钠",
    )

    mock_run_recognition.assert_called_once_with(
        video_path="/tmp/video001.mp4",
        model="mock-model",
        knowledge=True,
        guide=False,
    )
    mock_apply_verification.assert_called_once_with(
        recognition_result,
        "注射用头孢噻呋钠",
    )
    assert result == final_result
