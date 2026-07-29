import pytest
import numpy as np
from pathlib import Path


def test_compute_blur_score():
    from pipeline.preprocess import compute_blur_score
    sharp = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    blurry = np.ones((100, 100, 3), dtype=np.uint8) * 128
    assert compute_blur_score(sharp) > compute_blur_score(blurry)


def test_is_video_file(tmp_path):
    from pipeline.preprocess import is_video
    assert is_video(Path("test.mp4")) is False  # file doesn't exist
    assert not is_video(Path("test.jpg"))
    assert not is_video(Path("images/"))
    # positive case: a real file with a video extension
    real_mp4 = tmp_path / "clip.mp4"
    real_mp4.touch()
    assert is_video(real_mp4) is True
    # real file with non-video extension → False
    real_jpg = tmp_path / "photo.jpg"
    real_jpg.touch()
    assert is_video(real_jpg) is False


def test_filter_blurry_frames(tmp_path):
    from pipeline.preprocess import filter_blurry_frames
    import cv2

    for i in range(8):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{i:04d}.jpg"), img)
    for i in range(8, 10):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(tmp_path / f"frame_{i:04d}.jpg"), img)

    kept = filter_blurry_frames(tmp_path, percentile=20)
    assert len(kept) == 8


def test_extract_frames_delegates_to_ingest(monkeypatch, tmp_path):
    from pipeline import preprocess

    called = {}

    def fake_extract_frames(video_path, output_dir, fps=2):
        called["video_path"] = video_path
        called["output_dir"] = output_dir
        called["fps"] = fps
        return output_dir

    monkeypatch.setattr(preprocess, "ingest_extract_frames", fake_extract_frames)

    video_path = tmp_path / "clip.mp4"
    output_dir = tmp_path / "frames"
    result = preprocess.extract_frames(video_path, output_dir, fps=5)

    assert result == output_dir
    assert called == {
        "video_path": video_path,
        "output_dir": output_dir,
        "fps": 5,
    }


def test_run_preprocess_returns_selected_images_contract(tmp_path):
    from pipeline.preprocess import run_preprocess
    import cv2

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    for i in range(4):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(photos_dir / f"frame_{i:04d}.jpg"), img)

    scene_dir = tmp_path / "scene"
    selected_dir = run_preprocess(photos_dir, scene_dir, {"pose_path": None, "keep_ratio": 0.5})

    assert selected_dir == scene_dir / "selected_images"
    assert selected_dir.exists()
    assert (scene_dir / "images").exists()
    assert (scene_dir / "metadata" / "frames.json").exists()
    assert (scene_dir / "metadata" / "selection_metrics.json").exists()


def test_filter_blurry_frames_raises_for_unreadable_image(tmp_path):
    from pipeline.preprocess import filter_blurry_frames

    unreadable = tmp_path / "frame_0000.jpg"
    unreadable.write_bytes(b"not-an-image")

    with pytest.raises(ValueError, match="Unable to read image: .*frame_0000.jpg"):
        filter_blurry_frames(tmp_path)
