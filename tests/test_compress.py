import numpy as np
import pytest
from pathlib import Path
from plyfile import PlyData, PlyElement


def _make_test_ply(path: Path, n: int = 1000):
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    dtype += [("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
    for i in range(45):
        dtype.append((f"f_rest_{i}", "f4"))
    dtype.append(("opacity", "f4"))
    dtype += [("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")]
    dtype += [("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]

    rng = np.random.default_rng(0)
    arr = np.zeros(n, dtype=dtype)
    arr["x"] = rng.standard_normal(n)
    arr["y"] = rng.standard_normal(n)
    arr["z"] = rng.standard_normal(n)
    arr["f_dc_0"] = rng.standard_normal(n)
    arr["f_dc_1"] = rng.standard_normal(n)
    arr["f_dc_2"] = rng.standard_normal(n)
    for i in range(45):
        arr[f"f_rest_{i}"] = rng.standard_normal(n)
    arr["opacity"] = 0.0
    arr["scale_0"] = 0.0
    arr["scale_1"] = 0.0
    arr["scale_2"] = 0.0
    arr["rot_0"] = 1.0
    arr["rot_1"] = 0.0
    arr["rot_2"] = 0.0
    arr["rot_3"] = 0.0

    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    return path


def test_compress_default(tmp_path):
    from pipeline.compress import run_compress
    from pipeline.metadata import read_optional_json

    ply = _make_test_ply(tmp_path / "full.ply", n=1000)
    cfg = {"sh_degree": 0, "float16": True, "prune_threshold": 0.005, "downsample": 0.5}
    run_compress(ply, tmp_path, cfg)

    compressed = tmp_path / "compressed.ply"
    downsampled = tmp_path / "compressed_ds50.ply"

    assert compressed.exists()
    assert downsampled.exists()
    assert compressed.stat().st_size < ply.stat().st_size
    assert downsampled.stat().st_size < compressed.stat().st_size

    report = read_optional_json(tmp_path / "metadata" / "compression_report.json")
    assert report["input_path"] == str(ply)
    assert report["compression_settings"] == cfg
    assert [entry["path"] for entry in report["output_files"]] == [str(compressed), str(downsampled)]
    assert [entry["gaussian_count"] for entry in report["output_files"]] == [1000, 500]
    assert [entry["output_size_bytes"] for entry in report["output_files"]] == [compressed.stat().st_size, downsampled.stat().st_size]


def test_reduce_sh_preserves_coefficient_major_layout():
    from pipeline.compress import reduce_sh

    current_degree = 3
    target_degree = 1
    current_coeffs = (current_degree + 1) ** 2 - 1
    f_rest = np.arange(current_coeffs * 3, dtype=np.float32).reshape(1, current_coeffs * 3)

    reduced, reduced_degree = reduce_sh(f_rest, current_degree, target_degree)

    expected = np.arange(9, dtype=np.float32).reshape(1, 9)
    assert reduced_degree == target_degree
    assert np.array_equal(reduced, expected)


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1])
def test_compress_rejects_invalid_downsample_ratios(tmp_path, ratio):
    from pipeline.compress import compress_ply

    ply = _make_test_ply(tmp_path / "full.ply", n=10)

    with pytest.raises(ValueError, match=r"downsample_ratio must be in \(0, 1\], got"):
        compress_ply(ply, tmp_path / "invalid.ply", downsample_ratio=ratio)
