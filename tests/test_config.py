from pipeline.config import load_config, merge_configs


def test_load_default_config():
    cfg = load_config()
    assert cfg["train"]["max_steps"] == 30000
    assert cfg["compress"]["sh_degree"] == 0


def test_load_default_config_has_required_schema_defaults():
    cfg = load_config()

    assert cfg["ingest"]["fps"] == 2
    assert cfg["ingest"]["blur_filter"] is True
    assert cfg["ingest"]["pose_path"] is None

    assert cfg["select"]["enabled"] is True
    assert cfg["select"]["min_blur_score"] is None
    assert cfg["select"]["max_temporal_gap"] is None
    assert cfg["select"]["keep_ratio"] == 1.0
    assert cfg["select"]["write_selected_images"] is False

    assert cfg["reconstruct"]["pairing"]["strategy"] == "sequential"
    assert cfg["reconstruct"]["undistort"] is True

    assert cfg["train"]["profile"] == "default"
    assert cfg["train"]["phase_overrides"] == {}

    assert cfg["report"]["write_summary"] is True



def test_merge_overrides():
    base = {"train": {"max_steps": 30000, "gpu": 0}}
    override = {"train": {"max_steps": 75000}}
    merged = merge_configs(base, override)
    assert merged["train"]["max_steps"] == 75000
    assert merged["train"]["gpu"] == 0



def test_merge_nested_pipeline_sections():
    base = {
        "select": {
            "enabled": True,
            "thresholds": {"blur": 0.2, "gap": 5},
        },
        "reconstruct": {
            "pairing": {"strategy": "sequential", "window": 10},
            "undistort": True,
        },
    }
    override = {
        "select": {"thresholds": {"gap": 8}},
        "reconstruct": {"pairing": {"strategy": "retrieval"}},
    }

    merged = merge_configs(base, override)

    assert merged["select"]["enabled"] is True
    assert merged["select"]["thresholds"]["blur"] == 0.2
    assert merged["select"]["thresholds"]["gap"] == 8
    assert merged["reconstruct"]["pairing"]["strategy"] == "retrieval"
    assert merged["reconstruct"]["pairing"]["window"] == 10
    assert merged["reconstruct"]["undistort"] is True



def test_load_custom_config(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text("train:\n  max_steps: 50000\n")
    cfg = load_config(custom)
    assert cfg["train"]["max_steps"] == 50000
    assert cfg["ingest"]["fps"] == 2



def test_load_custom_config_deep_merges_nested_sections(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "select:\n"
        "  keep_ratio: 0.5\n"
        "  write_selected_images: true\n"
        "reconstruct:\n"
        "  pairing:\n"
        "    strategy: retrieval\n"
    )

    cfg = load_config(custom)

    assert cfg["select"]["keep_ratio"] == 0.5
    assert cfg["select"]["write_selected_images"] is True
    assert cfg["select"]["enabled"] is True
    assert cfg["select"]["min_blur_score"] is None
    assert cfg["reconstruct"]["pairing"]["strategy"] == "retrieval"
    assert cfg["reconstruct"]["pairing"]["window"] == 10
    assert cfg["reconstruct"]["undistort"] is True



def test_load_custom_config_maps_legacy_sfm_to_reconstruct(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "sfm:\n"
        "  matcher: superglue\n"
        "  pairing:\n"
        "    strategy: retrieval\n"
    )

    cfg = load_config(custom)

    assert cfg["reconstruct"]["matcher"] == "superglue"
    assert cfg["reconstruct"]["pairing"]["strategy"] == "retrieval"
    assert cfg["sfm"]["matcher"] == "superglue"
    assert cfg["sfm"]["pairing"]["strategy"] == "retrieval"
