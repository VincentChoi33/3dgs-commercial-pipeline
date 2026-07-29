# Quality-first training profiles

This repository owns the quality-first gaussian-splatting-lightning training profiles.

`pipeline/train.py` syncs these files into the external gaussian-splatting-lightning checkout under `$GS_LIGHTNING_PATH/configs/quality_first/` before launching training.

Source of truth in this repo:
- `configs/lightning/quality_first/quality.yaml`

Expected external target:
- `$GS_LIGHTNING_PATH/configs/quality_first/quality.yaml`

Keep edits here versioned with the pipeline repo rather than patching the external framework checkout by hand.
