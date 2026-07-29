# Benchmark Evidence

Benchmark records are machine-readable so README tables can be traced back to
a concrete result source.

`results/reported_rtx4090.json` preserves the historical figures currently
shown in the project README. They are marked `historical_reported` because the
original run did not record a complete environment lock and source commit.
They are portfolio evidence, not a reproducible release benchmark.

A release-grade record must add:

- source commit SHA
- exact input-data identifier and redistribution status
- resolved configuration
- Python, CUDA, PyTorch, gsplat, COLMAP, and HLoc versions
- GPU model
- stage durations
- PSNR, SSIM, LPIPS, Gaussian count, and artifact sizes
- links or hashes for generated reports

Run:

```bash
python scripts/validate_benchmarks.py
```

The validator checks the evidence schema and verifies that reported
compression percentages agree with artifact sizes.
