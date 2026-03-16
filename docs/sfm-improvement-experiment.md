# SfM Improvement Experiment (2026-03-13)

## Objective

Improve 3DGS quality by enhancing the SfM (Structure-from-Motion) stage of the pipeline.
Three improvements were applied and evaluated against the baseline.

## Improvements

### 1. Keypoint Count & Resolution Increase

- **Baseline**: `superpoint_aachen` (max_keypoints=4096, resize_max=1024)
- **Improved**: `superpoint_max` (max_keypoints=4096, resize_max=**1600**)
- Higher resolution feature extraction captures finer details

### 2. Smart Pair Generation Strategy

- **Baseline**: Exhaustive pairs (all N*(N-1)/2 combinations)
- **Improved**: Sequential (overlap=10, quadratic) + NetVLAD Retrieval (top-15)
- Sequential pairs leverage temporal continuity of video frames
- Retrieval pairs enable loop closing for revisited areas
- Result: **21% of exhaustive pairs** with equivalent or better coverage

| Scene | Exhaustive Pairs | Seq+Retrieval Pairs | Ratio |
|-------|-----------------|-------------------|-------|
| 6027 (151 imgs) | 11,325 | 2,375 | 21.0% |
| 6028 (173 imgs) | 14,878 | 2,715 | 18.2% |

### 3. Proper Image Undistortion

- **Baseline**: Drop radial distortion parameter k when converting SIMPLE_RADIAL to PINHOLE
- **Improved**: `pycolmap.undistort_images()` — physically undistorts images and outputs clean PINHOLE model
- No information loss from distortion parameter removal

## SfM Results

| Scene | Metric | Baseline | Improved | Change |
|-------|--------|----------|----------|--------|
| 6027 | Registered images | 141/151 | 142/151 | +1 (+0.7%) |
| 6027 | 3D points | 23,388 | **36,346** | **+12,958 (+55.4%)** |
| 6028 | Registered images | 168/173 | 169/173 | +1 (+0.6%) |
| 6028 | 3D points | 27,696 | **46,454** | **+18,758 (+67.7%)** |

Image registration was already high (93-97%), so minimal gain there.
The major improvement is in **point cloud density** (+55~68%), which provides better initialization for 3DGS training.

### SfM Timing (per scene)

| Step | 6027 | 6028 |
|------|------|------|
| Feature extraction | 5s | 6s |
| NetVLAD descriptors | 25s | 28s |
| LightGlue matching | 88s | 105s |
| COLMAP reconstruction | 224s | 500s |
| Undistortion | 2s | 2s |
| **Total** | **330s** | **648s** |

## 3DGS Training Results

Training: gsplat_v1.yaml, 30k steps, lambda_dssim=0.3

| Scene | Metric | Baseline SfM | Improved SfM | Change |
|-------|--------|-------------|-------------|--------|
| **6027** | PSNR | 29.37 | **31.70** | **+2.33** |
| **6027** | SSIM | 0.942 | **0.958** | **+0.016** |
| **6027** | LPIPS | - | 0.163 | - |
| **6028** | PSNR | 23.80 | **24.90** | **+1.10** |
| **6028** | SSIM | 0.874 | **0.898** | **+0.024** |
| **6028** | LPIPS | - | 0.275 | - |

### Key Takeaway

PSNR improvement of **+2.33** (6027) and **+1.10** (6028) from SfM changes alone.
This confirms that SfM quality is a significant lever for 3DGS output quality, particularly point cloud density.

## Full Resolution Experiment (resize_max=3840)

Additional experiment: extract features at native 4K resolution (3840x2160) instead of downscaled 1600px.

### SfM Comparison (6027)

| resize_max | SfM Points | Change | SfM Time |
|---|---|---|---|
| 1024 (baseline) | 23,388 | - | ~60s |
| 1600 (improved) | 36,346 | +55% | 330s |
| **3840 (full res)** | **53,707** | **+130%** | 454s |

### 3DGS Quality Comparison (6027, 30k steps)

| resize_max | SfM Points | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| 1024 (baseline) | 23,388 | 29.37 | 0.942 | - |
| 1600 (improved) | 36,346 | **31.70** | **0.958** | 0.163 |
| 3840 (full res) | 53,707 | **31.80** | **0.958** | 0.160 |

### Conclusion

Point cloud density increases proportionally with resolution (+130%), but **PSNR gain is only +0.10** (31.70 → 31.80).
**resize_max=1600 is the optimal trade-off** — 4K provides diminishing returns because:
- SuperPoint was trained on lower-resolution images
- max_keypoints=4096 cap limits the benefit of additional resolution
- 3DGS training doesn't fully utilize the extra point density

## Output Files

### Gaussian Counts

| Scene | Full | After Prune | After DS50% |
|-------|------|-------------|-------------|
| 6027 (1600) | 993,109 | 880,410 | 440,205 |
| 6027 (3840) | 990,747 | 879,085 | 439,542 |
| 6028 (1600) | 1,170,086 | 937,695 | 468,847 |

### File Sizes

| Scene | full.ply | sh0_f16.ply | sh0_f16_ds50.ply |
|-------|----------|-------------|------------------|
| 6027 (1600) | 235 MB | 47 MB (20%) | 24 MB (10%) |
| 6027 (3840) | 234 MB | 47 MB (20%) | 24 MB (10%) |
| 6028 (1600) | 277 MB | 50 MB (18%) | 25 MB (9%) |

## Cumulative Results (All Experiments)

| Scene | Source | PSNR | SSIM | Notes |
|-------|--------|------|------|-------|
| MeetingRoom | 69 photos | 28.25 | - | Photo-based, sparse |
| Office | 416 photos | 26.07 | - | Photo-based |
| 6027 (video) | 151 frames (baseline SfM) | 29.37 | 0.942 | Video, blur-filtered |
| 6027 (video) | 142 frames (improved SfM, 1600) | **31.70** | **0.958** | Best trade-off |
| 6027 (video) | 142 frames (full res SfM, 3840) | **31.80** | **0.958** | Marginal gain |
| 6028 (video) | 173 frames (baseline SfM) | 23.80 | 0.874 | Video, blur-filtered |
| 6028 (video) | 169 frames (improved SfM) | **24.90** | **0.898** | Improved |
| 6028 orig | 217 frames (baseline SfM) | 22.89 | 0.856 | No blur filter |

## Environment

- **Server**: train2 (RTX 4090 x8)
- **SfM env**: `gsplat` (hloc 1.5, pycolmap 3.13.0, lightglue)
- **Training env**: `lightning` (gsplat 1.4.0, lightning 2.3.3)
- **Framework**: gaussian-splatting-lightning + GSplatV1Renderer

## Experiment Script

`scripts/experiment_sfm_improved.py` — standalone script for running improved SfM experiments.

## Pipeline Integration TODO

The three improvements should be integrated into `pipeline/sfm.py` as configurable options:

```yaml
sfm:
  feature: superpoint_max        # was: superpoint (superpoint_aachen)
  matcher: superpoint+lightglue  # hloc conf name
  max_keypoints: 4096
  resize_max: 1600
  pair_strategy: sequential+retrieval  # was: exhaustive
  sequence_overlap: 10
  retrieval_num: 15
  undistort: true                # was: convert_to_pinhole only
```
