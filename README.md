# Urban Growth Change Detection

This is a Python project for detecting urban change between two satellite images of the same area (time T1 vs time T2). The current version focuses on a simple, classical **computer vision** baseline that is easy to understand and iterate on.


This project is fundamentally an **image understanding** pipeline: it takes raw satellite images, performs **pixel-level reasoning**, cleans and segments the output, extracts **geometric shapes**, and produces visual + quantitative results. Even though the current baseline does not train a deep neural network, it uses core computer vision techniques that are commonly used as strong baselines and as building blocks for ML-based systems.

Computer vision pieces in the current implementation:
- **Change detection (image differencing)**: absolute difference between aligned images (T1 vs T2)
- **Binary segmentation**: thresholding to produce a change mask
- **Morphology**: opening/closing to denoise and fill holes
- **Connected components**: removing small blobs / noise regions
- **Contour extraction**: converting mask boundaries to polygons
- **Visualization**: alpha blending overlays and comparison figures for interpretation

ML fit (planned):
- The same pipeline can be extended with **learned segmentation** (e.g., U-Net style models) or **self-supervised feature matching** for robustness to clouds/lighting/seasonality. This is listed in the roadmap as a future improvement.

## What it does

Given two aligned images of the same location:
- Creates a change mask (pixels that changed between T1 and T2)
- Cleans the mask with basic morphology + small-component filtering
- Produces visual outputs (mask and overlays)
- Computes summary stats (changed pixels, percent change, polygon count)

Inputs supported:
- Single images: PNG, JPG, GeoTIFF
- Sentinel-2 style separate RGB GeoTIFF bands:
  - B04 = Red, B03 = Green, B02 = Blue

## How it works (high level)

The pipeline is:
1. Load images (optionally combine separate RGB band files)
2. Convert to grayscale (for a simple baseline)
3. Compute absolute difference per pixel
4. Threshold to get a binary change mask
5. Post-process the mask (closing/opening + remove small blobs)
6. Vectorize mask to polygons (pixel coordinates for now)
7. Save outputs and stats

Main modules:
- `ugcd/io.py`: image loading (including Sentinel-2 RGB band files)
- `ugcd/change.py`: absolute difference + thresholding
- `ugcd/postprocess.py`: morphology + blob filtering
- `ugcd/vectorize.py`: mask contours -> polygons
- `ugcd/stats.py`: summary statistics
- `ugcd/viz.py`: overlay and comparison figure

## Project status

Completed:
- Milestone 0: repo setup + tests
- Milestone 1: baseline change detection pipeline (works with your data)

Pending (planned):
- Milestone 2: export GeoJSON with real-world coordinates from GeoTIFF metadata
- Milestone 3: optional alignment (ECC / feature matching) when T1/T2 are slightly misaligned
- Milestone 4: Streamlit demo UI (upload images, tune parameters, download results)
- Future: ML-based change detection (segmentation / feature-based approaches) for improved robustness

See `docs/ROADMAP.md` for details.

## Data layout (Sentinel-2 example)

Place your files under `data/`:
- `t1_B02.tif`, `t1_B03.tif`, `t1_B04.tif`
- `t2_B02.tif`, `t2_B03.tif`, `t2_B04.tif`


## Run

Simplest way (uses files in `data/` and writes to `outputs/`):

```bash
python run_change_detection.py
```


## Outputs

The run generates:
- `outputs/change_mask.png`: binary mask (white = change)
- `outputs/overlay.png`: mask overlaid on T2
- `outputs/comparison.png`: T1 / T2 / mask / overlay in one image
- `outputs/stats.json`: basic summary statistics

Note: polygons currently live in memory as Shapely geometries and are in pixel coordinates. GeoJSON export is planned for Milestone 2.


## Limitations (current baseline)

- Assumes T1 and T2 are already aligned and same size
- Sensitive to clouds, shadows, seasonal differences, and illumination changes
- Uses a simple grayscale difference (good baseline, not production-grade)
- Vector polygons are pixel-space only (GeoJSON with CRS comes next)
