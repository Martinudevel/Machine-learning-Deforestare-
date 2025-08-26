#!/usr/bin/env python3
"""
Fixed Sentinel-1 deforestation change detection script
-----------------------------------------------------
This is an improved/fixed version of the earlier script. Fixes and improvements:
- Correctly uses args.out_prefix (no hyphen bug)
- Safer raster writing (handles masked arrays, chooses sane nodata values)
- Better error messages when input grids don't match
- Accepts .tif/.tiff/.TIF/.TIFF inputs (rasterio handles extensions transparently)
- Optional --pause flag to keep console open after running (useful when launched by a bash script)
- Keeps same output format (GeoTIFF) but you can change extension in --out-prefix

Usage examples (same as before):
  python deforestation_s1_change_fixed.py before.tiff after.tiff --out-prefix results/s1_vh --method threshold --decrease-db -1.5
  python deforestation_s1_change_fixed.py before.tif after.tif --method otsu --pause

Dependencies:
  pip install numpy rasterio matplotlib scipy scikit-image

"""

import argparse
import os
import sys
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
import warnings

try:
    from skimage.filters import threshold_otsu
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    from scipy.ndimage import median_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def infer_units(arr):
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 'db'
    a = arr[finite]
    if (a.min() >= -60) and (a.max() <= 20):
        return 'db'
    if a.max() <= 5.0:
        return 'linear'
    return 'db'


def to_change_db(before, after, units):
    if units == 'auto':
        units = infer_units(np.concatenate([before.ravel(), after.ravel()]))
    if units == 'db':
        change_db = after - before
    elif units == 'linear':
        eps = 1e-10
        change_db = 10.0 * np.log10((after + eps) / (before + eps))
    else:
        raise ValueError("units must be one of {'auto','db','linear'}")
    return change_db, units


def load_raster(path):
    try:
        ds = rasterio.open(path)
    except Exception as e:
        raise RuntimeError(f"Could not open '{path}': {e}")
    arr = ds.read(1, masked=True)
    nodata = ds.nodata
    transform = ds.transform
    crs = ds.crs
    profile = ds.profile
    ds.close()
    return arr, nodata, transform, crs, profile


def write_raster(path, arr, profile, out_dtype=None, out_nodata=None):
    prof = profile.copy()
    if out_dtype is None:
        out_dtype = prof.get('dtype', 'float32')
    out_dtype = str(out_dtype)

    # Only use predictor=3 for float rasters
    if 'float' in out_dtype:
        prof.update(dtype=out_dtype, count=1, compress='deflate', predictor=3, tiled=True, blockxsize=256, blockysize=256)
    else:
        prof.update(dtype=out_dtype, count=1, compress='deflate', tiled=True, blockxsize=256, blockysize=256)

    if out_nodata is None:
        if 'float' in out_dtype:
            out_nodata = prof.get('nodata', -9999.0)
        else:
            out_nodata = prof.get('nodata', 0)
    prof['nodata'] = out_nodata

    if isinstance(arr, np.ma.MaskedArray):
        arr_write = arr.filled(out_nodata).astype(out_dtype)
    else:
        arr_write = np.array(arr)
        if np.issubdtype(arr_write.dtype, np.floating):
            arr_write = np.where(np.isfinite(arr_write), arr_write, out_nodata).astype(out_dtype)
        else:
            arr_write = arr_write.astype(out_dtype)

    with rasterio.open(path, 'w', **prof) as dst:
        dst.write(arr_write, 1)



def maybe_median(arr, k):
    if k is None or k <= 1:
        return arr
    if not _HAS_SCIPY:
        print("[warn] scipy not available; --smooth ignored", file=sys.stderr)
        return arr
    data = arr.filled(np.nan)
    mask = np.isfinite(data)
    filtered = data.copy()
    # apply median_filter only on the whole array: median_filter will treat NaN as values, so we mask
    try:
        filtered_full = median_filter(np.nan_to_num(data, nan=0.0), size=k)
        # preserve NaN where original was NaN
        filtered_full[~mask] = np.nan
        return np.ma.array(filtered_full, mask=~np.isfinite(filtered_full))
    except Exception as e:
        print(f"[warn] median filter failed: {e}", file=sys.stderr)
        return arr


def compute_classes(change_db, decrease_db=-1.5, method='threshold', urban_mask=None):
    classes = np.zeros(change_db.shape, dtype=np.uint8)
    valid = np.isfinite(change_db)
    ch = np.zeros_like(change_db)
    ch[valid] = change_db[valid]

    if method == 'otsu':
        if not _HAS_SKIMAGE:
            print("[warn] scikit-image not available; falling back to fixed threshold", file=sys.stderr)
            method = 'threshold'
        else:
            neg = ch[(valid) & (ch < 0)]
            if neg.size >= 50:
                try:
                    t = threshold_otsu(-neg)
                    decrease_db = -t
                except Exception:
                    print("[warn] Otsu failed; using default threshold", file=sys.stderr)
            else:
                print("[warn] not enough negative samples for Otsu; using default threshold", file=sys.stderr)

    dec = (ch <= decrease_db) & valid
    inc = (ch >= abs(decrease_db)) & valid
    stab = (~dec & ~inc) & valid

    classes[dec] = 1
    classes[stab] = 2
    classes[inc] = 3

    if urban_mask is not None:
        try:
            urb = (urban_mask != 0)
            classes[urb & (classes == 1)] = 2
        except Exception as e:
            print(f"[warn] could not apply urban mask: {e}", file=sys.stderr)

    classes[~valid] = 0
    return classes, decrease_db


def pixel_area_m2(transform: Affine, crs):
    if crs is None or not getattr(crs, 'is_projected', False):
        return None
    return abs(transform.a * transform.e)


def summarize_area(classes, pix_area_m2):
    if pix_area_m2 is None:
        return None
    vals, counts = np.unique(classes, return_counts=True)
    stats = {int(v): int(c) for v, c in zip(vals, counts)}
    out = {}
    for k, c in stats.items():
        out[k] = {
            'pixels': c,
            'area_m2': c * pix_area_m2,
            'area_ha': (c * pix_area_m2) / 10000.0
        }
    return out


def read_optional_mask(mask_path, ref_profile):
    if mask_path is None:
        return None
    with rasterio.open(mask_path) as ds:
        # If geometry/crs/shape differ, resample nearest to ref grid
        if (ds.crs != ref_profile['crs']) or (ds.transform != ref_profile['transform']) or (ds.width != ref_profile['width']) or (ds.height != ref_profile['height']):
            data = ds.read(1, out_shape=(ref_profile['height'], ref_profile['width']), resampling=Resampling.nearest)
        else:
            data = ds.read(1)
    data = (data != 0).astype(np.uint8)
    return data


def quicklook_png(path, before_db, after_db, change_db, classes, decrease_db):
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(before_db, vmin=np.nanpercentile(before_db, 2), vmax=np.nanpercentile(before_db, 98))
    ax1.set_title('Before (dB)')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(after_db, vmin=np.nanpercentile(after_db, 2), vmax=np.nanpercentile(after_db, 98))
    ax2.set_title('After (dB)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(2, 2, 3)
    im3 = ax3.imshow(change_db, vmin=-5, vmax=5)
    ax3.set_title('Change (after - before) dB')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = plt.subplot(2, 2, 4)
    im4 = ax4.imshow(classes, vmin=0, vmax=3)
    ax4.set_title(f"Classes (1=decrease≤{decrease_db:.2f} dB)")
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Sentinel-1 deforestation change detection (fixed version)')
    parser.add_argument('before', help='Path to BEFORE GeoTIFF (single band VV or VH)')
    parser.add_argument('after', help='Path to AFTER GeoTIFF (same grid as BEFORE)')
    parser.add_argument('--out-prefix', default='s1_change', help='Output path prefix (default: s1_change)')
    parser.add_argument('--units', choices=['auto','db','linear'], default='auto', help='Units of input rasters')
    parser.add_argument('--smooth', type=int, default=0, help='Median filter window size (odd int, e.g., 3 or 5)')
    parser.add_argument('--method', choices=['threshold','otsu'], default='threshold', help='Decrease decision method')
    parser.add_argument('--decrease-db', type=float, default=-1.5, help="Threshold in dB for 'decrease' (<= value)")
    parser.add_argument('--urban-mask', default=None, help='Optional 0/1 raster to mask out built-up areas')
    parser.add_argument('--pause', action='store_true', help='Pause and wait for Enter before exiting (useful when launching from a terminal that closes)')
    args = parser.parse_args()

    before, nodata_b, transform, crs, profile = load_raster(args.before)
    after, nodata_a, transform2, crs2, profile2 = load_raster(args.after)

    # sanity checks
    if (transform != transform2) or (profile['width'] != profile2['width']) or (profile['height'] != profile2['height']):
        print("[error] BEFORE and AFTER rasters must be perfectly aligned (same grid, transform, size).", file=sys.stderr)
        print(f" BEFORE: {args.before}: size={profile['width']}x{profile['height']} transform={transform}")
        print(f" AFTER : {args.after }: size={profile2['width']}x{profile2['height']} transform={transform2}")
        sys.exit(2)

    before = np.ma.masked_invalid(before)
    after = np.ma.masked_invalid(after)

    if nodata_b is not None:
        before = np.ma.masked_where(before == nodata_b, before)
    if nodata_a is not None:
        after = np.ma.masked_where(after == nodata_a, after)

    if args.smooth and args.smooth > 1 and (args.smooth % 2 == 1):
        before = maybe_median(before, args.smooth)
        after = maybe_median(after, args.smooth)
    elif args.smooth and args.smooth % 2 == 0:
        print('[warn] --smooth should be an odd number; ignoring', file=sys.stderr)

    change_db, used_units = to_change_db(before.filled(np.nan), after.filled(np.nan), args.units)

    urb_mask = read_optional_mask(args.urban_mask, profile) if args.urban_mask else None
    classes, used_dec_thr = compute_classes(change_db, decrease_db=args.decrease_db, method=args.method, urban_mask=urb_mask)

    out_change = f"{args.out_prefix}_change_db.tif"
    out_classes = f"{args.out_prefix}_classes.tif"
    out_png = f"{args.out_prefix}_quicklook.png"

    # Save rasters (choose sane nodata)
    try:
        write_raster(out_change, change_db, profile, out_dtype='float32', out_nodata=-9999.0)
        write_raster(out_classes, classes, profile, out_dtype='uint8', out_nodata=0)
    except Exception as e:
        print(f"[fatal] could not write outputs: {e}", file=sys.stderr)
        sys.exit(1)

    if used_units == 'linear':
        eps = 1e-10
        before_db_vis = 10.0 * np.log10(before.filled(np.nan) + eps)
        after_db_vis  = 10.0 * np.log10(after.filled(np.nan) + eps)
    else:
        before_db_vis = before.filled(np.nan)
        after_db_vis = after.filled(np.nan)

    quicklook_png(out_png, before_db_vis, after_db_vis, change_db, classes, used_dec_thr)

    pix_area = pixel_area_m2(transform, crs)
    stats = summarize_area(classes, pix_area)

    print('=== Change Detection Summary ===')
    print(f'Units used: {used_units}')
    print(f'Decrease threshold (dB): {used_dec_thr:.3f}')
    if pix_area is None:
        print('Pixel area: unknown (CRS not projected in meters). Area stats unavailable.')
    else:
        print(f'Pixel area: {pix_area:.3f} m^2')
    if stats is not None:
        legend = {0:'nodata',1:'decrease (deforestation cand.)',2:'stable',3:'increase'}
        for cls in [1,2,3]:
            if cls in stats:
                s = stats[cls]
                print(f"Class {cls} ({legend[cls]}): {s['pixels']} px | {s['area_ha']:.3f} ha")

    print('\nOutputs:')
    print(f'  {out_change}')
    print(f'  {out_classes}')
    print(f'  {out_png}')

    if args.pause:
        try:
            input('\nProcess complete. Press Enter to exit...')
        except Exception:
            pass


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[fatal] {e}", file=sys.stderr)
        sys.exit(1)
