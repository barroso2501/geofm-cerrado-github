"""
spatial_split.py
================
Generates a spatially-stratified train/val/test split for GeoFM v2 Cerrado.

Strategy:
  1. Assign each pixel (row, col) to its nearest hexagon centroid
     using Albers Equal Area projection (same as hex_Cerrado_class_Change.shp)
  2. Split HEXAGONS (not pixels) into train/val/test (70/15/15)
     stratified by Class8590 pattern code
  3. All pixels within a hexagon go to the same split
     → guaranteed zero spatial overlap between splits

Inputs (all in same directory, or update paths below):
  - treino_balanceado_FINAL.csv
  - hex_Cerrado_class_Change.shp (.dbf, .shx, .prj)

Outputs:
  - spatial_split_train.csv
  - spatial_split_val.csv
  - spatial_split_test.csv
  - treino_balanceado_SPATIAL_SPLIT.csv  (master file with 'split' column)
  - spatial_split_metadata.json

Usage:
  python spatial_split.py

No external dependencies beyond numpy and pandas.
"""

import numpy as np
import pandas as pd
import struct
import json
import os

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
CSV_PATH  = r"D:\Projetos\Cerrado\GeoFM_sampling\balanced_dataset\treino_balanceado_FINAL.csv"
SHP_PATH  = r"D:\Projetos\Cerrado\hex_Cerrado_class_Change.shp"
DBF_PATH  = r"D:\Projetos\Cerrado\hex_Cerrado_class_Change.dbf"
OUT_DIR   = r"D:\Projetos\Cerrado\GeoFM_sampling\spatial_split"

SEED       = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# Raster parameters (brazil_coverage_*_Cerrado.tif)
LEFT   = -60.47269846482955
RIGHT  = -41.277407642235204
TOP    =  -2.3319366460458646
BOTTOM = -24.681931083411143
NROWS  = 82933
NCOLS  = 71227

# ── PROJECTION (Albers Equal Area, SAD69) ─────────────────────────────────────
a    = 6378160.0; f = 1/298.25; b = a*(1-f); e2 = 1-(b/a)**2; e = np.sqrt(e2)
lon0 = np.radians(-60.0); lat0 = np.radians(-32.0)
phi1 = np.radians(-5.0);  phi2 = np.radians(-42.0)

def _alpha(phi):
    sp = np.sin(phi)
    return (1-e2)*(sp/(1-e2*sp**2) - np.log((1-e*sp)/(1+e*sp))/(2*e))

def _Nf(phi): return np.cos(phi) / np.sqrt(1 - e2*np.sin(phi)**2)

_n  = (_Nf(phi1)**2 - _Nf(phi2)**2) / (2*(_alpha(phi2) - _alpha(phi1)))
_C  = _Nf(phi1)**2 + _n*_alpha(phi1)
_r0 = a * np.sqrt(_C - _n*_alpha(lat0)) / _n

def rowcol_to_albers(row, col):
    """Convert raster (row, col) indices to Albers XY coordinates."""
    lon = LEFT + (np.asarray(col, float) + 0.5) * (RIGHT - LEFT) / NCOLS
    lat = TOP  + (np.asarray(row, float) + 0.5) * (BOTTOM - TOP) / NROWS
    phi = np.radians(lat); lam = np.radians(lon)
    r   = a * np.sqrt(np.maximum(_C - _n*_alpha(phi), 0)) / _n
    th  = _n * (lam - lon0)
    return r*np.sin(th), _r0 - r*np.cos(th)

# ── I/O HELPERS ───────────────────────────────────────────────────────────────
def read_shp_centroids(path):
    """Read hexagon bounding-box centroids from SHP file."""
    ids, cxs, cys = [], [], []
    with open(path, 'rb') as f:
        f.read(100)
        while True:
            hdr = f.read(8)
            if len(hdr) < 8: break
            rec_num  = struct.unpack('>I', hdr[:4])[0]
            clen     = struct.unpack('>I', hdr[4:])[0]
            data     = f.read(clen * 2)
            if len(data) < 4: break
            stype = struct.unpack('<I', data[:4])[0]
            if stype == 0: continue
            bbox = struct.unpack('<4d', data[4:36])
            ids.append(rec_num)
            cxs.append((bbox[0] + bbox[2]) / 2)
            cys.append((bbox[1] + bbox[3]) / 2)
    return np.array(ids), np.array(cxs), np.array(cys)

def read_dbf(path):
    """Read DBF attribute table."""
    with open(path, 'rb') as f:
        f.read(4)
        n_recs = struct.unpack('<I', f.read(4))[0]
        hsize  = struct.unpack('<H', f.read(2))[0]
        f.read(22)
        fields = []
        while True:
            fd = f.read(32)
            if fd[0] == 0x0D: break
            name   = fd[:11].replace(b'\x00', b'').decode('latin-1').strip()
            ftype  = chr(fd[11])
            length = fd[16]
            fields.append((name, ftype, length))
        f.seek(hsize)
        records = []
        for _ in range(n_recs):
            d = f.read(1)
            if d == b'*':
                for _, _, l in fields: f.read(l)
                continue
            row = {}
            for name, ftype, length in fields:
                raw = f.read(length).decode('latin-1').strip()
                if ftype == 'N' and raw and '.' not in raw:
                    row[name] = int(raw)
                else:
                    row[name] = raw if raw else None
            records.append(row)
    return pd.DataFrame(records)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    SEP = "=" * 60

    # 1. Load hexagons
    print(f"\n{SEP}\n1. Loading hexagons\n{SEP}")
    ids, cxs, cys = read_shp_centroids(SHP_PATH)
    dbf = read_dbf(DBF_PATH)
    dbf['hex_id'] = dbf['OBJECTID_1'].astype(int)
    hex_df = pd.DataFrame({'hex_id': ids, 'cx': cxs, 'cy': cys})
    hex_df = hex_df.merge(
        dbf[['hex_id','Class8590','Class9095','Class9500',
             'Class0005','Class0510','Class1015','Class1520']],
        on='hex_id', how='left')
    print(f"  Hexagons: {len(hex_df):,}")

    # 2. Load pixels
    print(f"\n{SEP}\n2. Loading pixels\n{SEP}")
    df = pd.read_csv(CSV_PATH)
    df_valid = df[df['label'].notna()].reset_index(drop=True)
    df_valid['label'] = df_valid['label'].astype(int)
    print(f"  Valid pixels: {len(df_valid):,}")
    print(f"  label=1: {(df_valid['label']==1).sum():,} | label=0: {(df_valid['label']==0).sum():,}")

    # 3. Assign each pixel to nearest hexagon
    print(f"\n{SEP}\n3. Assigning pixels to hexagons (nearest centroid)\n{SEP}")
    px, py = rowcol_to_albers(df_valid['row'].values, df_valid['col'].values)
    HX = hex_df['cx'].values; HY = hex_df['cy'].values
    BATCH = 500; assigned = []
    for i in range(0, len(px), BATCH):
        dx = px[i:i+BATCH, None] - HX[None, :]
        dy = py[i:i+BATCH, None] - HY[None, :]
        assigned.extend(np.argmin(dx**2 + dy**2, axis=1).tolist())

    df_valid['hex_id']    = hex_df['hex_id'].values[assigned]
    df_valid['Class8590'] = hex_df['Class8590'].values[assigned]
    df_valid['dist_km']   = np.sqrt((px - HX[assigned])**2 + (py - HY[assigned])**2) / 1000

    n_hex_used = df_valid['hex_id'].nunique()
    print(f"  Hexagons used: {n_hex_used:,}")
    print(f"  Distance to nearest centroid (km):")
    print(f"    median={df_valid['dist_km'].median():.1f}  "
          f"p95={df_valid['dist_km'].quantile(0.95):.1f}  "
          f"max={df_valid['dist_km'].max():.1f}")

    # 4. Stratified hex split
    print(f"\n{SEP}\n4. Stratified hex split (70/15/15 by Class8590)\n{SEP}")
    hex_summary = df_valid.groupby('hex_id').agg(
        n_pixels = ('label', 'count'),
        pattern  = ('Class8590', lambda x: x.mode()[0])
    ).reset_index()

    train_hexes, val_hexes, test_hexes = [], [], []
    for pat, grp in hex_summary.groupby('pattern'):
        hex_ids = grp['hex_id'].values.copy()
        rng_pat = np.random.default_rng(SEED + hash(pat) % 10000)
        rng_pat.shuffle(hex_ids)
        n    = len(hex_ids)
        n_tr = max(1, round(n * TRAIN_FRAC))
        n_va = max(1, round(n * VAL_FRAC))
        n_te = max(1, n - n_tr - n_va)
        if n_tr + n_va + n_te > n: n_tr = n - n_va - n_te
        train_hexes.extend(hex_ids[:n_tr].tolist())
        val_hexes.extend(hex_ids[n_tr:n_tr+n_va].tolist())
        test_hexes.extend(hex_ids[n_tr+n_va:n_tr+n_va+n_te].tolist())

    df_valid['split'] = 'UNASSIGNED'
    df_valid.loc[df_valid['hex_id'].isin(set(train_hexes)), 'split'] = 'train'
    df_valid.loc[df_valid['hex_id'].isin(set(val_hexes)),   'split'] = 'val'
    df_valid.loc[df_valid['hex_id'].isin(set(test_hexes)),  'split'] = 'test'

    # Verify
    tr = set(df_valid[df_valid['split']=='train']['hex_id'])
    te = set(df_valid[df_valid['split']=='test']['hex_id'])
    va = set(df_valid[df_valid['split']=='val']['hex_id'])
    assert len(tr & te) == 0, "Train/test hex overlap!"
    assert len(tr & va) == 0, "Train/val hex overlap!"
    assert len(va & te) == 0, "Val/test hex overlap!"
    assert (df_valid['split'] == 'UNASSIGNED').sum() == 0, "Unassigned pixels!"

    # 5. Print summary
    print(f"\n{'Split':<8} {'Pixels':>8} {'Hexagons':>10} {'label=1':>9} {'label=0':>9} {'%pos':>6}")
    print("-"*55)
    meta_splits = {}
    for s in ['train', 'val', 'test']:
        sub  = df_valid[df_valid['split'] == s]
        n1   = (sub['label'] == 1).sum()
        n0   = (sub['label'] == 0).sum()
        nhex = sub['hex_id'].nunique()
        print(f"{s:<8} {len(sub):>8,} {nhex:>10,} {n1:>9,} {n0:>9,} {100*n1/len(sub):>5.1f}%")
        meta_splits[s] = {
            'n_pixels':    int(len(sub)),
            'n_hexagons':  int(nhex),
            'n_label1':    int(n1),
            'n_label0':    int(n0),
            'pct_label1':  round(100*n1/len(sub), 1),
            'pattern_dist': sub['Class8590'].value_counts().to_dict()
        }
    print(f"\n✅ Spatial overlap train∩test: {len(tr & te)} hexagons")

    # 6. Save
    print(f"\n{SEP}\n5. Saving outputs to {OUT_DIR}\n{SEP}")
    cols_model = ['row', 'col', 'T', 'label', 'hex_id', 'Class8590']
    for s in ['train', 'val', 'test']:
        path = os.path.join(OUT_DIR, f"spatial_split_{s}.csv")
        df_valid[df_valid['split'] == s][cols_model].reset_index(drop=True).to_csv(path, index=False)
        print(f"  ✓ spatial_split_{s}.csv")

    master_path = os.path.join(OUT_DIR, "treino_balanceado_SPATIAL_SPLIT.csv")
    df_valid[cols_model + ['split', 'dist_km']].to_csv(master_path, index=False)
    print(f"  ✓ treino_balanceado_SPATIAL_SPLIT.csv")

    meta = {
        "split_method":    "hexagon_stratified_by_Class8590",
        "seed":             SEED,
        "fractions":        {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "source_file":      os.path.basename(CSV_PATH),
        "shapefile":        os.path.basename(SHP_PATH),
        "projection":       "South_America_Albers_Equal_Area_Conic (SAD69)",
        "assignment":       "nearest hexagon centroid (Albers Euclidean distance)",
        "total_samples":    int(len(df_valid)),
        "spatial_overlap_train_test_hexagons": 0,
        "splits": meta_splits
    }
    meta_path = os.path.join(OUT_DIR, "spatial_split_metadata.json")
    with open(meta_path, 'w') as f: json.dump(meta, f, indent=2)
    print(f"  ✓ spatial_split_metadata.json")
    print(f"\n{'='*60}\n✅ Done.\n{'='*60}")


if __name__ == "__main__":
    main()
