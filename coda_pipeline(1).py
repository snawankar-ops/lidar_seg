#!/usr/bin/env python3
"""
coda_pipeline.py - everything in one file:
  ray ground filter -> KD-tree Euclidean clustering -> min-area 3D boxes,
  run on CODa frames, scored against CODa labels, with top-down images.

  python3 coda_pipeline.py --root <folder with 3d_raw, 3d_bbox, ...> --seq 0 --count 20 --step 10 --png 5

Needs: numpy, scipy, matplotlib (numba optional, speeds up the ray filter).
Frame convention: x forward, y left, z up (same as CODa).
"""
# ======================================================================
# RAY GROUND FILTER
# ======================================================================
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _ray_kernel(r, z, ray, z_ref, tan_g, tan_l, tan_i, min_h, reclass, out):
    """r, z, ray sorted by (ray, r). Writes True to out[i] for ground points."""
    cur = -1
    prev_r = 0.0
    prev_z = 0.0
    prev_g = True
    first = True
    an_r = 0.0
    an_z = 0.0
    for i in range(len(r)):
        if ray[i] != cur:                     # new ray: start at the ground below the sensor
            cur = ray[i]
            prev_r = 0.0
            prev_z = z_ref
            prev_g = True
            first = True
            an_r = 0.0
            an_z = z_ref
        ri = r[i]
        zi = z[i]
        dr = ri - prev_r

        g_thr = tan_g * ri                    # global cone (vs ground level below sensor)
        if g_thr < min_h:
            g_thr = min_h
        g_pass = abs(zi - z_ref) <= g_thr

        if prev_g:
            slope = tan_i if first else tan_l
            l_thr = slope * dr                # local cone (vs previous point)
            if l_thr < min_h:
                l_thr = min_h
            a_thr = slope * (ri - an_r)       # anti-creep: also vs the ground anchor
            if a_thr < min_h:
                a_thr = min_h
            if abs(zi - prev_z) <= l_thr and abs(zi - an_z) <= a_thr:
                g = True
            elif g_pass and dr >= reclass:
                g = True
            else:
                g = False
        else:
            # previous point was non-ground: recover only if inside the global cone
            # AND consistent with the last ground anchor
            l_thr = tan_l * (ri - an_r)
            if l_thr < min_h:
                l_thr = min_h
            g = g_pass and abs(zi - an_z) <= l_thr

        if g and ri - an_r >= reclass and abs(zi - an_z) <= tan_l * (ri - an_r):
            an_r = ri                         # anchor advances only along a real slope
            an_z = zi
        prev_r = ri
        prev_z = zi
        prev_g = g
        first = False
        out[i] = g


_kernel = njit(cache=True)(_ray_kernel) if njit is not None else _ray_kernel


def ray_ground_mask(xyz, sensor_height=None, radial_div_deg=1.0,
                    general_max_slope_deg=10.0, local_max_slope_deg=10.0,
                    initial_max_slope_deg=3.0, min_height=0.15, reclass_dist=0.2):
    """Returns a boolean mask, True = ground."""
    n = len(xyz)
    if n == 0:
        return np.zeros(0, bool)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.hypot(x, y)                        # horizontal radius, not 3D range

    if sensor_height is None:                 # estimate ground level from low points nearby
        near = r < 10.0
        z_ref = float(np.percentile(z[near] if near.sum() > 100 else z, 5))
    else:
        z_ref = -float(sensor_height)

    n_rays = int(round(360.0 / radial_div_deg))
    ray = ((np.arctan2(y, x) + np.pi) / (2 * np.pi) * n_rays).astype(np.int64)
    ray = np.minimum(ray, n_rays - 1)
    order = np.lexsort((r, ray))              # by ray, then near -> far
    rs, zs, rays = r[order], z[order], ray[order]

    args = (z_ref,
            float(np.tan(np.radians(general_max_slope_deg))),
            float(np.tan(np.radians(local_max_slope_deg))),
            float(np.tan(np.radians(initial_max_slope_deg))),
            float(min_height), float(reclass_dist))
    if njit is not None:
        out = np.zeros(n, np.bool_)
        _kernel(rs, zs, rays, *args, out)
    else:                                     # lists index faster than numpy scalars
        out = [False] * n
        _kernel(rs.tolist(), zs.tolist(), rays.tolist(), *args, out)
        out = np.asarray(out, bool)

    mask = np.zeros(n, bool)
    mask[order] = out
    return mask

# ======================================================================
# EUCLIDEAN CLUSTERING (KD-TREE)
# ======================================================================
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _labels_fast(xyz, dist):
    n = len(xyz)
    tree = cKDTree(xyz)
    pairs = tree.query_pairs(dist, output_type='ndarray')
    if len(pairs) == 0:
        return np.arange(n)
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab


def _labels_region(xyz, dist):
    """Literal region-growing version (processed flags + stack). Slower, for reference."""
    n = len(xyz)
    tree = cKDTree(xyz)
    processed = np.zeros(n, bool)
    lab = np.full(n, -1, np.int64)
    cid = 0
    for seed in range(n):
        if processed[seed]:
            continue
        processed[seed] = True
        lab[seed] = cid
        stack = [seed]
        while stack:
            i = stack.pop()
            for j in tree.query_ball_point(xyz[i], dist):
                if not processed[j]:
                    processed[j] = True
                    lab[j] = cid
                    stack.append(j)
        cid += 1
    return lab


def euclidean_clustering(xyz, distance_threshold=0.5, min_size=10, max_size=None, method="fast"):
    """xyz: (N,3). Returns a list of index arrays (indices into xyz), one per cluster."""
    if len(xyz) == 0:
        return []
    lab = (_labels_fast if method == "fast" else _labels_region)(xyz, distance_threshold)
    counts = np.bincount(lab)
    order = np.argsort(lab, kind="stable")
    groups = np.split(order, np.cumsum(counts)[:-1])
    return [g for g, c in zip(groups, counts)
            if c >= min_size and (max_size is None or c <= max_size)]

# ======================================================================
# MIN-AREA 3D BOX FITTING
# ======================================================================
import numpy as np
from scipy.spatial import ConvexHull


def min_area_rect(xy):
    """xy: (N,2). Returns cx, cy, length, width, yaw (yaw in [-pi/2, pi/2))."""
    xy = np.asarray(xy, float)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    fallback = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2,
                max(hi[0] - lo[0], 1e-3), max(hi[1] - lo[1], 1e-3), 0.0)
    if len(xy) < 3:
        return fallback
    try:
        hull = ConvexHull(xy)
    except Exception:                         # collinear / degenerate points
        return fallback
    h = xy[hull.vertices]
    edges = np.roll(h, -1, axis=0) - h
    angles = np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2), 6))
    best = None
    for a in angles:
        c, s = np.cos(a), np.sin(a)
        rot = np.array([[c, s], [-s, c]])     # rotate by -a
        p = h @ rot.T
        mn, mx = p.min(axis=0), p.max(axis=0)
        area = (mx[0] - mn[0]) * (mx[1] - mn[1])
        if best is None or area < best[0]:
            best = (area, a, rot, mn, mx)
    _, a, rot, mn, mx = best
    center = rot.T @ ((mn + mx) / 2.0)
    dx, dy = mx[0] - mn[0], mx[1] - mn[1]
    if dx >= dy:
        length, width, yaw = dx, dy, a
    else:
        length, width, yaw = dy, dx, a + np.pi / 2
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    return float(center[0]), float(center[1]), float(max(length, 1e-3)), float(max(width, 1e-3)), float(yaw)


def fit_box_3d(points):
    """points: (N,3) of one cluster -> dict(cx, cy, cz, l, w, h, yaw)."""
    cx, cy, l, w, yaw = min_area_rect(points[:, :2])
    zmin, zmax = float(points[:, 2].min()), float(points[:, 2].max())
    return dict(cx=cx, cy=cy, cz=(zmin + zmax) / 2, l=l, w=w, h=max(zmax - zmin, 1e-3), yaw=yaw)


def box_corners_xy(b):
    c, s = np.cos(b['yaw']), np.sin(b['yaw'])
    local = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], float) * [b['l'] / 2, b['w'] / 2]
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + [b['cx'], b['cy']]


def points_in_box(xyz, b, margin=0.0):
    """Boolean mask of points inside a (yaw-rotated) box."""
    d = xyz[:, :2] - [b['cx'], b['cy']]
    c, s = np.cos(b['yaw']), np.sin(b['yaw'])
    lx = d[:, 0] * c + d[:, 1] * s
    ly = -d[:, 0] * s + d[:, 1] * c
    return ((np.abs(lx) <= b['l'] / 2 + margin) & (np.abs(ly) <= b['w'] / 2 + margin) &
            (np.abs(xyz[:, 2] - b['cz']) <= b['h'] / 2 + margin))

# ======================================================================
# CODa READER
# ======================================================================
import glob
import json
import os
import re
import numpy as np

# 3d_semantic ids: 0 Unlabeled (= not terrain, i.e. objects/walls), 1-19 + 21-23 terrain types,
# 20 Stairs and 24 Unknown are ignored when scoring the ground filter.
GROUND_IDS = [i for i in range(1, 24) if i != 20]
IGNORE_IDS = [20, 24]


class CODa:
    def __init__(self, root, seq, cloud="comp"):
        self.root = os.path.expanduser(root)
        self.seq = str(seq)
        self.cloud = cloud                    # "comp" (ego-compensated) or "raw"

    def _p(self, *parts):
        return os.path.join(self.root, *parts)

    def _kinds(self):
        return [self.cloud, "raw"] if self.cloud != "raw" else ["raw"]

    def cloud_path(self, frame):
        for k in self._kinds():
            p = self._p(f"3d_{k}", "os1", self.seq, f"3d_{k}_os1_{self.seq}_{frame}.bin")
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def _frame_numbers(paths, ext):
        out = []
        for p in paths:
            m = re.search(r"_(\d+)\." + ext + r"$", os.path.basename(p))
            if m:
                out.append(int(m.group(1)))
        return sorted(set(out))

    def cloud_frames(self):
        for k in self._kinds():
            fr = self._frame_numbers(glob.glob(self._p(f"3d_{k}", "os1", self.seq, "*.bin")), "bin")
            if fr:
                return fr
        return []

    def bbox_path(self, frame):
        name = f"3d_bbox_os1_{self.seq}_{frame}.json"
        for p in (self._p("3d_bbox", "os1", self.seq, name), self._p("3d_bbox", "os1", name)):
            if os.path.exists(p):
                return p
        return None

    def annotated_frames(self):
        paths = (glob.glob(self._p("3d_bbox", "os1", self.seq, "*.json")) +
                 glob.glob(self._p("3d_bbox", "os1", f"3d_bbox_os1_{self.seq}_*.json")))
        return self._frame_numbers(paths, "json")

    def load_cloud(self, frame):
        p = self.cloud_path(frame)
        if p is None:
            return None
        return np.fromfile(p, dtype=np.float32).reshape(-1, 4)

    def load_boxes(self, frame):
        p = self.bbox_path(frame)
        if p is None:
            return None
        with open(p) as f:
            data = json.load(f)
        out = []
        for b in data.get("3dbbox", []):
            try:
                out.append(dict(cx=b["cX"], cy=b["cY"], cz=b["cZ"], l=b["l"], w=b["w"], h=b["h"],
                                yaw=b["y"], cls=b.get("classId", "?"), iid=b.get("instanceId", "")))
            except KeyError:
                continue
        return out

    def load_semantic(self, frame):
        p = self._p("3d_semantic", "os1", self.seq, f"3d_semantic_os1_{self.seq}_{frame}.bin")
        if not os.path.exists(p):
            return None
        return np.fromfile(p, dtype=np.uint8)

# ======================================================================
# PIPELINE + SCORING + CLI
# ======================================================================
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree



def voxel_downsample(xyz, v):
    if v <= 0 or len(xyz) == 0:
        return np.arange(len(xyz))
    k = np.floor(xyz / v).astype(np.int64)
    k -= k.min(axis=0)
    dims = k.max(axis=0) + 1
    flat = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]    # one int per voxel: much faster than unique(axis=0)
    _, idx = np.unique(flat, return_index=True)
    return np.sort(idx)


def in_roi(xyz, a):
    """Region of interest: radial range AND an x/y/z box in the sensor frame (x fwd, y left, z up)."""
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    return ((r > a.min_range) & (r < a.max_range) &
            (xyz[:, 0] >= a.roi_x[0]) & (xyz[:, 0] <= a.roi_x[1]) &
            (xyz[:, 1] >= a.roi_y[0]) & (xyz[:, 1] <= a.roi_y[1]) &
            (xyz[:, 2] >= a.roi_z[0]) & (xyz[:, 2] <= a.roi_z[1]))


def box_in_roi(b, a):
    return (a.roi_x[0] <= b["cx"] <= a.roi_x[1] and a.roi_y[0] <= b["cy"] <= a.roi_y[1] and
            np.hypot(b["cx"], b["cy"]) < a.max_range)


def segment(pts, a):
    """pts: (N,4) raw CODa cloud -> dict with filtered xyz, ground mask, cluster labels, boxes.
    Order (cheapest first): ROI crop -> voxel downsample -> ground filter -> clustering -> boxes."""
    tp0 = time.perf_counter()
    xyz = pts[:, :3].astype(np.float64)
    valid = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 1e-6)   # drop (0,0,0) no-returns
    valid &= in_roi(xyz, a)
    idx = np.where(valid)[0]
    n_roi = len(idx)
    keep = idx[voxel_downsample(xyz[idx], a.voxel)]       # raw indices of the points we keep
    xyz = xyz[keep]

    t0 = time.perf_counter()
    ground = ray_ground_mask(xyz, sensor_height=a.sensor_height, radial_div_deg=a.ray_div,
                             general_max_slope_deg=a.slope_deg, local_max_slope_deg=a.slope_deg,
                             initial_max_slope_deg=a.init_slope_deg, min_height=a.min_h)
    t1 = time.perf_counter()

    gz = xyz[ground]
    near = gz[np.hypot(gz[:, 0], gz[:, 1]) < 10.0] if len(gz) else gz
    if len(near) > 50:
        z_ground = float(np.median(near[:, 2]))
    elif len(gz):
        z_ground = float(np.median(gz[:, 2]))
    elif len(xyz):
        z_ground = float(np.percentile(xyz[:, 2], 5))
    else:
        z_ground = 0.0
    obs = np.where((~ground) & (xyz[:, 2] - z_ground < a.max_height))[0]

    clusters = euclidean_clustering(xyz[obs], a.eps, a.min_size)
    boxes = []
    obs_label = np.full(len(obs), -1, np.int64)
    for c in clusters:
        b = fit_box_3d(xyz[obs[c]])
        if max(b["l"], b["w"]) > a.max_box:
            continue
        b["id"] = len(boxes)
        b["n"] = int(len(c))
        obs_label[c] = b["id"]
        boxes.append(b)
    t2 = time.perf_counter()

    point_label = np.full(len(xyz), -1, np.int64)          # cluster id of every non-ground point
    point_label[obs] = obs_label

    return dict(keep=keep, xyz=xyz, ground=ground, point_label=point_label, boxes=boxes,
                ms_pre=(t0 - tp0) * 1000, ms_ground=(t1 - t0) * 1000, ms_cluster=(t2 - t1) * 1000,
                z_ground=z_ground, n_raw=len(pts), n_roi=n_roi, n_vox=len(xyz))


def save_png(path, seg, gt, title, a):
    fig, ax = plt.subplots(figsize=(9, 9))
    xyz, g, lab = seg["xyz"], seg["ground"], seg["point_label"]
    ax.scatter(xyz[g, 0], xyz[g, 1], s=0.2, c="#c8c8c8")
    other = (~g) & (lab < 0)
    ax.scatter(xyz[other, 0], xyz[other, 1], s=0.5, c="#4060ff")
    cl = lab >= 0
    ax.scatter(xyz[cl, 0], xyz[cl, 1], s=1.5, c=lab[cl] % 20, cmap="tab20")
    for b in seg["boxes"]:
        c = box_corners_xy(b)
        ax.plot(*np.vstack([c, c[:1]]).T, color="lime", lw=1.3)
        ax.plot([b["cx"], b["cx"] + 0.5 * b["l"] * np.cos(b["yaw"])],
                [b["cy"], b["cy"] + 0.5 * b["l"] * np.sin(b["yaw"])], color="lime", lw=1.3)
    for b in (gt or []):
        if not box_in_roi(b, a):
            continue
        c = box_corners_xy(b)
        ax.plot(*np.vstack([c, c[:1]]).T, color="red", lw=1.0, ls="--")
        ax.text(b["cx"], b["cy"], b["cls"], color="red", fontsize=6)
    ax.plot(0, 0, "k^", ms=8)
    ax.add_patch(plt.Rectangle((a.roi_x[0], a.roi_y[0]), a.roi_x[1] - a.roi_x[0], a.roi_y[1] - a.roi_y[0],
                               fill=False, ec="black", lw=1.0, ls=":"))
    ax.set_xlim(-a.max_range, a.max_range)
    ax.set_ylim(-a.max_range, a.max_range)
    ax.set_aspect("equal")
    ax.set_title(title + "\ngray=ground  blue=non-ground unused  colored=clusters  green=our boxes  red dashed=CODa GT  dotted black=ROI",
                 fontsize=8)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder that contains 3d_raw, 3d_bbox, ...")
    ap.add_argument("--seq", default="0")
    ap.add_argument("--cloud", default="comp", choices=["comp", "raw"])
    ap.add_argument("--start", type=int, default=0, help="index into the frame list")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--all-frames", action="store_true", help="use all frames, not only annotated ones")
    ap.add_argument("--png", type=int, default=5, help="save images for the first N frames")
    ap.add_argument("--out", default="coda_out")
    ap.add_argument("--dump", action="store_true", help="also write detections as CODa-style json")
    # preprocessing
    ap.add_argument("--min-range", type=float, default=0.8)
    ap.add_argument("--max-range", type=float, default=30.0)
    # region of interest in the sensor frame (x forward, y left, z up), meters. Points outside are dropped first.
    ap.add_argument("--roi-x", type=float, nargs=2, default=[-10.0, 30.0], metavar=("MIN", "MAX"))
    ap.add_argument("--roi-y", type=float, nargs=2, default=[-15.0, 15.0], metavar=("MIN", "MAX"))
    ap.add_argument("--roi-z", type=float, nargs=2, default=[-3.0, 3.0], metavar=("MIN", "MAX"),
                    help="z relative to the sensor")
    ap.add_argument("--no-roi", action="store_true", help="only use the circular --max-range limit")
    ap.add_argument("--max-height", type=float, default=2.5, help="ignore points higher than this above ground")
    # ray ground filter
    ap.add_argument("--sensor-height", type=float, default=None, help="LiDAR height above ground in m (None = estimate)")
    ap.add_argument("--ray-div", type=float, default=1.0)
    ap.add_argument("--slope-deg", type=float, default=10.0)
    ap.add_argument("--init-slope-deg", type=float, default=3.0)
    ap.add_argument("--min-h", type=float, default=0.15)
    # clustering / boxes
    ap.add_argument("--voxel", type=float, default=0.1,
                    help="voxel size in m, applied BEFORE the ground filter (0 = off). Bigger = faster, coarser")
    ap.add_argument("--min-gt-pts", type=int, default=8,
                    help="ignore CODa boxes with fewer than this many (downsampled) points inside")
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--min-size", type=int, default=8)
    ap.add_argument("--max-box", type=float, default=12.0)
    a = ap.parse_args()
    if a.no_roi:
        a.roi_x, a.roi_y, a.roi_z = [-a.max_range, a.max_range], [-a.max_range, a.max_range], [-10.0, 10.0]

    ds = CODa(a.root, a.seq, a.cloud)
    frames = ds.cloud_frames()
    if not frames:
        raise SystemExit(f"No point clouds found under {ds.root}/3d_{a.cloud}/os1/{a.seq} (or 3d_raw). Check --root and --seq.")
    ann = [f for f in ds.annotated_frames() if f in set(frames)]
    if ann and not a.all_frames:
        frames = ann
    frames = frames[a.start::a.step][:a.count]
    os.makedirs(a.out, exist_ok=True)
    print(f"sequence {a.seq}: {len(frames)} frames selected "
          f"({'annotated frames' if ann and not a.all_frames else 'all frames'})")

    tp = fp = fn = 0
    ms_p, ms_g, ms_c, nbox = [], [], [], []
    n_raw, n_roi, n_vox = [], [], []
    cls_stats = defaultdict(lambda: [0, 0])
    center_err = []
    sem_frames = 0

    for k, f in enumerate(frames):
        pts = ds.load_cloud(f)
        seg = segment(pts, a)
        ms_p.append(seg["ms_pre"])
        n_raw.append(seg["n_raw"])
        n_roi.append(seg["n_roi"])
        n_vox.append(seg["n_vox"])
        ms_g.append(seg["ms_ground"])
        ms_c.append(seg["ms_cluster"])
        nbox.append(len(seg["boxes"]))

        sem = ds.load_semantic(f)
        if sem is not None and len(sem) == len(pts):
            s = sem[seg["keep"]]
            gt_g = np.isin(s, GROUND_IDS)
            gt_n = (s == 0)
            m = gt_g | gt_n
            pg = seg["ground"]
            tp += int((pg & gt_g & m).sum())
            fp += int((pg & gt_n).sum())
            fn += int(((~pg) & gt_g).sum())
            sem_frames += 1

        gt = ds.load_boxes(f)
        for b in (gt or []):
            if not box_in_roi(b, a):
                continue
            inside = points_in_box(seg["xyz"], b, margin=0.1)
            if inside.sum() < a.min_gt_pts:
                continue
            labs = seg["point_label"][inside]
            found = (labs >= 0).mean() >= 0.5
            cls_stats[b["cls"]][1] += 1
            if found:
                cls_stats[b["cls"]][0] += 1
                best = np.bincount(labs[labs >= 0]).argmax()
                db = seg["boxes"][best]
                center_err.append(float(np.hypot(db["cx"] - b["cx"], db["cy"] - b["cy"])))

        if a.dump:
            det = [dict(classId="Unknown", instanceId=f"Unknown:{b['id']}", labelAttributes={},
                        cX=b["cx"], cY=b["cy"], cZ=b["cz"], h=b["h"], l=b["l"], w=b["w"],
                        r=0.0, p=0.0, y=b["yaw"]) for b in seg["boxes"]]
            with open(f"{a.out}/det_{a.seq}_{f}.json", "w") as fh:
                json.dump({"3dbbox": det}, fh)
        if k < a.png:
            save_png(f"{a.out}/frame_{a.seq}_{f}.png", seg, gt, f"seq {a.seq} frame {f}", a)

    print(f"\nframes: {len(frames)}   images/json in: {a.out}/")
    print(f"points/frame: raw {np.mean(n_raw):.0f} -> after ROI {np.mean(n_roi):.0f} -> after voxel {np.mean(n_vox):.0f}")
    print(f"speed:   ROI+voxel {np.mean(ms_p):.0f} ms   ground filter {np.mean(ms_g):.0f} ms   "
          f"clustering+boxes {np.mean(ms_c):.0f} ms   boxes/frame {np.mean(nbox):.1f}")
    if sem_frames:
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        print(f"ground vs CODa 3d_semantic ({sem_frames} frames):  precision {100 * prec:.1f}%   "
              f"recall {100 * rec:.1f}%   F1 {100 * 2 * prec * rec / max(prec + rec, 1e-9):.1f}%")
        print("   low precision = obstacle points swallowed as ground;  low recall = floor leaking into obstacles")
    else:
        print("ground scoring: no 3d_semantic files for the selected frames (only ~5K frames have them)")
    tot = sum(v[1] for v in cls_stats.values())
    if tot:
        fnd = sum(v[0] for v in cls_stats.values())
        print(f"\nCODa 3D boxes found (>=50% of their points land in a cluster): {fnd}/{tot} = {100 * fnd / tot:.1f}%")
        for c, (x, y) in sorted(cls_stats.items(), key=lambda kv: -kv[1][1])[:12]:
            print(f"   {c:28s} {x:4d}/{y:<4d} {100 * x / y:5.1f}%")
        if center_err:
            print(f"center error of our box vs CODa box (found objects): median {np.median(center_err):.2f} m, "
                  f"mean {np.mean(center_err):.2f} m   (our box only covers the visible side, so some error is expected)")
    else:
        print("\nno CODa 3D boxes available for the selected frames")


if __name__ == "__main__":
    main()
