#!/usr/bin/env python3
"""
live_pipeline.py - everything in one file, for a live lidar (ROS 2):
  PointCloud2 topic -> ray ground filter -> KD-tree Euclidean clustering -> min-area 3D boxes
  -> publishes boxes (MarkerArray) and clustered obstacle points.

  python3 live_pipeline.py --ros-args -p topic:=/ouster/points -p sensor_height:=0.9

Needs: numpy, scipy, ROS 2 (rclpy, sensor_msgs_py) (numba optional, speeds up the ray filter).
Frame convention: x forward, y left, z up (ROS REP-103, same as CODa).
Output frame_id / stamp = those of the input cloud.
"""
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray

try:
    from numba import njit
except ImportError:
    njit = None

# ======================================================================
# VEHICLE GEOMETRY (from vehicle.yaml - built-in copy; pass -p vehicle_yaml:=/path/vehicle.yaml to override)
# ======================================================================
VEHICLE = dict(
    base_link_height=0.23,                       # base_link (rear-axle centre) above the ground
    footprint=[[-0.83, -0.72], [2.80, -0.72], [2.80, 0.72], [-0.83, 0.72]],   # in base_link, x fwd, y left
    ouster_lidar_z=0.038195,                     # os_*_sensor -> os_*_lidar (cloud frame is the lidar frame)
    sensors={                                    # base_link -> os_*_sensor: x, y, z, yaw
        "os_top": (1.94, 0.0, 1.617, 0.0),
        "os_front": (2.716, 0.655, 0.327, 0.785398),
        "os_back": (-0.75, -0.57, 0.282, -2.356194),
    },
)


def load_vehicle(path):
    """Read the same fields from vehicle.yaml (needs PyYAML, which ROS 2 installs)."""
    import yaml
    with open(path) as f:
        y = yaml.safe_load(f)
    return dict(
        base_link_height=float(y["geometry"]["base_link_height"]),
        footprint=[list(map(float, p)) for p in y["footprint"]],
        ouster_lidar_z=float(y["ouster_lidar_z"]),
        sensors={k: (*map(float, v["xyz"]), float(v["yaw"]))
                 for k, v in y["sensors"].items() if k.startswith("os_")})


def sensor_mount(veh, name):
    """-> (tx, ty, tz, yaw): cloud (os_*_lidar) frame in base_link, plus its height above the ground."""
    x, y, z, yaw = veh["sensors"][name]
    tz = z + veh["ouster_lidar_z"]
    return dict(tx=x, ty=y, tz=tz, yaw=yaw, height=tz + veh["base_link_height"])


def to_base(xyz, m):
    """Cloud-frame points -> base_link frame (yaw + translation; mounts are assumed level)."""
    c, s = np.cos(m["yaw"]), np.sin(m["yaw"])
    return np.stack([m["tx"] + xyz[:, 0] * c - xyz[:, 1] * s,
                     m["ty"] + xyz[:, 0] * s + xyz[:, 1] * c,
                     m["tz"] + xyz[:, 2]], axis=1)

# ======================================================================
# RAY GROUND FILTER
# ======================================================================


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
    if njit is None:
        # No numba: the python loop is ~100x slower and is the #1 cause of lag.
        return _ground_mask_vectorized(xyz, r, z_ref, args[1], min_height)
    out = np.zeros(n, np.bool_)
    _kernel(rs, zs, rays, *args, out)

    mask = np.zeros(n, bool)
    mask[order] = out
    return mask


def _ground_mask_vectorized(xyz, r, z_ref, tan_g, min_h, cell=0.5):
    """Fast numpy-only fallback: ground = close to the lowest point of its 0.5 m cell AND inside the
    global cone around the known ground level. Less precise than the ray filter on slopes."""
    z = xyz[:, 2]
    k = np.floor(xyz[:, :2] / cell).astype(np.int64)
    k -= k.min(axis=0)
    flat = k[:, 0] * (k[:, 1].max() + 1) + k[:, 1]
    _, inv = np.unique(flat, return_inverse=True)
    zmin = np.full(inv.max() + 1, np.inf)
    np.minimum.at(zmin, inv, z)
    thr = np.maximum(tan_g * r, min_h)
    return ((z - zmin[inv]) <= min_h) & (np.abs(z - z_ref) <= thr)

# ======================================================================
# EUCLIDEAN CLUSTERING (KD-TREE)
# ======================================================================


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
# PREPROCESSING + SEGMENTATION
# ======================================================================
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
    r = np.hypot(xyz[:, 0], xyz[:, 1])        # range limits are from the sensor
    m = getattr(a, "mount", None)
    b = to_base(xyz, m) if m else xyz         # ROI box is in base_link (vehicle) frame when a mount is known
    ok = ((r > a.min_range) & (r < a.max_range) &
          (b[:, 0] >= a.roi_x[0]) & (b[:, 0] <= a.roi_x[1]) &
          (b[:, 1] >= a.roi_y[0]) & (b[:, 1] <= a.roi_y[1]) &
          (b[:, 2] >= a.roi_z[0]) & (b[:, 2] <= a.roi_z[1]))
    ego = getattr(a, "ego_box", None)
    if m and ego is not None:                 # cut out the vehicle's own body (footprint + margin)
        body = ((b[:, 0] >= ego[0]) & (b[:, 0] <= ego[1]) &
                (b[:, 1] >= ego[2]) & (b[:, 1] <= ego[3]) &
                (b[:, 2] >= a.ego_z[0]) & (b[:, 2] <= a.ego_z[1]))
        ok &= ~body
    return ok


def _fast_clusters(pts, eps, min_size, cv):
    """Clustering on a coarser grid (cell cv): query_pairs cost grows with neighbours^2, so linking one
    representative per cv-cell and mapping labels back is several times cheaper at the same eps."""
    if cv <= 0 or len(pts) == 0:
        return euclidean_clustering(pts, eps, min_size)
    k = np.floor(pts / cv).astype(np.int64)
    k -= k.min(axis=0)
    dims = k.max(axis=0) + 1
    flat = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, first, inv = np.unique(flat, return_index=True, return_inverse=True)
    lab_rep = _labels_fast(pts[first], eps)
    lab = lab_rep[inv]
    counts = np.bincount(lab)
    order = np.argsort(lab, kind="stable")
    groups = np.split(order, np.cumsum(counts)[:-1])
    return [g for g, c in zip(groups, counts) if c >= min_size]


def segment(pts, a):
    """pts: (N,3+) lidar cloud (only x,y,z used) -> dict with filtered xyz, ground mask, cluster labels, boxes.
    Order (cheapest first): ROI crop -> voxel downsample -> ground filter -> clustering -> boxes."""
    tp0 = time.perf_counter()
    p3 = pts[:, :3]                                        # stay float32 until the ROI has thinned the cloud
    r2 = p3[:, 0] * p3[:, 0] + p3[:, 1] * p3[:, 1]
    cand = np.flatnonzero((r2 > a.min_range ** 2) & (r2 < a.max_range ** 2) &   # cheap range gate first
                          np.isfinite(p3[:, 2]) & ((r2 + p3[:, 2] * p3[:, 2]) > 1e-9))   # drops NaN and (0,0,0)
    cand = cand[in_roi(p3[cand], a)]                       # base-frame ROI + body cut-out on the survivors only
    xyz = np.zeros((len(pts), 3))                          # float64 only for what is left
    idx = cand
    xyz[idx] = p3[idx]
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

    clusters = _fast_clusters(xyz[obs], a.eps, a.min_size, getattr(a, "cluster_voxel", 0.0))
    boxes = []
    obs_label = np.full(len(obs), -1, np.int64)
    split_len = float(getattr(a, "split_len", 0.0))
    max_len = 0.0
    for c in clusters:
        pc = xyz[obs[c]]
        b = fit_box_3d(pc)
        length = max(b["l"], b["w"])
        max_len = max(max_len, length)
        if split_len > 0 and length > split_len:
            # Long thing (wall, fence, hedge, or a wall merged with a bit of leaked floor): box it in
            # equal pieces along its long axis. Never dropped, however long (max_box is no longer used).
            yaw = b["yaw"] if b["l"] >= b["w"] else b["yaw"] + np.pi / 2
            proj = (pc[:, 0] - b["cx"]) * np.cos(yaw) + (pc[:, 1] - b["cy"]) * np.sin(yaw)
            n_seg = min(int(np.ceil(length / split_len)), 16)     # cap the marker count
            piece = np.minimum(((proj - proj.min()) / (length / n_seg + 1e-9)).astype(int), n_seg - 1)
            for k in range(n_seg):
                sel = np.where(piece == k)[0]
                if len(sel) < a.min_size:
                    continue
                bk = fit_box_3d(pc[sel])
                bk["id"] = len(boxes)
                bk["n"] = int(len(sel))
                obs_label[c[sel]] = bk["id"]
                boxes.append(bk)
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
                z_ground=z_ground, n_raw=len(pts), n_roi=n_roi, n_vox=len(xyz),
                n_clusters=len(clusters), max_len=max_len)

# ======================================================================
# ROS 2 LIVE NODE
# ======================================================================
# sensor_msgs/PointField datatype -> numpy dtype
_PF = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


# ----------------------------------------------------------------------
# PointCloud2 -> (N,3) float32. Works for any driver layout (Ouster, Velodyne,
# Livox...) because it reads x/y/z by their declared offsets.
# ----------------------------------------------------------------------
def cloud_to_xyz(msg):
    end = ">" if msg.is_bigendian else "<"
    f = {fl.name: fl for fl in msg.fields}
    dt = np.dtype(dict(
        names=["x", "y", "z"],
        formats=[np.dtype(end + _PF[f[k].datatype]) for k in "xyz"],
        offsets=[f[k].offset for k in "xyz"],
        itemsize=msg.point_step))
    raw = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.row_step)
    raw = np.ascontiguousarray(raw[:, :msg.width * msg.point_step])   # drop row padding
    arr = raw.view(dt).reshape(-1)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)


def yaw_to_quat(yaw):
    return 0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))


def xyz_to_cloud(header, xyz):
    """(N,3) -> PointCloud2 with plain numpy (much faster than point_cloud2.create_cloud_xyz32)."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    m = PointCloud2()
    m.header = header
    m.height, m.width = 1, len(xyz)
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, n in enumerate("xyz")]
    m.is_bigendian = False
    m.point_step = 12
    m.row_step = 12 * len(xyz)
    m.is_dense = True
    m.data = xyz.tobytes()
    return m


class BoxTracker:
    """Constant-velocity tracker. Predicts where each box should be this frame, matches detections to
    the prediction (so fast movers - bikes - are not lost), keeps stable ids, coasts through missed
    frames, and can push the output forward in time to hide the processing delay."""

    def __init__(self, gate=2.5, min_hits=1, max_miss=4, alpha=0.8, max_speed=15.0):
        self.gate, self.min_hits, self.max_miss, self.alpha = gate, min_hits, max_miss, alpha
        self.max_speed = max_speed
        self.tracks = []
        self.next_id = 0
        self.t = None

    def update(self, boxes, t, lead=0.0):
        dt = 0.1 if self.t is None else float(np.clip(t - self.t, 0.02, 0.5))
        self.t = t
        T, D = self.tracks, boxes
        for tr in T:                                  # predict
            tr["px"] = tr["cx"] + tr["vx"] * dt
            tr["py"] = tr["cy"] + tr["vy"] * dt
        pairs = sorted(((np.hypot(tr["px"] - d["cx"], tr["py"] - d["cy"]), i, j)
                        for i, tr in enumerate(T) for j, d in enumerate(D)), key=lambda p: p[0])
        used_t, used_d = set(), set()
        for dist, i, j in pairs:
            if dist > self.gate:
                break
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            tr, d, al = T[i], D[j], self.alpha
            ox, oy = tr["cx"], tr["cy"]
            tr["cx"] = tr["px"] + al * (d["cx"] - tr["px"])      # high alpha = little lag
            tr["cy"] = tr["py"] + al * (d["cy"] - tr["py"])
            vx, vy = (tr["cx"] - ox) / dt, (tr["cy"] - oy) / dt
            sp = np.hypot(vx, vy)
            if sp > self.max_speed:
                vx, vy = vx * self.max_speed / sp, vy * self.max_speed / sp
            tr["vx"] = 0.5 * tr["vx"] + 0.5 * vx
            tr["vy"] = 0.5 * tr["vy"] + 0.5 * vy
            for k in ("cz", "l", "w", "h"):
                tr[k] = 0.5 * d[k] + 0.5 * tr[k]
            # yaw has period pi -> blend on the doubled angle
            s = 0.5 * np.sin(2 * d["yaw"]) + 0.5 * np.sin(2 * tr["yaw"])
            c = 0.5 * np.cos(2 * d["yaw"]) + 0.5 * np.cos(2 * tr["yaw"])
            tr["yaw"] = float(0.5 * np.arctan2(s, c))
            tr["hits"] += 1
            tr["miss"] = 0
            tr["n"] = d["n"]
        for i, tr in enumerate(T):
            if i not in used_t:                       # coast on the prediction
                tr["cx"], tr["cy"] = tr["px"], tr["py"]
                tr["miss"] += 1
        for j, d in enumerate(D):
            if j not in used_d:
                T.append(dict(d, id=self.next_id, hits=1, miss=0, vx=0.0, vy=0.0, px=d["cx"], py=d["cy"]))
                self.next_id += 1
        self.tracks = [tr for tr in T if tr["miss"] <= self.max_miss]
        out = []
        for tr in self.tracks:
            if tr["hits"] >= self.min_hits:
                o = dict(tr)
                o["cx"] += tr["vx"] * lead            # show where it is NOW, not where it was at scan time
                o["cy"] += tr["vy"] * lead
                out.append(o)
        return out


class LiveNode(Node):
    def __init__(self):
        super().__init__("lidar_pipeline_live")

        # every knob from the old argparse section becomes a ROS parameter
        defaults = dict(
            topic="/os_front/points",
            sensor="auto",              # os_top | os_front | os_back; auto = guessed from the topic name
            vehicle_yaml="",            # optional path to vehicle.yaml; empty = built-in copy
            min_range=0.8, max_range=25.0,
            # ROI is in base_link (rear-axle centre, x fwd, y left, z up; ground is at z = -0.23)
            roi_x=[-6.0, 25.0], roi_y=[-12.0, 12.0], roi_z=[-0.8, 2.6],
            # vehicle body cut-out: footprint + margin, between these base_link heights
            ego_margin=0.15, ego_z=[-0.55, 1.95],
            max_height=2.2,              # vehicle is ~1.8-1.9 m tall: anything higher can be driven under
            sensor_height=0.0,           # 0.0 = taken from vehicle.yaml mount (axle height + z + lidar z)
            ray_div=1.0, slope_deg=10.0, init_slope_deg=3.0, min_h=0.15,
            # eps 0.8: sparse far rings on a wall are >0.6 m apart and would not link into one cluster
            voxel=0.15, eps=0.8, min_size=6,
            cluster_voxel=0.3,           # clustering grid (faster); 0 = cluster every voxel point
            split_len=4.0,              # clusters longer than this (walls) are boxed in pieces; 0 = off
            max_box=0.0,                 # unused now: long clusters are split, never dropped
            # temporal smoothing of boxes (stops flicker)
            track_gate=2.5, track_min_hits=1, track_max_miss=4, track_alpha=0.8,
            lead_extra=0.03,             # s added to the measured compute time when pushing boxes forward
            box_lifetime=1.0,            # long on purpose: removed boxes are DELETEd explicitly, so a short
                                         # lifetime only causes blinking if stamps lag RViz's clock
        )
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        p = lambda k: self.get_parameter(k).value
        self.a = SimpleNamespace(**{k: p(k) for k in defaults if k != "topic"})
        veh = load_vehicle(p("vehicle_yaml")) if p("vehicle_yaml") else VEHICLE
        name = p("sensor")
        if name == "auto":
            name = next((k for k in veh["sensors"] if k in p("topic")), None)
        if name in veh["sensors"]:
            m = sensor_mount(veh, name)
            self.a.mount = m
            fp = np.asarray(veh["footprint"], float)
            mg = self.a.ego_margin
            self.a.ego_box = (fp[:, 0].min() - mg, fp[:, 0].max() + mg,
                              fp[:, 1].min() - mg, fp[:, 1].max() + mg)
            if self.a.sensor_height <= 0:
                self.a.sensor_height = m["height"]
            self.get_logger().info(f"{name}: {m['height']:.3f} m above ground, yaw {m['yaw']:.3f} rad, "
                                   f"body cut-out x[{self.a.ego_box[0]:.2f},{self.a.ego_box[1]:.2f}] "
                                   f"y[{self.a.ego_box[2]:.2f},{self.a.ego_box[3]:.2f}]")
        else:
            self.a.mount = None
            self.get_logger().warn(f"unknown sensor for topic '{p('topic')}': set -p sensor:=os_front|os_top|"
                                   "os_back. ROI/body cut-out fall back to the raw sensor frame!")
            if self.a.sensor_height <= 0:
                self.a.sensor_height = None  # segment() treats None as "estimate"
        self.tracker = BoxTracker(self.a.track_gate, self.a.track_min_hits,
                                  self.a.track_max_miss, self.a.track_alpha)
        self.prev_ids = set()
        self.lead = 0.1

        if njit is None:
            self.get_logger().warn("numba NOT installed -> slow fallback ground filter. "
                                   "Run: pip install numba  (biggest single fix for lag)")
        # Warm up numba now so the first real frame isn't a 5-second compile stall.
        ray_ground_mask(np.random.rand(500, 3) * [20, 20, 0.1] - [10, 10, 1.8])

        # Sensor-style QoS: best effort, depth 1 -> if we're slower than the lidar,
        # old frames are dropped and we always process the newest one.
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub = self.create_subscription(PointCloud2, p("topic"), self.on_cloud, qos)
        self.pub_boxes = self.create_publisher(MarkerArray, "/detections/boxes", 1)
        self.pub_obs = self.create_publisher(PointCloud2, "/detections/obstacles", 1)
        self.get_logger().info(f"listening on {p('topic')}")

    def on_cloud(self, msg):
        t0 = time.perf_counter()
        pts = cloud_to_xyz(msg)
        seg = segment(pts, self.a)

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # How stale will these boxes be when they appear? = time spent on THIS frame so far (+ a bit for
        # publishing). Push each box forward along its velocity by that much so a fast bike's box sits on it.
        lead = (time.perf_counter() - t0) + self.a.lead_extra
        tracks = self.tracker.update(seg["boxes"], stamp, lead)
        obs = seg["point_label"] >= 0
        self.pub_obs.publish(xyz_to_cloud(msg.header, seg["xyz"][obs]))
        self.publish_boxes(msg.header, tracks)

        ms = (time.perf_counter() - t0) * 1000
        if ms > 200:                      # vehicle.yaml command_timeout is 0.4 s: keep perception well inside it
            self.get_logger().warn(f"frame took {ms:.0f} ms - too slow for the 0.4 s command_timeout budget",
                                   throttle_duration_sec=2.0)
        self.get_logger().info(
            f"{len(pts)} pts -> roi {seg['n_roi']} -> vox {seg['n_vox']} | {len(tracks)} boxes | "
            f"{ms:.0f} ms (pre {seg['ms_pre']:.0f}, ground {seg['ms_ground']:.0f}, "
            f"cluster {seg['ms_cluster']:.0f}) | {seg['n_clusters']} clusters, longest {seg['max_len']:.1f} m",
            throttle_duration_sec=2.0)

    def publish_boxes(self, header, boxes):
        arr = MarkerArray()
        # No DELETEALL: wiping + re-adding every frame is what makes RViz boxes blink.
        # Each box keeps its tracker id, and only ids that disappeared get an explicit DELETE.
        # Stamp 0 = "use the newest TF". The boxes are already pushed forward to 'now', and a scan-time
        # stamp makes RViz silently drop a marker whenever TF for that exact time isn't there yet (blinking).
        header = type(header)(frame_id=header.frame_id)
        ids = {int(b["id"]) for b in boxes}
        for old in self.prev_ids - ids:
            d = Marker()
            d.header = header
            d.ns, d.id, d.action = "boxes", old, Marker.DELETE
            arr.markers.append(d)
        self.prev_ids = ids
        life = float(self.a.box_lifetime)
        for b in boxes:
            m = Marker()
            m.header = header
            m.ns, m.id = "boxes", int(b["id"])
            m.type, m.action = Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = b["cx"], b["cy"], b["cz"]
            (m.pose.orientation.x, m.pose.orientation.y,
             m.pose.orientation.z, m.pose.orientation.w) = yaw_to_quat(b["yaw"])
            m.scale.x, m.scale.y, m.scale.z = b["l"], b["w"], b["h"]
            m.color.g, m.color.a = 1.0, 0.4
            m.lifetime.sec = int(life)           # auto-expire if the node stalls (must exceed frame time)
            m.lifetime.nanosec = int((life % 1) * 1e9)
            arr.markers.append(m)
        self.pub_boxes.publish(arr)


def main():
    rclpy.init()
    node = LiveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
