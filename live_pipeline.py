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
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray

try:
    from numba import njit
except ImportError:
    njit = None

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
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    return ((r > a.min_range) & (r < a.max_range) &
            (xyz[:, 0] >= a.roi_x[0]) & (xyz[:, 0] <= a.roi_x[1]) &
            (xyz[:, 1] >= a.roi_y[0]) & (xyz[:, 1] <= a.roi_y[1]) &
            (xyz[:, 2] >= a.roi_z[0]) & (xyz[:, 2] <= a.roi_z[1]))


def segment(pts, a):
    """pts: (N,3+) lidar cloud (only x,y,z used) -> dict with filtered xyz, ground mask, cluster labels, boxes.
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


class LiveNode(Node):
    def __init__(self):
        super().__init__("lidar_pipeline_live")

        # every knob from the old argparse section becomes a ROS parameter
        defaults = dict(
            topic="/os_front/points",
            min_range=0.8, max_range=30.0,
            roi_x=[-10.0, 30.0], roi_y=[-15.0, 15.0], roi_z=[-3.0, 3.0],
            max_height=2.5,
            sensor_height=0.0,           # metres above ground; 0.0 = estimate from the cloud
            ray_div=1.0, slope_deg=10.0, init_slope_deg=3.0, min_h=0.15,
            voxel=0.1, eps=0.5, min_size=8, max_box=12.0,
        )
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        p = lambda k: self.get_parameter(k).value
        self.a = SimpleNamespace(**{k: p(k) for k in defaults if k != "topic"})
        if self.a.sensor_height <= 0:
            self.a.sensor_height = None  # segment() treats None as "estimate"

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

        self.publish_boxes(msg.header, seg["boxes"])
        obs = seg["point_label"] >= 0
        self.pub_obs.publish(point_cloud2.create_cloud_xyz32(
            msg.header, seg["xyz"][obs].astype(np.float32)))

        ms = (time.perf_counter() - t0) * 1000
        self.get_logger().info(f"{len(pts)} pts -> {len(seg['boxes'])} boxes in {ms:.0f} ms",
                               throttle_duration_sec=2.0)

    def publish_boxes(self, header, boxes):
        arr = MarkerArray()
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL          # wipe last frame's boxes
        arr.markers.append(clear)
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
            m.lifetime.nanosec = 200_000_000     # auto-expire if the node stalls
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
