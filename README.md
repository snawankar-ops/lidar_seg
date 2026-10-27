# live_pipeline.py — real-time LiDAR obstacle detection (ROS 2)

A single-file ROS 2 node for the Sedrica buggy. It subscribes to one Ouster
`PointCloud2` topic, finds obstacles in every scan, and publishes 3D boxes and the
obstacle points so you can see them in RViz or feed them to other nodes.

```
/os_front/points ──► crop to vehicle ROI ──► voxel filter ──► ray ground filter
                         (+ cut out the            (0.15 m)        (numba)
                          buggy's own body)
                                                                       │
RViz  ◄── /detections/boxes      ◄── box tracker ◄── min-area 3D boxes ◄── KD-tree clustering
      ◄── /detections/obstacles      (stable ids,     (long walls are       (on a coarse grid)
                                      velocity)        split, not dropped)
```

Frame convention: x forward, y left, z up (REP-103), in metres. Outputs carry the
frame and timestamp of the input cloud (boxes are stamped 0, see *Limitations and quirks*).

A detailed explanation of every stage is in `live_pipeline_report.pdf`
(source: `live_pipeline_report.tex`). The older `pipeline_report.pdf` describes an
earlier version and its defaults no longer match the code.

---

## 1. Requirements

| Needed | Notes |
|---|---|
| Ubuntu + ROS 2 | Written for Humble. Check yours with `echo $ROS_DISTRO` and use it wherever this README says `humble`. |
| Python packages | `numpy`, `scipy` (ROS usually brings them), **`numba`** |
| PyYAML | Only if you pass `-p vehicle_yaml:=...`; ROS 2 installs it |

**Install numba. It is not optional in practice.** Without it the node falls back to a
simpler, less precise ground filter and prints a warning at start-up:

```bash
pip3 install numba          # if numpy complains: pip3 install "numpy<2"
```

---

## 2. Quick start on the lab PC (replaying the recorded bag)

Use three terminals. Source ROS in each one: `source /opt/ros/humble/setup.bash`
(or your distro). The paths below are the ones used so far; change them if the bag or
script moved.

**Terminal 1 — play the bag** (the bag only contains the front Ouster):

```bash
ros2 bag play /home/sedrica/saraah/campus_2026-10-23_00-26/bag \
    --topics /os_front/points /tf_static /tf --loop --clock
```

`--clock` makes the bag publish simulated time, so everything else must run with
`use_sim_time:=true`. Include `/tf` and `/tf_static`, otherwise RViz has no frames and
the Fixed Frame dropdown is empty.

**Terminal 2 — RViz:**

```bash
ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true
```

**Terminal 3 — the node:**

```bash
cd ~/saraah
python3 live_pipeline.py --ros-args -p use_sim_time:=true
```

You should see `listening on /os_front/points`, a line with the sensor height and body
cut-out, and then one status line every 2 s.

### RViz setup (first time)

1. **Fixed Frame:** `os_front_sensor` (the cloud's own frame, always works). `base_link`
   also works while `/tf_static` is playing.
2. **Raw cloud:** Add → By topic → `/os_front/points` → PointCloud2. Expand it:
   - Topic → **Reliability Policy: Best Effort** (the Ouster stream is best-effort; a
     Reliable subscriber receives nothing and the colour dropdown stays empty)
   - Color Transformer: **AxisColor**, Axis **Z**; Size **0.03**; Style **Flat Squares**
3. **Detected points:** Add → By topic → `/detections/obstacles` → PointCloud2. The
   node publishes no intensity, so use Color Transformer **FlatColor** (white or
   magenta) and Size **0.06**. The default Reliable setting is correct for this one.
4. **Boxes:** Add → By topic → `/detections/boxes` → MarkerArray.
5. Untick the leftover Camera / Imu / Map displays from the default config.

Toggle the raw cloud on and off to compare. Detected points should sit on walls, cars
and people, not on a flat ring on the floor.

---

## 3. Running on the real sensor

1. Start the Ouster driver the way you normally do and confirm the cloud is flowing:
   ```bash
   ros2 topic hz /os_front/points        # expect ~10 Hz
   ```
2. Run the node **without** `use_sim_time` (there is no bag clock):
   ```bash
   python3 live_pipeline.py
   ```
3. Run RViz without `use_sim_time` too.

### Other sensors on the buggy

The node knows the mounts of all three Ousters (built in from `vehicle.yaml`). It picks
the sensor from the topic name:

```bash
python3 live_pipeline.py --ros-args -p topic:=/os_top/points
python3 live_pipeline.py --ros-args -p topic:=/os_back/points
```

If your topic names do not contain `os_front`, `os_top` or `os_back`, add
`-p sensor:=os_front` (or the right one). Without a known sensor the node warns and
falls back to the raw sensor frame, which gives wrong ROI and body cut-out.

**One node handles one sensor.** The output topics are fixed, so for several sensors
remap them and rename each node:

```bash
python3 live_pipeline.py --ros-args -r __node:=det_back -p topic:=/os_back/points \
    -r /detections/boxes:=/detections/back/boxes \
    -r /detections/obstacles:=/detections/back/obstacles
```

### Using a different `vehicle.yaml`

The geometry is built in. To use the file on the machine instead:
`-p vehicle_yaml:=/path/to/vehicle.yaml`.

### On a Jetson

- JetPack 6 (Ubuntu 22.04): ROS 2 Humble installs natively. JetPack 5 (Ubuntu 20.04):
  use a Humble Docker container.
- `pip3 install numpy scipy numba`
- `sudo nvpmodel -m 0 && sudo jetson_clocks` for full clocks
- The first start is slow while numba compiles; later starts use the cache.
- The timings below were measured on the lab PC, **not on a Jetson**. Read the status
  line there. If a frame takes more than about 100 ms, raise `voxel` to 0.2 or shrink
  the ROI.

---

## 4. Reading the output

The status line, every 2 s:

```
24940 pts -> roi 14099 -> vox 4812 | 12 boxes | 14 ms (pre 3, ground 1, cluster 4) | 4 clusters, longest 12.9 m
```

(example values) Raw points, points left after the ROI crop, points left after voxel
downsampling, boxes published, total time per frame with the three main stages, and the
longest cluster found. A LiDAR at 10 Hz leaves 100 ms per frame. On the lab PC the
frame time has been 8–19 ms for roughly 23,000 points per frame.

A warning appears if a frame takes over 200 ms, because `vehicle.yaml` sets a 0.4 s
command timeout and perception must stay well inside it.

Published topics:

| Topic | Type | Content |
|---|---|---|
| `/detections/boxes` | `MarkerArray` | One green semi-transparent cube per tracked box. Marker id = track id, so ids stay stable between frames. |
| `/detections/obstacles` | `PointCloud2` | The downsampled points that belong to a kept cluster (x, y, z only). |

---

## 5. Parameters

Set with `--ros-args -p name:=value`. Lists use `-p roi_x:="[-6.0, 25.0]"`.

**Input and geometry**

| Parameter | Default | Meaning |
|---|---|---|
| `topic` | `/os_front/points` | Input cloud |
| `sensor` | `auto` | `os_top`, `os_front` or `os_back`; `auto` reads it from the topic name |
| `vehicle_yaml` | `""` | Path to a `vehicle.yaml`; empty uses the built-in copy |
| `sensor_height` | `0.0` | Lidar height above ground in m. `0.0` = take it from the mount (front: 0.595 m) |

**Region of interest.** `roi_*` and `ego_*` are in `base_link` (rear-axle centre; x
forward, y left; the ground is at z = −0.23). The two range limits are measured from the
sensor.

| Parameter | Default | Meaning |
|---|---|---|
| `min_range`, `max_range` | `0.8`, `25.0` m | Horizontal range limits from the sensor |
| `roi_x`, `roi_y`, `roi_z` | `[-6,25]`, `[-12,12]`, `[-0.8,2.6]` m | Box around the vehicle that is kept |
| `ego_margin` | `0.15` m | Margin added to the footprint when cutting out the buggy's own body |
| `ego_z` | `[-0.55, 1.95]` m | Heights of the body cut-out |
| `max_height` | `2.2` m | Non-ground points higher than this above the ground are ignored (canopy, overhangs) |

**Ground filter and clustering**

| Parameter | Default | Meaning |
|---|---|---|
| `voxel` | `0.15` m | Voxel downsampling size; 0 turns it off |
| `ray_div` | `1.0`° | Azimuth width of one ground-filter ray |
| `slope_deg` | `10`° | Maximum ground slope (global and local test) |
| `init_slope_deg` | `3`° | Slope allowed for the first point of each ray |
| `min_h` | `0.15` m | Height tolerance floor; lower catches smaller obstacles but passes more floor |
| `eps` | `0.8` m | Linking distance between cluster points |
| `min_size` | `6` | Minimum cluster size, in downsampled points |
| `cluster_voxel` | `0.3` m | Clustering runs on one point per cell of this size (much faster); 0 = every point |
| `split_len` | `4.0` m | Clusters longer than this are boxed in equal pieces; 0 = off |
| `max_box` | `0.0` | **Unused.** Long clusters are split, never dropped |

**Tracking and output**

| Parameter | Default | Meaning |
|---|---|---|
| `track_gate` | `2.5` m | Max distance between a track's prediction and a detection to match them |
| `track_min_hits` | `1` | Detections needed before a track is shown |
| `track_max_miss` | `4` | Frames a track coasts without a detection before it is deleted |
| `track_alpha` | `0.8` | Position update weight; higher follows the detection more closely |
| `lead_extra` | `0.03` s | Added to the measured compute time when pushing boxes forward along their velocity |
| `box_lifetime` | `1.0` s | Marker lifetime; vanished boxes are also deleted explicitly |

---

## 6. Tuning guide

Tune in this order: geometry (`topic`, `sensor`, ROI), ground filter, clustering, and only
then speed.

| Symptom | Try |
|---|---|
| Floor shows up as obstacles | Confirm numba is installed; raise `min_h` or `slope_deg`; check that the mount is level (see Limitations) |
| Low obstacles or curbs treated as ground | Lower `slope_deg` and `min_h` |
| One object split into several boxes | Raise `eps`; lower `min_size` |
| Separate objects merged | Lower `eps` |
| Walls become many box pieces | Intended. Raise `split_len` for fewer, longer boxes |
| Too many tiny boxes | Raise `min_size`; raise `min_range` |
| A box stuck to the buggy | Raise `ego_margin` (e.g. `-p ego_margin:=0.3`) |
| Boxes appear ahead of a fast object | `-p lead_extra:=0.0` |
| Too slow / frames dropped | Raise `voxel`, raise `cluster_voxel`, shrink `roi_*` or `max_range` |

---

## 7. Troubleshooting

| Problem | Cause and fix |
|---|---|
| RViz Fixed Frame dropdown is empty | No TF. The bag command must include `/tf` and `/tf_static`, and RViz must run with `use_sim_time:=true`. As a fallback type `os_front_sensor` into the box. |
| Raw cloud display is empty and its Color Transformer is blank | QoS mismatch. Set that display's Reliability Policy to **Best Effort**, then untick and re-tick it. |
| Detections show but the raw cloud does not | Same cause: your node subscribes best-effort and works, RViz defaults to Reliable. |
| Detected points are all white | The cloud has no intensity. Use FlatColor or AxisColor. |
| No boxes at all | Node not running, wrong `topic`, or the ROI cuts the scene. Check `ros2 topic hz /detections/boxes`. |
| `numba NOT installed` warning | `pip3 install numba`. The fallback filter is a different, less precise method, so results change too, not only speed. |
| Boxes blink once per loop | Expected with `--loop`: stamps jump back in time, old tracks coast, new ids appear. |
| `inotify_add_watch ... No space left on device` | Harmless ibus message, unrelated to RViz. |

### Why you may still see lag in RViz

The code is not the bottleneck, so if the display looks late, the cause is elsewhere:

- **Compute is small.** 8–19 ms per frame against a 100 ms budget. Ground filtering takes 0–1 ms, so numba is active.
- **The scan itself is slow.** One sweep takes 100 ms, so something moving at 8 m/s travels about 0.8 m while it is being scanned. That blur is in the data.
- **QoS in RViz.** A display left at Reliable on a best-effort topic shows nothing or stale data, and that looks like lag.
- **Box lead.** Boxes are pushed forward by compute time plus `lead_extra` (about 45 ms). The raw points in a bag replay are not extrapolated, so a fast box can look ahead of its points. Try `-p lead_extra:=0.0`.
- **Bag looping.** Time jumps back at each loop and tracks restart.

To check: `ros2 topic hz /os_front/points` and `ros2 topic hz /detections/boxes`.
If both are about 10 Hz, the node is keeping up.

---

## 8. What changed since the first version

- Boxes no longer blink: they are tracked with stable ids, vanished boxes get an explicit delete, and no marker is wiped and redrawn each frame.
- Long walls are split into pieces of up to `split_len` instead of being thrown away by `max_box`.
- The ROI is now in the vehicle frame, so the rotated front sensor looks where the buggy looks, and the buggy's own body is cut out.
- A constant-velocity tracker predicts motion, so fast objects are matched instead of lost, and boxes are pushed forward by the processing delay.
- Clustering runs on a coarse grid, which is much cheaper on dense walls; the output cloud is built with numpy instead of `point_cloud2.create_cloud_xyz32`.

---

## 9. Limitations and quirks

- **Level mount assumed.** `vehicle.yaml` gives yaw only, so roll and pitch are taken as zero. A tilted mount breaks the ground filter; there is no tilt correction.
- **Cloud frame.** The code treats the cloud as being in the Ouster *lidar* frame (3.8 cm above the sensor frame). `/os_front/points` carries `frame_id: os_front_sensor`. The 3.8 cm offset is far below the 0.15 m height tolerance, so it has no visible effect, but confirm it if you need centimetre accuracy.
- **No motion compensation.** This node reads the raw cloud, so fast motion smears objects slightly.
- **Velocities are relative to the sensor.** They include the buggy's own motion, so a static wall moves in the tracker while the buggy drives.
- **Boxes cover the visible side** of an object only, and are class-agnostic.
- **Far ground returns.** A far return that follows a range gap of at least 0.2 m and lies inside the 10° cone is accepted as ground, so the lowest return on a distant obstacle can be absorbed. This comes from reading the filter, not from a measurement.
- **Clustering on the coarse grid is approximate:** it can merge slightly more than exact clustering. Set `cluster_voxel:=0` to compare.
- **Boxes carry stamp 0** (use the latest TF), so a box is never dropped because TF for its exact scan time is not there yet.

---

## 10. Files

| File | Purpose |
|---|---|
| `live_pipeline.py` | The node (everything in one file) |
| `README.md` | This file |
| `live_pipeline_report.tex` / `.pdf` | Detailed description of the code and its analysis |
