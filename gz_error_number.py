#!/usr/bin/env python3
import time
import math
import csv
import os
import cv2
import numpy as np

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_v_pb2 import Pose_V

# =========================================================
# TOPICS
# =========================================================
IMAGE_TOPIC = "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/imager/image"
POSE_TOPIC = "/world/default/pose/info"

# =========================================================
# ENTITY NAMES TRONG GAZEBO
# Chay len no se in pose names ra terminal.
# Sau do sua dung ten camera va ring cho chinh xac.
# =========================================================
CAMERA_ENTITY_NAME = "camera_link"
RING_ENTITY_NAME = "tz1"

# =========================================================
# CAMERA / RING PARAMETERS
# Ban can tune lai cho dung
# =========================================================
FX = 542.6
FY = 542.6
RING_DIAMETER_M = 1.80

# De None thi tu dong lay tam anh
CX0 = None
CY0 = None

# =========================================================
# LOCK TARGET
# =========================================================
LOCKED = False
LOCK_CENTER = None
LOCK_VISIBLE_SIZE = 0.0
LOCK_MISSED = 0

MAX_LOCK_MISSED = 35
LOCK_MIN_SIZE_RATIO = 0.80
LOCK_MAX_DIST = 130.0
MIN_MAIN_VISIBLE_SIZE = 35.0

# =========================================================
# RUNTIME
# =========================================================
last_frame_time = None
frame_count = 0

latest_camera_pose = None
latest_ring_pose = None
printed_pose_names = False

# =========================================================
# CSV LOG
# =========================================================
RAW_CSV = "ring_detection_xyz_log.csv"
EVAL_CSV = "ring_eval_with_gt.csv"
RAW_HEADER_WRITTEN = False
EVAL_HEADER_WRITTEN = False

# =========================================================
# METRICS
# =========================================================
error_stats = {
    "count": 0,
    "sum_abs_x": 0.0,
    "sum_abs_y": 0.0,
    "sum_abs_z": 0.0,
    "sum_sq_x": 0.0,
    "sum_sq_y": 0.0,
    "sum_sq_z": 0.0,
    "sum_sq_norm": 0.0,
}

# =========================================================
# LATEST RESULT
# =========================================================
LATEST_RESULT = {
    "ring_detected": False,
    "cx": 0.0,
    "cy": 0.0,
    "radius_px": 0.0,
    "bbox_w": 0.0,
    "bbox_h": 0.0,
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "x_gt": 0.0,
    "y_gt": 0.0,
    "z_gt": 0.0,
    "err_x": 0.0,
    "err_y": 0.0,
    "err_z": 0.0,
    "err_norm": 0.0,
    "ex_norm": 0.0,
    "ey_norm": 0.0,
    "radius_norm": 0.0,
    "guidance": "SEARCH",
}


# =========================================================
# CSV HELPERS
# =========================================================
def ensure_raw_csv_header():
    global RAW_HEADER_WRITTEN
    if RAW_HEADER_WRITTEN:
        return

    need_header = (not os.path.exists(RAW_CSV)) or os.path.getsize(RAW_CSV) == 0

    with open(RAW_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow([
                "frame",
                "timestamp",
                "cx",
                "cy",
                "radius_px",
                "bbox_w",
                "bbox_h",
                "x",
                "y",
                "z",
                "ex_norm",
                "ey_norm",
                "radius_norm",
                "guidance",
            ])
    RAW_HEADER_WRITTEN = True


def append_raw_csv_row(row):
    ensure_raw_csv_header()
    with open(RAW_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def ensure_eval_csv_header():
    global EVAL_HEADER_WRITTEN
    if EVAL_HEADER_WRITTEN:
        return

    need_header = (not os.path.exists(EVAL_CSV)) or os.path.getsize(EVAL_CSV) == 0

    with open(EVAL_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow([
                "frame", "timestamp",
                "x_est", "y_est", "z_est",
                "x_gt", "y_gt", "z_gt",
                "err_x", "err_y", "err_z", "err_norm"
            ])
    EVAL_HEADER_WRITTEN = True


def append_eval_csv_row(row):
    ensure_eval_csv_header()
    with open(EVAL_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# =========================================================
# QUATERNION -> ROTATION MATRIX (NO SCIPY)
# q format: [x, y, z, w]
# =========================================================
def quat_xyzw_to_rotmat(q):
    x, y, z, w = q

    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)

    s = 2.0 / n

    xx = x * x * s
    yy = y * y * s
    zz = z * z * s
    xy = x * y * s
    xz = x * z * s
    yz = y * z * s
    wx = w * x * s
    wy = w * y * s
    wz = w * z * s

    Rm = np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)]
    ], dtype=np.float64)

    return Rm


# =========================================================
# DETECTION
# =========================================================
def detect_ring_candidates(frame):
    if frame is None or frame.size == 0:
        return [], None

    h, w = frame.shape[:2]

    margin_x = int(w * 0.04)
    margin_y = int(h * 0.04)
    roi = frame[margin_y:h - margin_y, margin_x:w - margin_x]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
    edges = cv2.Canny(blur, 80, 160)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 700.0:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1e-6:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        if bw <= 0 or bh <= 0:
            continue

        aspect = bw / float(bh)
        if aspect < 0.30 or aspect > 3.50:
            continue

        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.05:
            continue

        (cx_roi, cy_roi), radius = cv2.minEnclosingCircle(contour)
        if radius < 12.0:
            continue

        contour_full = contour.copy()
        contour_full[:, 0, 0] += margin_x
        contour_full[:, 0, 1] += margin_y

        center_full = np.array([cx_roi + margin_x, cy_roi + margin_y], dtype=np.float32)

        if (
            center_full[0] - radius < 2
            or center_full[1] - radius < 2
            or center_full[0] + radius > w - 2
            or center_full[1] + radius > h - 2
        ):
            continue

        visible_size = float(max(bw, bh))
        bbox_area = float(bw * bh)

        candidates.append({
            "center": center_full,
            "radius": float(radius),
            "contour": contour_full,
            "area": float(area),
            "circularity": float(circularity),
            "bbox_w": float(bw),
            "bbox_h": float(bh),
            "bbox_area": bbox_area,
            "visible_size": visible_size,
        })

    return candidates, edges


def select_main_ring(candidates):
    if not candidates:
        return None

    filtered = [c for c in candidates if c["visible_size"] >= MIN_MAIN_VISIBLE_SIZE]
    if not filtered:
        filtered = candidates

    filtered.sort(
        key=lambda c: (c["visible_size"], c["bbox_area"], c["area"]),
        reverse=True,
    )
    return filtered[0]


def find_closest_to_locked(candidates, locked_center, locked_visible_size,
                           max_dist=130.0, min_size_ratio=0.80):
    if locked_center is None or not candidates:
        return None

    best = None
    best_score = -1e9

    for c in candidates:
        dist = np.linalg.norm(c["center"] - locked_center)
        if dist > max_dist:
            continue

        if c["visible_size"] < locked_visible_size * min_size_ratio:
            continue

        size_ratio = c["visible_size"] / max(locked_visible_size, 1e-6)
        size_score = -abs(1.0 - size_ratio) * 120.0
        dist_score = -dist

        score = dist_score + size_score + 0.05 * c["bbox_area"]

        if score > best_score:
            best = c
            best_score = score

    return best


# =========================================================
# IMAGE SPACE -> XYZ
# =========================================================
def compute_ring_error(target, image_shape):
    h, w = image_shape[:2]
    image_center = np.array([w * 0.5, h * 0.5], dtype=np.float32)

    center = target["center"]
    radius = target["radius"]

    ex = float(center[0] - image_center[0])
    ey = float(center[1] - image_center[1])

    ex_norm = ex / image_center[0]
    ey_norm = ey / image_center[1]
    radius_norm = float(radius / min(w, h))

    return ex_norm, ey_norm, radius_norm, image_center


def estimate_ring_pose_from_image(target, image_shape,
                                  fx, fy, ring_diameter_m,
                                  cx0=None, cy0=None):
    h, w = image_shape[:2]

    if cx0 is None:
        cx0 = w * 0.5
    if cy0 is None:
        cy0 = h * 0.5

    cx = float(target["center"][0])
    cy = float(target["center"][1])
    radius_px = float(target["radius"])

    diameter_px = max(2.0 * radius_px, 1e-6)

    z = float(fx * ring_diameter_m / diameter_px)
    x = float((cx - cx0) * z / fx)
    y = float((cy - cy0) * z / fy)

    return x, y, z


def compute_guidance(ex_norm, ey_norm, radius_norm):
    deadband_xy = 0.05
    near_gate = 0.20

    cmds = []

    if ex_norm > deadband_xy:
        cmds.append("MOVE_RIGHT")
    elif ex_norm < -deadband_xy:
        cmds.append("MOVE_LEFT")

    if ey_norm > deadband_xy:
        cmds.append("MOVE_DOWN")
    elif ey_norm < -deadband_xy:
        cmds.append("MOVE_UP")

    aligned = abs(ex_norm) < deadband_xy and abs(ey_norm) < deadband_xy

    if aligned:
        cmds.append("GO_FORWARD")

    if radius_norm > near_gate:
        cmds.append("NEAR_RING")

    if not cmds:
        cmds.append("HOLD")

    return " | ".join(cmds), aligned


# =========================================================
# GAZEBO GT
# =========================================================
def pose_msg_to_dict(pose_msg):
    return {
        "name": pose_msg.name,
        "position": np.array([
            pose_msg.position.x,
            pose_msg.position.y,
            pose_msg.position.z
        ], dtype=np.float64),
        "quat_xyzw": np.array([
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w
        ], dtype=np.float64)
    }


def on_pose_info(msg: Pose_V):
    global latest_camera_pose, latest_ring_pose, printed_pose_names

    try:
        if not printed_pose_names:
            print("===== POSE NAMES FROM GAZEBO =====")
            for pose in msg.pose:
                print(pose.name)
            printed_pose_names = True
            print("==================================")

        for pose in msg.pose:
            if pose.name == CAMERA_ENTITY_NAME:
                latest_camera_pose = pose_msg_to_dict(pose)
            elif pose.name == RING_ENTITY_NAME:
                latest_ring_pose = pose_msg_to_dict(pose)

    except Exception as e:
        print(f"[POSE ERROR] {e}")


def world_to_camera_relative(camera_pose, target_pose):
    p_cam_w = camera_pose["position"]
    p_tgt_w = target_pose["position"]

    q_cam_xyzw = camera_pose["quat_xyzw"]
    R_cam_w = quat_xyzw_to_rotmat(q_cam_xyzw)

    p_rel_w = p_tgt_w - p_cam_w
    p_rel_cam_link = R_cam_w.T @ p_rel_w

    return p_rel_cam_link


def camera_link_to_optical(p_link):
    # Mapping thuong gap:
    # x_opt = -y_link
    # y_opt = -z_link
    # z_opt =  x_link
    x_link, y_link, z_link = p_link
    x_opt = -y_link
    y_opt = -z_link
    z_opt = x_link
    return np.array([x_opt, y_opt, z_opt], dtype=np.float64)


def get_ring_ground_truth_xyz():
    if latest_camera_pose is None or latest_ring_pose is None:
        return None

    p_rel_cam_link = world_to_camera_relative(latest_camera_pose, latest_ring_pose)
    p_rel_cam_opt = camera_link_to_optical(p_rel_cam_link)

    x_gt = float(p_rel_cam_opt[0])
    y_gt = float(p_rel_cam_opt[1])
    z_gt = float(p_rel_cam_opt[2])

    return x_gt, y_gt, z_gt


# =========================================================
# METRICS
# =========================================================
def update_error_stats(err_x, err_y, err_z):
    error_stats["count"] += 1

    error_stats["sum_abs_x"] += abs(err_x)
    error_stats["sum_abs_y"] += abs(err_y)
    error_stats["sum_abs_z"] += abs(err_z)

    error_stats["sum_sq_x"] += err_x * err_x
    error_stats["sum_sq_y"] += err_y * err_y
    error_stats["sum_sq_z"] += err_z * err_z

    err_norm = math.sqrt(err_x * err_x + err_y * err_y + err_z * err_z)
    error_stats["sum_sq_norm"] += err_norm * err_norm

    n = error_stats["count"]

    mae_x = error_stats["sum_abs_x"] / n
    mae_y = error_stats["sum_abs_y"] / n
    mae_z = error_stats["sum_abs_z"] / n

    rmse_x = math.sqrt(error_stats["sum_sq_x"] / n)
    rmse_y = math.sqrt(error_stats["sum_sq_y"] / n)
    rmse_z = math.sqrt(error_stats["sum_sq_z"] / n)
    rmse_norm = math.sqrt(error_stats["sum_sq_norm"] / n)

    return mae_x, mae_y, mae_z, rmse_x, rmse_y, rmse_z, rmse_norm


# =========================================================
# DRAW
# =========================================================
def draw_debug(frame, target, ex_norm, ey_norm, radius_norm,
               x, y, z, gt, guidance, fps_text, locked):
    h, w = frame.shape[:2]
    image_center = (int(w * 0.5), int(h * 0.5))

    center = target["center"]
    radius = target["radius"]
    contour = target["contour"]

    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
    cv2.circle(frame, (int(center[0]), int(center[1])), 4, (0, 0, 255), -1)
    cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (255, 255, 0), 2)
    cv2.circle(frame, image_center, 4, (255, 0, 0), -1)
    cv2.line(frame, image_center, (int(center[0]), int(center[1])), (0, 255, 255), 2)

    y0 = 35
    dy = 35

    cv2.putText(frame, "RING DETECTED", (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"ex={ex_norm:.3f} ey={ey_norm:.3f} r={radius_norm:.3f}",
                (20, y0 + dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"est xyz=({x:.3f}, {y:.3f}, {z:.3f})",
                (20, y0 + 2 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    if gt is not None:
        x_gt, y_gt, z_gt = gt
        err_x = x - x_gt
        err_y = y - y_gt
        err_z = z - z_gt
        err_norm = math.sqrt(err_x**2 + err_y**2 + err_z**2)

        cv2.putText(frame, f"gt  xyz=({x_gt:.3f}, {y_gt:.3f}, {z_gt:.3f})",
                    (20, y0 + 3 * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (200, 255, 200), 2, cv2.LINE_AA)

        cv2.putText(frame, f"err=({err_x:.3f}, {err_y:.3f}, {err_z:.3f}) |e|={err_norm:.3f}",
                    (20, y0 + 4 * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 165, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "gt xyz=(N/A)",
                    (20, y0 + 3 * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, guidance,
                (20, y0 + 5 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"LOCKED={locked}",
                (20, y0 + 6 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"visible_size={target['visible_size']:.1f}",
                (20, y0 + 7 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"bbox_w={target['bbox_w']:.1f} bbox_h={target['bbox_h']:.1f}",
                (20, y0 + 8 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"radius_px={target['radius']:.1f}",
                (20, y0 + 9 * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"FPS: {fps_text}",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)


# =========================================================
# IMAGE CALLBACK
# =========================================================
def on_image(msg: Image):
    global last_frame_time, frame_count
    global LOCKED, LOCK_CENTER, LOCK_VISIBLE_SIZE, LOCK_MISSED
    global LATEST_RESULT

    try:
        width = msg.width
        height = msg.height
        data = msg.data

        if width <= 0 or height <= 0:
            return

        arr = np.frombuffer(data, dtype=np.uint8)
        expected = width * height * 3
        if arr.size != expected:
            return

        rgb = arr.reshape((height, width, 3))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        frame_count += 1
        now = time.time()

        if last_frame_time is None:
            fps_text = "N/A"
        else:
            dt = now - last_frame_time
            fps_text = f"{1.0 / dt:.2f}" if dt > 0 else "inf"
        last_frame_time = now

        candidates, edges = detect_ring_candidates(frame)
        target = None

        if LOCKED:
            target = find_closest_to_locked(
                candidates,
                LOCK_CENTER,
                LOCK_VISIBLE_SIZE,
                max_dist=LOCK_MAX_DIST,
                min_size_ratio=LOCK_MIN_SIZE_RATIO
            )

            if target is not None:
                alpha = 0.7
                LOCK_CENTER = alpha * LOCK_CENTER + (1.0 - alpha) * target["center"]
                LOCK_VISIBLE_SIZE = alpha * LOCK_VISIBLE_SIZE + (1.0 - alpha) * target["visible_size"]
                LOCK_MISSED = 0

                target = dict(target)
                target["center"] = LOCK_CENTER.copy()
            else:
                LOCK_MISSED += 1
                if LOCK_MISSED > MAX_LOCK_MISSED:
                    LOCKED = False
                    LOCK_CENTER = None
                    LOCK_VISIBLE_SIZE = 0.0
                    LOCK_MISSED = 0

        if not LOCKED:
            target = select_main_ring(candidates)
            if target is not None:
                LOCKED = True
                LOCK_CENTER = target["center"].copy()
                LOCK_VISIBLE_SIZE = target["visible_size"]
                LOCK_MISSED = 0

        if target is not None:
            ex_norm, ey_norm, radius_norm, _ = compute_ring_error(target, frame.shape)
            guidance, _ = compute_guidance(ex_norm, ey_norm, radius_norm)

            cx = float(target["center"][0])
            cy = float(target["center"][1])
            radius_px = float(target["radius"])
            bbox_w = float(target["bbox_w"])
            bbox_h = float(target["bbox_h"])

            x, y, z = estimate_ring_pose_from_image(
                target=target,
                image_shape=frame.shape,
                fx=FX,
                fy=FY,
                ring_diameter_m=RING_DIAMETER_M,
                cx0=CX0,
                cy0=CY0,
            )

            gt = get_ring_ground_truth_xyz()

            x_gt = y_gt = z_gt = 0.0
            err_x = err_y = err_z = err_norm = 0.0

            if gt is not None:
                x_gt, y_gt, z_gt = gt

                err_x = x - x_gt
                err_y = y - y_gt
                err_z = z - z_gt
                err_norm = math.sqrt(err_x**2 + err_y**2 + err_z**2)

                mae_x, mae_y, mae_z, rmse_x, rmse_y, rmse_z, rmse_norm = update_error_stats(
                    err_x, err_y, err_z
                )

                print(
                    f"[GT {frame_count}] "
                    f"gt=({x_gt:+.3f}, {y_gt:+.3f}, {z_gt:+.3f}) "
                    f"err=({err_x:+.3f}, {err_y:+.3f}, {err_z:+.3f}) "
                    f"|e|={err_norm:.3f}"
                )

                print(
                    f"[METRIC] "
                    f"MAE=({mae_x:.3f}, {mae_y:.3f}, {mae_z:.3f}) "
                    f"RMSE=({rmse_x:.3f}, {rmse_y:.3f}, {rmse_z:.3f}) "
                    f"RMSE_norm={rmse_norm:.3f}"
                )

                append_eval_csv_row([
                    frame_count,
                    f"{now:.6f}",
                    f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                    f"{x_gt:.6f}", f"{y_gt:.6f}", f"{z_gt:.6f}",
                    f"{err_x:.6f}", f"{err_y:.6f}", f"{err_z:.6f}", f"{err_norm:.6f}",
                ])
            else:
                print("[GT] chua co du lieu pose camera/ring")

            draw_debug(
                frame=frame,
                target=target,
                ex_norm=ex_norm,
                ey_norm=ey_norm,
                radius_norm=radius_norm,
                x=x, y=y, z=z,
                gt=gt,
                guidance=guidance,
                fps_text=fps_text,
                locked=LOCKED
            )

            LATEST_RESULT = {
                "ring_detected": True,
                "cx": cx,
                "cy": cy,
                "radius_px": radius_px,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "x": x,
                "y": y,
                "z": z,
                "x_gt": x_gt,
                "y_gt": y_gt,
                "z_gt": z_gt,
                "err_x": err_x,
                "err_y": err_y,
                "err_z": err_z,
                "err_norm": err_norm,
                "ex_norm": ex_norm,
                "ey_norm": ey_norm,
                "radius_norm": radius_norm,
                "guidance": guidance,
            }

            print(
                f"[RING_RAW {frame_count}] "
                f"cx={cx:.2f} cy={cy:.2f} r={radius_px:.2f} "
                f"bw={bbox_w:.2f} bh={bbox_h:.2f}"
            )

            print(
                f"[RING_XYZ {frame_count}] "
                f"x={x:+.3f} y={y:+.3f} z={z:.3f} "
                f"ex={ex_norm:+.3f} ey={ey_norm:+.3f} rn={radius_norm:.3f} "
                f"lock={LOCKED} missed={LOCK_MISSED} -> {guidance}"
            )

            append_raw_csv_row([
                frame_count,
                f"{now:.6f}",
                f"{cx:.6f}",
                f"{cy:.6f}",
                f"{radius_px:.6f}",
                f"{bbox_w:.6f}",
                f"{bbox_h:.6f}",
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                f"{ex_norm:.6f}",
                f"{ey_norm:.6f}",
                f"{radius_norm:.6f}",
                guidance,
            ])

        else:
            cv2.putText(frame, "RING LOST", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, f"FPS: {fps_text}", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

            if LOCKED:
                cv2.putText(frame,
                            f"KEEP LOCK ({LOCK_MISSED}/{MAX_LOCK_MISSED})",
                            (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

            LATEST_RESULT = {
                "ring_detected": False,
                "cx": 0.0,
                "cy": 0.0,
                "radius_px": 0.0,
                "bbox_w": 0.0,
                "bbox_h": 0.0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "x_gt": 0.0,
                "y_gt": 0.0,
                "z_gt": 0.0,
                "err_x": 0.0,
                "err_y": 0.0,
                "err_z": 0.0,
                "err_norm": 0.0,
                "ex_norm": 0.0,
                "ey_norm": 0.0,
                "radius_norm": 0.0,
                "guidance": "SEARCH",
            }

            print(f"[VISION {frame_count}] ring lost lock={LOCKED} missed={LOCK_MISSED}")

        cv2.imshow("Gazebo Ring Tracker", frame)
        if edges is not None:
            cv2.imshow("Ring Edges", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            raise KeyboardInterrupt

    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[ERROR] {e}")


# =========================================================
# MAIN
# =========================================================
def run_vision():
    ensure_raw_csv_header()
    ensure_eval_csv_header()

    node = Node()

    ok_img = node.subscribe(Image, IMAGE_TOPIC, on_image)
    ok_pose = node.subscribe(Pose_V, POSE_TOPIC, on_pose_info)

    print(f"Subscribe image topic: {IMAGE_TOPIC}")
    print(f"Subscribe image result: {ok_img}")
    print(f"Subscribe pose topic:  {POSE_TOPIC}")
    print(f"Subscribe pose result: {ok_pose}")
    print(f"Raw CSV:  {os.path.abspath(RAW_CSV)}")
    print(f"Eval CSV: {os.path.abspath(EVAL_CSV)}")
    print(f"FX={FX}, FY={FY}, RING_DIAMETER_M={RING_DIAMETER_M}, CX0={CX0}, CY0={CY0}")
    print(f"CAMERA_ENTITY_NAME={CAMERA_ENTITY_NAME}")
    print(f"RING_ENTITY_NAME={RING_ENTITY_NAME}")

    if not ok_img:
        print("[ERROR] Khong subscribe duoc image topic")
        return

    if not ok_pose:
        print("[WARN] Khong subscribe duoc pose topic, phan GT se khong hoat dong")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nThoat vision.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_vision()