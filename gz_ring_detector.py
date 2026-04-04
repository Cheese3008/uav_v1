#!/usr/bin/env python3
import time
import math
import csv
import os
import cv2
import numpy as np

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

TOPIC = "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/imager/image"

last_frame_time = None
frame_count = 0

# ===== Lock target =====
LOCKED = False
LOCK_CENTER = None
LOCK_VISIBLE_SIZE = 0.0
LOCK_MISSED = 0

MAX_LOCK_MISSED = 35
LOCK_MIN_SIZE_RATIO = 0.80
LOCK_MAX_DIST = 130.0
MIN_MAIN_VISIBLE_SIZE = 35.0

# ===== Camera / Ring parameters =====
# BAN PHAI SUA CAC GIA TRI NAY CHO DUNG
FX = 320.0
FY = 320.0
RING_DIAMETER_M = 1.0   # duong kinh that cua vong (m)

# Neu muon co principal point co dinh thi dien vao
# Neu de None thi tu dong lay tam anh
CX0 = None
CY0 = None

# ===== CSV log =====
LOG_CSV = "ring_detection_xyz_log.csv"
CSV_HEADER_WRITTEN = False

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
    "ex_norm": 0.0,
    "ey_norm": 0.0,
    "radius_norm": 0.0,
    "guidance": "SEARCH",
}


def ensure_csv_header():
    global CSV_HEADER_WRITTEN
    if CSV_HEADER_WRITTEN:
        return

    file_exists = os.path.exists(LOG_CSV)
    need_header = True

    if file_exists and os.path.getsize(LOG_CSV) > 0:
        need_header = False

    with open(LOG_CSV, "a", newline="") as f:
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

    CSV_HEADER_WRITTEN = True


def append_csv_row(row):
    ensure_csv_header()
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def detect_ring_candidates(frame):
    """
    Tra ve danh sach candidate vong.
    Uu tien theo kich thuoc nhin thay tren anh.
    """
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


def select_main_ring(candidates, image_shape):
    """
    Luon uu tien vong to nhat ma camera nhin thay.
    """
    if not candidates:
        return None

    filtered = [c for c in candidates if c["visible_size"] >= MIN_MAIN_VISIBLE_SIZE]
    if not filtered:
        filtered = candidates

    filtered.sort(
        key=lambda c: (
            c["visible_size"],
            c["bbox_area"],
            c["area"],
        ),
        reverse=True,
    )
    return filtered[0]


def find_closest_to_locked(candidates, locked_center, locked_visible_size,
                           max_dist=130.0, min_size_ratio=0.80):
    """
    Khi da lock roi:
    - candidate phai gan target cu
    - visible_size khong duoc nho hon qua nhieu
    """
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
    """
    Uoc luong x, y, z tu tam anh va kich thuoc vong tren anh.

    Gia su:
    - Camera pinhole
    - Biet duong kinh that cua vong
    - diameter_px = 2 * radius_px

    z = fx * D_real / D_pixel
    x = (cx - cx0) * z / fx
    y = (cy - cy0) * z / fy
    """
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


def draw_debug(frame, target, ex_norm, ey_norm, radius_norm,
               x, y, z, guidance, fps_text, locked):
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

    cv2.putText(frame, "RING DETECTED", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame,
                f"ex={ex_norm:.3f} ey={ey_norm:.3f} r={radius_norm:.3f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame,
                f"x={x:.3f} y={y:.3f} z={z:.3f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame,
                guidance,
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame,
                f"LOCKED={locked}",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame,
                f"visible_size={target['visible_size']:.1f}",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame,
                f"bbox_w={target['bbox_w']:.1f} bbox_h={target['bbox_h']:.1f}",
                (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame,
                f"radius_px={target['radius']:.1f}",
                (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame,
                f"FPS: {fps_text}",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)


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
            target = select_main_ring(candidates, frame.shape)
            if target is not None:
                LOCKED = True
                LOCK_CENTER = target["center"].copy()
                LOCK_VISIBLE_SIZE = target["visible_size"]
                LOCK_MISSED = 0

        if target is not None:
            ex_norm, ey_norm, radius_norm, _ = compute_ring_error(target, frame.shape)
            guidance, aligned = compute_guidance(ex_norm, ey_norm, radius_norm)

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

            draw_debug(
                frame, target,
                ex_norm, ey_norm, radius_norm,
                x, y, z,
                guidance, fps_text, LOCKED
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

            append_csv_row([
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


def run_vision():
    ensure_csv_header()

    node = Node()
    ok = node.subscribe(Image, TOPIC, on_image)
    print(f"Subscribe topic: {TOPIC}")
    print(f"Subscribe result: {ok}")
    print(f"Log CSV: {os.path.abspath(LOG_CSV)}")
    print(f"FX={FX}, FY={FY}, RING_DIAMETER_M={RING_DIAMETER_M}, CX0={CX0}, CY0={CY0}")

    if not ok:
        print("[ERROR] Khong subscribe duoc topic")
        return

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nThoat vision.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_vision()