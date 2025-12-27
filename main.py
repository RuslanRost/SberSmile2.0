import random
import time
from pathlib import Path
import json
import math
import re
import cv2
import numpy as np
import ctypes
import mediapipe as mp
from mediapipe.tasks.python.vision import face_landmarker as mp_face_landmarker
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.core import mediapipe_c_bindings as mp_cb

# Settings
WINDOW_NAME = "SberSmile"

IDLE_DIR = Path("idle")
TRIGGER_DIR = Path("active")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
DEFAULT_CONFIG = {
    "video_size": [336, 672],
    "camera_index": 0,
    "camera_url": "",
    "idle_delay_seconds": 5.0,
    "trigger_cooldown_seconds": 5.0,
    "smile_smooth_alpha": 0.1,
    "smile_drop_factor": 0.85,
    "trigger_sync_frames": [0, 150, 283, 452, 537, 637, 712, 845],
    "camera_view_scale": 0.5,
    "camera_capture_resolution": [1920, 1080],
    "camera_autofocus": True,
    "camera_auto_exposure": True,
    "min_face_area_ratio": 0.02,
    "face_detection_confidence": 0.5,
    "face_tracking_confidence": 0.5,
    "face_landmarker_model": "models/face_landmarker.task",
    "baseline_recalc_factor": 1.15,
}


def _json_load_with_comments(text: str):
    result = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        result.append(ch)
        i += 1
    return json.loads("".join(result))


def load_config(path: Path):
    cfg = DEFAULT_CONFIG.copy()
    if path.exists():
        try:
            data_text = path.read_text(encoding="utf-8")
            data = _json_load_with_comments(data_text)
            cfg.update({k: v for k, v in data.items() if k in cfg})
        except Exception as exc:
            print(f"Failed to read config {path}, using defaults. Error: {exc}")
    cfg["video_size"] = tuple(int(x) for x in cfg["video_size"])
    cfg["trigger_sync_frames"] = [int(x) for x in cfg["trigger_sync_frames"]]
    cfg["camera_index"] = int(cfg["camera_index"])
    cfg["camera_url"] = str(cfg.get("camera_url", "") or "")
    cfg["idle_delay_seconds"] = float(cfg["idle_delay_seconds"])
    cfg["trigger_cooldown_seconds"] = float(cfg["trigger_cooldown_seconds"])
    cfg["smile_smooth_alpha"] = float(cfg["smile_smooth_alpha"])
    cfg["camera_view_scale"] = float(cfg["camera_view_scale"])
    cfg["camera_capture_resolution"] = tuple(int(x) for x in cfg["camera_capture_resolution"])
    cfg["camera_autofocus"] = bool(cfg["camera_autofocus"])
    cfg["camera_auto_exposure"] = bool(cfg["camera_auto_exposure"])
    cfg["min_face_area_ratio"] = float(cfg["min_face_area_ratio"])
    cfg["face_detection_confidence"] = float(cfg["face_detection_confidence"])
    cfg["face_tracking_confidence"] = float(cfg["face_tracking_confidence"])
    cfg["face_landmarker_model"] = str(cfg["face_landmarker_model"])
    cfg["smile_drop_factor"] = float(cfg["smile_drop_factor"])
    cfg["baseline_recalc_factor"] = float(cfg["baseline_recalc_factor"])
    return cfg


def patch_mediapipe_bindings():
    """Work around mediapipe Windows dll missing free() export."""
    def _patched_load_raw_library(signatures=()):
        from importlib import resources

        global _shared_lib  # not used, but keeps signature similar
        if mp_cb._shared_lib is None:
            lib_filename = "libmediapipe.dll"
            lib_path_context = resources.files("mediapipe.tasks.c")
            absolute_lib_path = str(lib_path_context / lib_filename)
            mp_cb._shared_lib = ctypes.CDLL(absolute_lib_path)

        for signature in signatures:
            c_func = getattr(mp_cb._shared_lib, signature.func_name)
            c_func.argtypes = signature.argtypes
            c_func.restype = signature.restype

        # Attach free; fallback to msvcrt.free if not exported.
        try:
            free_func = mp_cb._shared_lib.free
            free_func.argtypes = [ctypes.c_void_p]
            free_func.restype = None
        except AttributeError:
            mp_cb._shared_lib.free = ctypes.cdll.msvcrt.free
        return mp_cb._shared_lib

    def _patched_load_shared_library(signatures=()):
        raw_lib = _patched_load_raw_library(signatures)
        return mp_cb.serial_dispatcher.SerialDispatcher(raw_lib, signatures)

    mp_cb.load_raw_library = _patched_load_raw_library
    mp_cb.load_shared_library = _patched_load_shared_library


class VideoLooper:
    def __init__(self, path: Path):
        self.path = path
        self.cap = None
        self.frame_idx = -1
        self._open()

    def _open(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(self.path))
        self.frame_idx = -1

    def frame(self):
        if not self.cap or not self.cap.isOpened():
            self._open()
        looped = False
        ret, frame = self.cap.read()
        if not ret:
            self._open()
            looped = True
            ret, frame = self.cap.read()
        if not ret:
            return None, -1, looped
        if self.frame_idx < 0:
            self.frame_idx = 0
        else:
            self.frame_idx += 1
        return frame, self.frame_idx, looped

    def release(self):
        if self.cap:
            self.cap.release()


def overlay_frame(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Place overlay frame at top-right of base frame."""
    h, w = base.shape[:2]
    ow, oh = CONFIG["video_size"]
    x1, y1 = max(w - ow, 0), 0
    x2, y2 = min(w, x1 + ow), min(h, oh)
    resized = cv2.resize(overlay, (x2 - x1, y2 - y1))
    result = base.copy()
    result[y1:y2, x1:x2] = resized
    return result


def place_camera_view(base: np.ndarray, cam_frame: np.ndarray) -> np.ndarray:
    """Place camera view on the canvas, scaled to a fraction of height."""
    if cam_frame is None:
        return base
    h, w = base.shape[:2]
    ch, cw = cam_frame.shape[:2]
    target_h = max(1, int(h * CONFIG["camera_view_scale"]))
    target_w = max(1, int(cw * (target_h / ch)))
    if target_w > w:
        target_w = w
        target_h = int(ch * (target_w / cw))
    resized = cv2.resize(cam_frame, (target_w, target_h))
    margin = 20
    x1 = margin
    y1 = h - target_h - margin
    x2 = min(w, x1 + target_w)
    y2 = min(h, y1 + target_h)
    base[y1:y2, x1:x2] = resized[: y2 - y1, : x2 - x1]
    return base


FACE_MOUTH_LANDMARKS = {
    "left": 61,
    "right": 291,
    "top": 13,
    "bottom": 14,
}
# Landmark sets for eye horizontal bounds (MediaPipe FaceMesh indices)
EYE_LANDMARKS_LEFT = [33, 7, 163, 144, 145, 153, 154, 155, 133]
EYE_LANDMARKS_RIGHT = [263, 249, 390, 373, 374, 380, 381, 382, 362]


def capture_photo(frame_bgr: np.ndarray, label: str, folder: Path, ts: str | None = None) -> Path | None:
    """Save BGR frame to disk with label; returns saved path or None."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        ts = ts or time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}-{label}.jpg"
        path = folder / filename
        cv2.imwrite(str(path), frame_bgr)
        return path
    except Exception as exc:
        print(f"Failed to save photo {label}: {exc}")
        return None


def detect_smile(frame_bgr, face_mesh):
    """Detect face and mouth metrics using MediaPipe Face Landmarker."""
    h, w = frame_bgr.shape[:2]
    yaw_rad = None
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bgr)
    results = face_mesh.detect(mp_image)
    if not results.face_landmarks:
        return False, 0.0, 0.0, [], [], None, yaw_rad, {}

    if getattr(results, "facial_transformation_matrixes", None):
        m = results.facial_transformation_matrixes[0]
        # Rotation matrix entries
        r00, r01, r02 = m[0][0], m[0][1], m[0][2]
        r10, r11, r12 = m[1][0], m[1][1], m[1][2]
        r20, r21, r22 = m[2][0], m[2][1], m[2][2]
        # Yaw extraction from rotation matrix (camera coords)
        yaw_rad = math.atan2(-r20, math.sqrt(r00 * r00 + r10 * r10))

    landmarks = results.face_landmarks[0]
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    face_box = (min_x, min_y, max_x - min_x, max_y - min_y)

    face_area_ratio = 0.0
    if w * h > 0:
        face_area_ratio = max(0.0, (face_box[2] * face_box[3]) / float(w * h))

    mouth_pts = {}
    for key, idx in FACE_MOUTH_LANDMARKS.items():
        lm = landmarks[idx]
        mouth_pts[key] = (int(lm.x * w), int(lm.y * h))

    left = mouth_pts["left"]
    right = mouth_pts["right"]
    top = mouth_pts["top"]
    bottom = mouth_pts["bottom"]

    mouth_width = max(1.0, abs(right[0] - left[0]))
    mouth_height = max(1.0, abs(bottom[1] - top[1]))
    lift_left = bottom[1] - left[1]
    lift_right = bottom[1] - right[1]
    avg_lift = (lift_left + lift_right) / 2.0
    lift_norm = max(0.0, avg_lift / mouth_height)
    width_height_ratio = mouth_width / mouth_height
    smile_raw = min(1.0, lift_norm * 1.5 * width_height_ratio / 2.0)

    smile_boxes = []
    mouth_metrics = {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "delta_left": lift_left,
        "delta_right": lift_right,
        "delta_left_norm": lift_left / mouth_height,
        "delta_right_norm": lift_right / mouth_height,
        "width_height_ratio": width_height_ratio,
        "lift_norm": lift_norm,
        "center": (
            int((left[0] + right[0]) / 2),
            int((top[1] + bottom[1]) / 2),
        ),
    }

    def _eye_span(indices):
        pts = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in indices if idx < len(landmarks)]
        if not pts:
            return None, None, None, None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x = min(xs)
        max_x = max(xs)
        center_y = int(sum(ys) / len(ys))
        corner_left = min(pts, key=lambda p: p[0])
        corner_right = max(pts, key=lambda p: p[0])
        return min_x, max_x, corner_left, corner_right

    left_min, _left_max, left_corner_pt, _ = _eye_span(EYE_LANDMARKS_LEFT)
    _right_min, right_max, _, right_corner_pt = _eye_span(EYE_LANDMARKS_RIGHT)
    eye_bounds = {
        "left_min": left_min,
        "right_max": right_max,
        "left_corner": left_corner_pt,
        "right_corner": right_corner_pt,
    }

    return True, smile_raw, face_area_ratio, [face_box], smile_boxes, mouth_metrics, yaw_rad, eye_bounds


def play_video_once(path: Path):
    cap = cv2.VideoCapture(str(path))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def find_idle_video() -> Path:
    if not IDLE_DIR.exists():
        raise RuntimeError(f"Idle folder not found: {IDLE_DIR}")
    for candidate in sorted(IDLE_DIR.iterdir()):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    raise RuntimeError(f"No idle video found in {IDLE_DIR}")


def list_trigger_videos():
    if not TRIGGER_DIR.exists():
        return []
    return [
        p for p in sorted(TRIGGER_DIR.iterdir()) if p.suffix.lower() in VIDEO_EXTS
    ]


def get_screen_size():
    user32 = ctypes.windll.user32
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


def setup_camera_props(cam):
    """Attempt to set camera properties for focus/exposure/resolution."""
    try:
        w, h = CONFIG["camera_capture_resolution"]
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    except Exception:
        pass
    try:
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 1 if CONFIG["camera_autofocus"] else 0)
    except Exception:
        pass
    try:
        if CONFIG["camera_auto_exposure"]:
            # 0.75 works for many DirectShow drivers as "auto"
            cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        else:
            cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception:
        pass
    try:
        cam.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
    except Exception:
        pass


def hide_cursor():
    try:
        ctypes.windll.user32.ShowCursor(False)
    except Exception:
        pass


def show_cursor():
    try:
        ctypes.windll.user32.ShowCursor(True)
    except Exception:
        pass


def next_sync_frame(current_idx: int):
    for f in CONFIG["trigger_sync_frames"]:
        if f >= current_idx:
            return f
    return CONFIG["trigger_sync_frames"][0]


def main():
    global CONFIG
    CONFIG = load_config(Path("config.json"))
    patch_mediapipe_bindings()
    model_path = Path(CONFIG["face_landmarker_model"])
    if not model_path.exists():
        raise RuntimeError(
            f"Face landmarker model not found at {model_path}. "
            "Download face_landmarker.task from MediaPipe and update config."
        )
    options = mp_face_landmarker.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.IMAGE,
        min_face_detection_confidence=CONFIG["face_detection_confidence"],
        min_face_presence_confidence=CONFIG["face_tracking_confidence"],
        min_tracking_confidence=CONFIG["face_tracking_confidence"],
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    face_mesh = mp_face_landmarker.FaceLandmarker.create_from_options(options)

    cam_source = CONFIG["camera_url"] if CONFIG.get("camera_url") else CONFIG["camera_index"]
    cam = cv2.VideoCapture(cam_source)
    if not cam.isOpened():
        raise RuntimeError("Cannot open camera")
    print(f"Camera source: {cam_source}")
    print(f"Camera opened: {cam.isOpened()}")
    setup_camera_props(cam)

    idle_path = find_idle_video()
    idle = VideoLooper(idle_path)
    smile_started_at = None
    playing_trigger = False
    current_trigger_path = None
    trigger_frames = iter(())
    screen_w, screen_h = get_screen_size()
    idle_delay_until = None
    trigger_cooldown_until = 0.0
    pending_trigger = None
    target_sync_frame = None
    pending_requires_loop = False
    last_idle_idx = 0
    face_area_smooth = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    hide_cursor()
    try:
        BASELINE_DURATION = 2.0  # seconds to average mouth percent sum after face appears
        MOUTH_SUM_SMOOTH_ALPHA = 0.2
        BASELINE_RECALC_FACTOR = CONFIG["baseline_recalc_factor"]  # trigger recalibration if current sum exceeds baseline by this factor
        MOUTH_DROP_FACTOR = CONFIG["smile_drop_factor"]  # trigger smile when sum is below this fraction of baseline
        MOUTH_DROP_HOLD_SECONDS = 1.0
        FACE_LOSS_GRACE = 0.3  # seconds to tolerate face loss before resetting baseline
        BASELINE_RECALC_HOLD = 0.3  # seconds threshold exceed must persist to trigger recalibration
        baseline_start_at = None
        baseline_acc_sum = 0.0
        baseline_acc_count = 0
        baseline_sum_final = None
        mouth_pct_sum_smooth = None
        mouth_drop_started_at = None
        smile_indicator = False
        smile_progress_ms = None
        baseline_reset_until = None
        baseline_reset_text = None
        baseline_reset_started_at = None
        face_absent_since = None
        baseline_photo_frame = None
        baseline_locked = False
        photos_dir = Path("photos")
        while True:
            ret, cam_frame = cam.read()
            if not ret:
                continue
            mouth_pct_left = None
            mouth_pct_right = None
            smile_indicator = False
            smile_progress_ms = None

            # Detect smile duration
            (
                face_present,
                _smile_score_raw,
                face_area_ratio_raw,
                face_boxes,
                smile_boxes,
                mouth_metrics,
                yaw_rad,
                eye_bounds,
            ) = detect_smile(cam_frame, face_mesh)
            now = time.monotonic()
            if not face_present:
                if face_absent_since is None:
                    face_absent_since = now
                elif now - face_absent_since >= FACE_LOSS_GRACE:
                    baseline_sum_final = None
                    baseline_start_at = None
                    baseline_acc_sum = 0.0
                    baseline_acc_count = 0
                    mouth_pct_sum_smooth = None
                    mouth_drop_started_at = None
                    smile_indicator = False
                    smile_progress_ms = None
                    baseline_reset_started_at = None
                    baseline_photo_frame = None
            alpha = CONFIG["smile_smooth_alpha"]
            face_area_smooth = (1 - alpha) * face_area_smooth + alpha * face_area_ratio_raw
            face_ok = face_area_smooth >= CONFIG["min_face_area_ratio"]
            # Prepare base black canvas matching screen
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cam_debug = cam_frame.copy()
            # draw face and smile boxes on camera view
            for (fx, fy, fw, fh) in face_boxes:
                cv2.rectangle(cam_debug, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
            for (sx, sy, sw, sh) in smile_boxes:
                cv2.rectangle(cam_debug, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            # draw mouth landmarks from metrics
            if mouth_metrics:
                for key in ["left", "right", "center", "bottom"]:
                    px, py = mouth_metrics[key]
                    color = (0, 255, 255) if key != "bottom" else (0, 165, 255)
                    cv2.circle(cam_debug, (int(px), int(py)), 4, color, -1)
                # draw distance from each mouth corner to eye horizontal span with percent
                if eye_bounds.get("left_min") is not None and eye_bounds.get("right_max") is not None:
                    left_edge = eye_bounds["left_min"]
                    right_edge = eye_bounds["right_max"]
                    yaw_cos = 1.0
                    if yaw_rad is not None:
                        yaw_cos = max(0.2, abs(math.cos(yaw_rad)))
                    width = max(1.0, right_edge - left_edge)
                    width_corrected = width / yaw_cos
                    sides = [
                        ("left", (0, 255, 0), left_edge),
                        ("right", (255, 0, 255), right_edge),
                    ]
                    for key, color, edge_x in sides:
                        px, py = mouth_metrics[key]
                        edge_pt = (edge_x, py)
                        if key == "left":
                            dist_px_signed = px - edge_x  # mouth moves right -> positive
                        else:
                            dist_px_signed = edge_x - px  # mouth moves left -> positive
                        dist_corrected = dist_px_signed / yaw_cos
                        percent = max(-100.0, min(100.0, (dist_corrected / max(width_corrected, 1.0)) * 100.0))
                        if key == "left":
                            mouth_pct_left = percent
                        elif key == "right":
                            mouth_pct_right = percent
                        cv2.line(cam_debug, (px, py), edge_pt, color, 2, cv2.LINE_AA)
                        label = f"{percent:.0f}%"
                        text_x = int((px + edge_pt[0]) / 2)
                        text_y = int((py + edge_pt[1]) / 2) - 8
                        cv2.putText(
                            cam_debug,
                            label,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                # vertical lines through eye and mouth corners for alignment visualization
                left_eye_corner = eye_bounds.get("left_corner")
                right_eye_corner = eye_bounds.get("right_corner")
                if left_eye_corner and mouth_metrics.get("left"):
                    lx, ly = left_eye_corner
                    mx, my = mouth_metrics["left"]
                    x_line = lx
                    y1, y2 = sorted([ly, my])
                    cv2.line(cam_debug, (x_line, y1), (x_line, y2), (0, 180, 0), 2, cv2.LINE_AA)
                if right_eye_corner and mouth_metrics.get("right"):
                    rx, ry = right_eye_corner
                    mx, my = mouth_metrics["right"]
                    x_line = rx
                    y1, y2 = sorted([ry, my])
                    cv2.line(cam_debug, (x_line, y1), (x_line, y2), (180, 0, 180), 2, cv2.LINE_AA)
            canvas = place_camera_view(canvas, cam_debug)

            # Pick frame to overlay
            overlay = None
            if playing_trigger:
                overlay = next(trigger_frames, None)
                if overlay is None:
                    playing_trigger = False
                    current_trigger_path = None
                    idle_delay_until = time.monotonic() + CONFIG["idle_delay_seconds"]
                    trigger_cooldown_until = time.monotonic() + CONFIG["trigger_cooldown_seconds"]
                    pending_trigger = None
                    target_sync_frame = None
                    pending_requires_loop = False
            if overlay is None:
                if idle_delay_until and time.monotonic() < idle_delay_until:
                    overlay = None  # stay black during delay
                else:
                    idle_delay_until = None
                overlay, last_idle_idx, looped = idle.frame()
                if pending_trigger and target_sync_frame is not None:
                    if looped and last_idle_idx <= target_sync_frame:
                        pending_requires_loop = False  # loop satisfied
                    if (
                        last_idle_idx >= target_sync_frame
                        and (not pending_requires_loop or looped)
                    ):
                        overlay = next(trigger_frames, None)
                        if overlay is not None:
                            playing_trigger = True
                            current_trigger_path = pending_trigger
                            pending_trigger = None
                            target_sync_frame = None
                            pending_requires_loop = False
                        else:
                            pending_trigger = None
                            target_sync_frame = None
                            pending_requires_loop = False
                if overlay is None and not playing_trigger:
                    pending_trigger = None
                    target_sync_frame = None
                    pending_requires_loop = False

            if overlay is not None:
                canvas = overlay_frame(canvas, overlay)
            # unlock baseline collection after trigger playback finishes
            if baseline_locked and not playing_trigger and pending_trigger is None:
                baseline_locked = False
                baseline_sum_final = None
                baseline_start_at = None
                baseline_acc_sum = 0.0
                baseline_acc_count = 0
                baseline_photo_frame = None
                mouth_pct_sum_smooth = None

            # Debug overlay text
            debug_lines = []
            debug_lines.append(f"source: {cam_source}")
            debug_lines.append(f"idle_frame: {last_idle_idx}")
            debug_lines.append(f"next_sync: {next_sync_frame(last_idle_idx)}")
            # mode = "trigger" if playing_trigger else "idle"
            # if pending_trigger and not playing_trigger:
            #     mode = f"pending->{target_sync_frame}"
            # debug_lines.append(f"mode: {mode}")
            # current_video = "idle"
            # if playing_trigger and current_trigger_path:
            #     current_video = current_trigger_path.name
            # elif pending_trigger:
            #     current_video = f"pending {pending_trigger.name}"
            # debug_lines.append(f"video: {current_video}")
            # debug_lines.append(f"idle_frame: {last_idle_idx}")
            # debug_lines.append(f"sync_frames: {CONFIG['trigger_sync_frames']}")
            # if target_sync_frame is not None:
            #     debug_lines.append(
            #         f"target_frame: {target_sync_frame} (wait_loop={pending_requires_loop})"
            #     )
            # debug_lines.append(f"face_present: {face_present}")
            # debug_lines.append(f"faces: {len(face_boxes)} smiles: {len(smile_boxes)}")
            # debug_lines.append(f"face_area_raw: {face_area_ratio_raw:.3f}")
            # debug_lines.append(f"face_area: {face_area_smooth:.3f}")
            # debug_lines.append(f"area_scale: {scale_area:.2f}")
            # debug_lines.append(f"smile_raw: {smile_score_raw:.2f}")
            # debug_lines.append(f"smile_avg: {window_avg:.2f}")
            # debug_lines.append(f"smile_score: {smile_score:.2f}")
            # if mouth_metrics:
            #     debug_lines.append(
            #         f"mouth_deltas: L {mouth_metrics['delta_left_norm']:.2f} R {mouth_metrics['delta_right_norm']:.2f}"
            #     )
            #     debug_lines.append(
            #         f"mouth_delta_px: L {mouth_metrics['delta_left']:.1f} R {mouth_metrics['delta_right']:.1f}"
            #     )
            # if smile_started_at:
            #     debug_lines.append(f"smile_hold: {now - smile_started_at:.2f}s")
            # debug_lines.append(f"cooldown_ready: {now >= trigger_cooldown_until}")
            # if idle_delay_until:
            #     debug_lines.append(f"idle_delay_left: {max(0.0, idle_delay_until - now):.2f}s")
            if mouth_pct_left is not None and mouth_pct_right is not None:
                mouth_pct_sum_raw = mouth_pct_left + mouth_pct_right
                face_absent_since = None
                if not baseline_locked:
                    if mouth_pct_sum_smooth is None:
                        mouth_pct_sum_smooth = mouth_pct_sum_raw
                    else:
                        mouth_pct_sum_smooth = (
                            (1 - MOUTH_SUM_SMOOTH_ALPHA) * mouth_pct_sum_smooth
                            + MOUTH_SUM_SMOOTH_ALPHA * mouth_pct_sum_raw
                        )
                    debug_lines.append(f"mouth_pct_sum: {mouth_pct_sum_smooth:.0f}%")
                    if (
                        baseline_sum_final is not None
                        and mouth_pct_sum_raw > baseline_sum_final * BASELINE_RECALC_FACTOR
                    ):
                        baseline_reset_text = (
                            f"baseline_reset: raw {mouth_pct_sum_raw:.0f}% > baseline {baseline_sum_final:.0f}% * {BASELINE_RECALC_FACTOR}"
                        )
                        baseline_reset_started_at = baseline_reset_started_at or now
                        if now - baseline_reset_started_at >= BASELINE_RECALC_HOLD:
                            baseline_reset_until = now + 2.0
                            baseline_sum_final = None
                            baseline_start_at = now
                            baseline_acc_sum = mouth_pct_sum_raw
                            baseline_acc_count = 1
                            baseline_reset_started_at = None
                            baseline_photo_frame = None
                    else:
                        baseline_reset_started_at = None
                    # baseline accumulation for first seconds after face appears
                    if baseline_sum_final is None:
                        if baseline_start_at is None:
                            baseline_start_at = now
                        elapsed = now - baseline_start_at
                        baseline_acc_sum += mouth_pct_sum_raw
                        baseline_acc_count += 1
                        if elapsed >= BASELINE_DURATION and baseline_acc_count > 0:
                            baseline_sum_final = baseline_acc_sum / baseline_acc_count
                            baseline_photo_frame = cam_frame.copy()
                    if baseline_sum_final is not None:
                        debug_lines.append(f"baseline_sum: {baseline_sum_final:.0f}%")
                        # drop_factor intentionally hidden
                # smile indicator based on drop below baseline
                smile_trigger = False
                smile_indicator = False
                if baseline_sum_final is not None:
                    threshold = baseline_sum_final * MOUTH_DROP_FACTOR
                    if mouth_pct_sum_raw <= threshold:
                        if mouth_drop_started_at is None:
                            mouth_drop_started_at = now
                        elapsed = now - mouth_drop_started_at
                        smile_progress_ms = int(elapsed * 1000)
                        if elapsed >= MOUTH_DROP_HOLD_SECONDS:
                            smile_trigger = True
                            smile_indicator = True
                    else:
                        mouth_drop_started_at = None
                        smile_progress_ms = None
                else:
                    mouth_drop_started_at = None
                    smile_progress_ms = None

                if (
                    smile_trigger
                    and not playing_trigger
                    and now >= trigger_cooldown_until
                    and pending_trigger is None
                ):
                    ts_send = time.strftime("%Y%m%d_%H%M%S")
                    smile_photo_path = capture_photo(cam_frame, "smile", photos_dir, ts=ts_send)
                    serious_photo_path = None
                    if baseline_photo_frame is not None:
                        serious_photo_path = capture_photo(baseline_photo_frame, "serious", photos_dir, ts=ts_send)
                    baseline_locked = True
                    baseline_sum_final = None
                    baseline_start_at = None
                    baseline_acc_sum = 0.0
                    baseline_acc_count = 0
                    baseline_photo_frame = None
                    available = list_trigger_videos()
                    if available:
                        chosen = random.choice(available)
                        pending_trigger = chosen
                        target_sync_frame = next_sync_frame(last_idle_idx)
                        pending_requires_loop = target_sync_frame < last_idle_idx
                        trigger_frames = play_video_once(chosen)
                        playing_trigger = False
                    mouth_drop_started_at = None
            else:
                mouth_pct_sum_smooth = None
                mouth_drop_started_at = None
                smile_indicator = False
                smile_progress_ms = None
                if face_present:
                    face_absent_since = None
                baseline_reset_started_at = None
            # show baseline reset reason for a short time
            if baseline_reset_until is not None and now < baseline_reset_until and baseline_reset_text:
                debug_lines.append(baseline_reset_text)
            elif baseline_reset_until is not None and now >= baseline_reset_until:
                baseline_reset_until = None
                baseline_reset_text = None
                baseline_reset_started_at = None

            indicator_color = (0, 255, 0) if smile_indicator else (0, 0, 255)
            cv2.putText(canvas, "SMILE READY", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, indicator_color, 2, cv2.LINE_AA)
            progress_color = (0, 200, 255) if smile_progress_ms is not None else (80, 80, 80)
            progress_text = "SMILE PROGRESS"
            if smile_progress_ms is not None:
                progress_text += f": {smile_progress_ms} ms"
            cv2.putText(canvas, progress_text, (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, progress_color, 2, cv2.LINE_AA)
            # mouth visibility/presence debug removed (not reliable for face landmarks)

            y = 100
            for line in debug_lines:
                cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                y += 24

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                break
    finally:
        show_cursor()
        idle.release()
        cam.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
