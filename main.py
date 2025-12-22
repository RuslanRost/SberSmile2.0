import random
import time
from pathlib import Path
import json
from collections import deque

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
    "smile_hold_seconds": 0.5,
    "idle_delay_seconds": 5.0,
    "trigger_cooldown_seconds": 5.0,
    "smile_score_threshold": 0.35,
    "smile_smooth_alpha": 0.1,
    "smile_smooth_window": 5,
    "trigger_sync_frames": [0, 150, 283, 452, 537, 637, 712, 845],
    "camera_view_scale": 0.5,
    "min_face_area_ratio": 0.02,
    "face_detection_confidence": 0.5,
    "face_tracking_confidence": 0.5,
    "face_area_reference": 0.05,
    "face_landmarker_model": "models/face_landmarker.task",
}


def load_config(path: Path):
    cfg = DEFAULT_CONFIG.copy()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            cfg.update({k: v for k, v in data.items() if k in cfg})
        except Exception as exc:
            print(f"Failed to read config {path}, using defaults. Error: {exc}")
    cfg["video_size"] = tuple(int(x) for x in cfg["video_size"])
    cfg["trigger_sync_frames"] = [int(x) for x in cfg["trigger_sync_frames"]]
    cfg["camera_index"] = int(cfg["camera_index"])
    cfg["smile_hold_seconds"] = float(cfg["smile_hold_seconds"])
    cfg["idle_delay_seconds"] = float(cfg["idle_delay_seconds"])
    cfg["trigger_cooldown_seconds"] = float(cfg["trigger_cooldown_seconds"])
    cfg["smile_score_threshold"] = float(cfg["smile_score_threshold"])
    cfg["smile_smooth_alpha"] = float(cfg["smile_smooth_alpha"])
    cfg["smile_smooth_window"] = int(cfg["smile_smooth_window"])
    cfg["camera_view_scale"] = float(cfg["camera_view_scale"])
    cfg["min_face_area_ratio"] = float(cfg["min_face_area_ratio"])
    cfg["face_detection_confidence"] = float(cfg["face_detection_confidence"])
    cfg["face_tracking_confidence"] = float(cfg["face_tracking_confidence"])
    cfg.setdefault("face_area_reference", DEFAULT_CONFIG["face_area_reference"])
    cfg["face_area_reference"] = float(cfg["face_area_reference"])
    cfg["face_landmarker_model"] = str(cfg["face_landmarker_model"])
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


def detect_smile(frame_bgr, face_mesh):
    """Detect face and mouth metrics using MediaPipe Face Landmarker."""
    h, w = frame_bgr.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bgr)
    results = face_mesh.detect(mp_image)
    if not results.face_landmarks:
        return False, 0.0, 0.0, [], [], None

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

    return True, smile_raw, face_area_ratio, [face_box], smile_boxes, mouth_metrics


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
        output_facial_transformation_matrixes=False,
    )
    face_mesh = mp_face_landmarker.FaceLandmarker.create_from_options(options)

    cam = cv2.VideoCapture(CONFIG["camera_index"])
    if not cam.isOpened():
        raise RuntimeError("Cannot open camera")

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
    smile_score_smooth = 0.0
    smile_window = deque(maxlen=CONFIG["smile_smooth_window"])
    face_area_smooth = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    hide_cursor()
    try:
        while True:
            ret, cam_frame = cam.read()
            if not ret:
                continue

            # Detect smile duration
            (
                face_present,
                smile_score_raw,
                face_area_ratio_raw,
                face_boxes,
                smile_boxes,
                mouth_metrics,
            ) = detect_smile(cam_frame, face_mesh)
            alpha = CONFIG["smile_smooth_alpha"]
            smile_window.append(smile_score_raw)
            window_avg = sum(smile_window) / len(smile_window)
            smile_score_smooth = (1 - alpha) * smile_score_smooth + alpha * window_avg
            face_area_smooth = (1 - alpha) * face_area_smooth + alpha * face_area_ratio_raw
            face_ok = face_area_smooth >= CONFIG["min_face_area_ratio"]
            # scale smile score down for large faces (closer to camera)
            area_ref = max(1e-6, CONFIG["face_area_reference"])
            scale_area = min(1.0, area_ref / max(face_area_smooth, 1e-6))
            smile_score = smile_score_smooth * scale_area
            smiling = smile_score >= CONFIG["smile_score_threshold"] and face_ok
            now = time.monotonic()
            if smiling:
                if smile_started_at is None:
                    smile_started_at = now
                elif (
                    now - smile_started_at >= CONFIG["smile_hold_seconds"]
                    and not playing_trigger
                    and now >= trigger_cooldown_until
                    and pending_trigger is None
                ):
                    available = list_trigger_videos()
                    if available:
                        chosen = random.choice(available)
                        pending_trigger = chosen
                        target_sync_frame = next_sync_frame(last_idle_idx)
                        pending_requires_loop = target_sync_frame < last_idle_idx
                        trigger_frames = play_video_once(chosen)
                        playing_trigger = False
                    smile_started_at = None
            else:
                smile_started_at = None

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

            # Debug overlay text
            debug_lines = []
            mode = "trigger" if playing_trigger else "idle"
            if pending_trigger and not playing_trigger:
                mode = f"pending->{target_sync_frame}"
            debug_lines.append(f"mode: {mode}")
            current_video = "idle"
            if playing_trigger and current_trigger_path:
                current_video = current_trigger_path.name
            elif pending_trigger:
                current_video = f"pending {pending_trigger.name}"
            debug_lines.append(f"video: {current_video}")
            debug_lines.append(f"idle_frame: {last_idle_idx}")
            debug_lines.append(f"sync_frames: {CONFIG['trigger_sync_frames']}")
            if target_sync_frame is not None:
                debug_lines.append(
                    f"target_frame: {target_sync_frame} (wait_loop={pending_requires_loop})"
                )
            debug_lines.append(f"face_present: {face_present}")
            debug_lines.append(f"faces: {len(face_boxes)} smiles: {len(smile_boxes)}")
            debug_lines.append(f"face_area_raw: {face_area_ratio_raw:.3f}")
            debug_lines.append(f"face_area: {face_area_smooth:.3f}")
            debug_lines.append(f"area_scale: {scale_area:.2f}")
            debug_lines.append(f"smile_raw: {smile_score_raw:.2f}")
            debug_lines.append(f"smile_avg: {window_avg:.2f}")
            debug_lines.append(f"smile_score: {smile_score:.2f}")
            if mouth_metrics:
                debug_lines.append(
                    f"mouth_deltas: L {mouth_metrics['delta_left_norm']:.2f} R {mouth_metrics['delta_right_norm']:.2f}"
                )
                debug_lines.append(
                    f"mouth_delta_px: L {mouth_metrics['delta_left']:.1f} R {mouth_metrics['delta_right']:.1f}"
                )
            if smile_started_at:
                debug_lines.append(f"smile_hold: {now - smile_started_at:.2f}s")
            debug_lines.append(f"cooldown_ready: {now >= trigger_cooldown_until}")
            if idle_delay_until:
                debug_lines.append(f"idle_delay_left: {max(0.0, idle_delay_until - now):.2f}s")

            y = 30
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
