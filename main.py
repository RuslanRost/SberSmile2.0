import time
from pathlib import Path
import json
import re

import cv2
import numpy as np
import ctypes
import threading

try:
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.models import load_model
except Exception:
    from keras.preprocessing.image import img_to_array
    from keras.models import load_model


DEFAULT_CONFIG = {
    "camera_index": 0,
    "camera_url": "",
    "camera_capture_resolution": [1280, 720],
    "camera_view_scale": 0.5,
    "camera_display_scale": 1.0,
    "video_size": [336, 672],
    "cnn_detect_scale": 0.5,
    "cnn_every_n": 2,
    "smile_hold_seconds": 0.5,
    "smile_on_debounce_seconds": 0.15,
    "smile_off_debounce_seconds": 0.2,
    "smile_prob_threshold": 0.85,
    "idle_smile_delay": 2.0,
    "trigger_sync_frames": [0, 150, 283, 395, 480, 649, 741, 817, 892, 1025, 1138, 1238],
    "min_face_area_ratio": 0.02,
    "smile_cnn_model": "model.h5",
    "smile_cascade_path": "haarcascade_frontalface_default.xml",
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
    cfg["camera_index"] = int(cfg["camera_index"])
    cfg["camera_url"] = str(cfg.get("camera_url", "") or "")
    cfg["camera_capture_resolution"] = tuple(int(x) for x in cfg["camera_capture_resolution"])
    cfg["camera_view_scale"] = float(cfg["camera_view_scale"])
    cfg["camera_display_scale"] = float(cfg["camera_display_scale"])
    cfg["video_size"] = tuple(int(x) for x in cfg["video_size"])
    cfg["cnn_detect_scale"] = float(cfg["cnn_detect_scale"])
    cfg["cnn_every_n"] = int(cfg["cnn_every_n"])
    cfg["smile_hold_seconds"] = float(cfg["smile_hold_seconds"])
    cfg["smile_on_debounce_seconds"] = float(cfg["smile_on_debounce_seconds"])
    cfg["smile_off_debounce_seconds"] = float(cfg["smile_off_debounce_seconds"])
    cfg["smile_prob_threshold"] = float(cfg["smile_prob_threshold"])
    cfg["idle_smile_delay"] = float(cfg["idle_smile_delay"])
    cfg["trigger_sync_frames"] = [int(x) for x in cfg["trigger_sync_frames"]]
    cfg["min_face_area_ratio"] = float(cfg["min_face_area_ratio"])
    cfg["smile_cnn_model"] = str(cfg["smile_cnn_model"])
    cfg["smile_cascade_path"] = str(cfg["smile_cascade_path"])
    return cfg


def setup_camera_props(cam, cfg):
    try:
        w, h = cfg["camera_capture_resolution"]
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    except Exception:
        pass


def get_screen_size():
    user32 = ctypes.windll.user32
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


class VideoLooper:
    def __init__(self, path: Path):
        self.path = path
        self.cap = None
        self.frame_idx = -1
        self.fps = 30.0
        self.loop_started_at = time.monotonic()
        self._open()

    def _open(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(self.path))
        self.frame_idx = -1
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0
        self.loop_started_at = time.monotonic()

    def frame(self):
        if not self.cap or not self.cap.isOpened():
            self._open()
        looped = False
        now = time.monotonic()
        expected_idx = int((now - self.loop_started_at) * self.fps)
        if self.frame_idx < 0:
            expected_idx = max(expected_idx, 0)
        max_skip = 60
        while self.frame_idx < expected_idx and max_skip > 0:
            ret, frame = self.cap.read()
            if not ret:
                self._open()
                looped = True
                return None, -1, looped
            self.frame_idx = 0 if self.frame_idx < 0 else self.frame_idx + 1
            max_skip -= 1
        ret, frame = self.cap.read()
        if not ret:
            self._open()
            looped = True
            return None, -1, looped
        self.frame_idx = 0 if self.frame_idx < 0 else self.frame_idx + 1
        return frame, self.frame_idx, looped

    def release(self):
        if self.cap:
            self.cap.release()


def find_idle_video() -> Path:
    idle_dir = Path("idle")
    if not idle_dir.exists():
        raise RuntimeError(f"Idle folder not found: {idle_dir}")
    for candidate in sorted(idle_dir.iterdir()):
        if candidate.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            return candidate
    raise RuntimeError(f"No idle video found in {idle_dir}")


def list_active_videos():
    active_dir = Path("active")
    if not active_dir.exists():
        return []
    return [p for p in sorted(active_dir.iterdir()) if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm")]


def play_video_once(path: Path):
    cap = cv2.VideoCapture(str(path))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def next_sync_frame(current_idx: int, frames: list[int]):
    for f in frames:
        if f >= current_idx:
            return f
    return frames[0] if frames else 0


def overlay_top_right(base: np.ndarray, overlay: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = base.shape[:2]
    ow, oh = size
    x1, y1 = max(w - ow, 0), 0
    x2, y2 = min(w, x1 + ow), min(h, oh)
    resized = cv2.resize(overlay, (x2 - x1, y2 - y1))
    # draw border outside the overlay area (2px), without covering content
    border = 2
    top = max(0, y1 - border)
    bottom = min(h, y2 + border)
    left = max(0, x1 - border)
    right = min(w, x2 + border)
    color = (0, 0, 255)
    if top < y1:
        base[top:y1, left:right] = color
    if y2 < bottom:
        base[y2:bottom, left:right] = color
    if left < x1:
        base[top:bottom, left:x1] = color
    if x2 < right:
        base[top:bottom, x2:right] = color
    base[y1:y2, x1:x2] = resized
    return base


def place_camera_bottom_left(base: np.ndarray, cam_frame: np.ndarray, scale: float):
    if cam_frame is None:
        return base, None
    h, w = base.shape[:2]
    ch, cw = cam_frame.shape[:2]
    target_h = max(1, int(h * scale))
    target_w = max(1, int(cw * (target_h / ch)))
    if target_w > w:
        target_w = w
        target_h = int(ch * (target_w / cw))
    resized = cv2.resize(cam_frame, (target_w, target_h))
    margin = 20
    x1 = margin
    y1 = max(0, h - target_h - margin)
    base[y1:y1 + target_h, x1:x1 + target_w] = resized
    return base, (x1, y1, target_w, target_h)


def resolve_resource(path_str: str):
    path = Path(path_str)
    if path.exists():
        return path
    fallback = Path(r"C:\Users\User\Documents\GitHub\Smile-Detector-using-Python-main") / path_str
    if fallback.exists():
        return fallback
    return path


def main():
    cfg = load_config(Path("config.json"))
    model_path = resolve_resource(cfg["smile_cnn_model"])
    cascade_path = resolve_resource(cfg["smile_cascade_path"])
    if not model_path.exists() or not cascade_path.exists():
        raise RuntimeError(
            f"Missing model or cascade. model={model_path} cascade={cascade_path}"
        )

    detector = cv2.CascadeClassifier(str(cascade_path))
    model = load_model(str(model_path))

    cam_source = cfg["camera_url"] if cfg["camera_url"] else cfg["camera_index"]
    camera = cv2.VideoCapture(cam_source)
    if not camera.isOpened():
        raise RuntimeError("Cannot open camera")
    setup_camera_props(camera, cfg)

    window_name = "Smile CNN Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_w, screen_h = get_screen_size()
    idle = VideoLooper(find_idle_video())
    active_videos = list_active_videos()
    active_frames = iter(())
    active_playing = False
    smile_started_at = None
    idle_start_at = time.monotonic()
    pending_active = False
    pending_target_frame = None

    frame_idx = 0
    last_smiling = False
    last_box = None
    last_area_ratio = None
    last_smile_prob = None
    debounced_smiling = False
    raw_smile_started_at = None
    raw_not_smile_started_at = None
    latest_frame = None
    latest_idx = -1
    processed_idx = -1
    lock = threading.Lock()
    stop_event = threading.Event()

    def cnn_worker():
        nonlocal last_smiling, last_box, last_area_ratio, last_smile_prob, processed_idx
        while not stop_event.is_set():
            with lock:
                frame = None if latest_frame is None else latest_frame.copy()
                idx = latest_idx
            if frame is None or idx == processed_idx:
                time.sleep(0.005)
                continue
            if idx % max(cfg["cnn_every_n"], 1) != 0:
                processed_idx = idx
                continue
            scale = cfg["cnn_detect_scale"]
            if scale <= 0 or scale > 1:
                scale = 1.0
            small = frame if scale == 1.0 else cv2.resize(
                frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            rects = detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )
            smiling = False
            box = None
            area_ratio = None
            smile_prob = None
            if len(rects) > 0:
                center = np.array([gray.shape[1] / 2.0, gray.shape[0] / 2.0])
                best = min(
                    rects,
                    key=lambda r: np.linalg.norm(
                        np.array([r[0] + r[2] / 2.0, r[1] + r[3] / 2.0]) - center
                    ),
                )
                fX, fY, fW, fH = best
                orig_h, orig_w = frame.shape[:2]
                face_w = fW / scale if scale != 1.0 else fW
                face_h = fH / scale if scale != 1.0 else fH
                area_ratio = (face_w * face_h) / max(1.0, (orig_w * orig_h))
                if cfg["min_face_area_ratio"] <= 0 or area_ratio >= cfg["min_face_area_ratio"]:
                    roi = gray[fY:fY + fH, fX:fX + fW]
                    if roi.size != 0:
                        roi = cv2.resize(roi, (28, 28))
                        roi = roi.astype("float") / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)
                        (not_smiling, smiling_prob) = model.predict(roi, verbose=0)[0]
                        smiling = smiling_prob > not_smiling
                        smile_prob = float(smiling_prob)
                        if scale != 1.0:
                            fX = int(fX / scale)
                            fY = int(fY / scale)
                            fW = int(fW / scale)
                            fH = int(fH / scale)
                        box = (fX, fY, fW, fH, smiling)
            with lock:
                last_smiling = smiling
                last_box = box
                last_area_ratio = area_ratio
                last_smile_prob = smile_prob
                processed_idx = idx

    worker = threading.Thread(target=cnn_worker, daemon=True)
    worker.start()
    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            continue

        frame_clone = frame.copy()
        with lock:
            latest_frame = frame
            latest_idx = frame_idx
            box = last_box
            area_ratio = last_area_ratio
            smile_prob = last_smile_prob
            last_smiling_snapshot = last_smiling
        if smile_prob is not None:
            raw_smiling = smile_prob >= cfg["smile_prob_threshold"]
        else:
            raw_smiling = last_smiling_snapshot
        now = time.monotonic()
        if raw_smiling:
            raw_smile_started_at = raw_smile_started_at or now
            raw_not_smile_started_at = None
            if not debounced_smiling and (now - raw_smile_started_at) >= cfg["smile_on_debounce_seconds"]:
                debounced_smiling = True
        else:
            raw_not_smile_started_at = raw_not_smile_started_at or now
            raw_smile_started_at = None
            if debounced_smiling and (now - raw_not_smile_started_at) >= cfg["smile_off_debounce_seconds"]:
                debounced_smiling = False

        if box is not None:
            fX, fY, fW, fH, _ = box
            color = (0, 255, 0) if debounced_smiling else (0, 0, 255)
            label = "Smiling" if debounced_smiling else "Not Smiling"
            cv2.putText(frame_clone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fH + fY), color, 2)

        indicator_color = (0, 255, 0) if debounced_smiling else (0, 0, 255)
        if cfg["camera_display_scale"] < 1.0:
            ds = max(0.1, min(1.0, cfg["camera_display_scale"]))
            frame_clone = cv2.resize(
                frame_clone, (int(frame_clone.shape[1] * ds), int(frame_clone.shape[0] * ds))
            )
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        canvas, cam_rect = place_camera_bottom_left(canvas, frame_clone, cfg["camera_view_scale"])
        if active_playing:
            active_frame = next(active_frames, None)
            if active_frame is None:
                active_playing = False
                idle_start_at = time.monotonic()
            else:
                canvas = overlay_top_right(canvas, active_frame, cfg["video_size"])
        if not active_playing:
            idle_frame, _, _ = idle.frame()
            if idle_frame is not None:
                canvas = overlay_top_right(canvas, idle_frame, cfg["video_size"])
        if cam_rect and (area_ratio is not None or smile_prob is not None):
            x1, y1, _, _ = cam_rect
            text_y = max(20, y1 - 10)
            if area_ratio is not None:
                threshold = cfg["min_face_area_ratio"]
                text = f"face_area={area_ratio * 100:.2f}% (min {threshold * 100:.2f}%)"
                cv2.putText(
                    canvas,
                    text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )
                text_y = max(20, text_y - 18)
            if smile_prob is not None:
                prob_text = f"smile_prob={smile_prob * 100:.1f}%"
                cv2.putText(
                    canvas,
                    prob_text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )
        cv2.rectangle(canvas, (20, 20), (80, 80), indicator_color, -1)
        cv2.imshow(window_name, canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1
        if not active_playing and active_videos and (now - idle_start_at) >= cfg["idle_smile_delay"]:
            if debounced_smiling:
                if smile_started_at is None:
                    smile_started_at = now
                elif now - smile_started_at >= cfg["smile_hold_seconds"]:
                    pending_active = True
                    smile_started_at = None
            else:
                smile_started_at = None
        else:
            smile_started_at = None

        if pending_active and not active_playing:
            target = next_sync_frame(idle.frame_idx, cfg["trigger_sync_frames"])
            if idle.frame_idx >= target:
                chosen = np.random.choice(active_videos)
                active_frames = play_video_once(chosen)
                active_playing = True
                pending_active = False

    camera.release()
    idle.release()
    cv2.destroyAllWindows()
    stop_event.set()
    worker.join(timeout=1.0)


if __name__ == "__main__":
    main()
