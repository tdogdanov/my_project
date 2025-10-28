# main.py
import argparse
import cv2
import os
import time
from datetime import datetime
from queue import Queue
import threading
import torch
from ultralytics import YOLO
import psutil
from utils.detect_face import detect_objects_for_face
from utils.ipcams import open_camera, list_available_sources

# ----------------------------
DEFAULT_CUSTOM_MODEL = "models/yolov8_custom.pt"
DEFAULT_PRETRAINED_MODEL = "models/yolov9s.pt"
OUTPUT_ROOT = "outputs"
CONFIDENCE_THRESHOLD = 0.8
DETECTION_SIZE = (640, 384)
MIN_DETECTION_SIZE = (320, 192)
MAX_DETECTION_SIZE = (640, 384)

os.makedirs(OUTPUT_ROOT, exist_ok=True)
_model = None
stop_event = threading.Event()

# ----------------------------
def log_message(msg, level="ИНФО"):
    print(f"[{level}] {msg}")

def log_system_load():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    log_message(f"Загрузка CPU: {cpu:.1f}% | RAM: {mem:.1f}%")

# ----------------------------
def get_model(model_path=None):
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = model_path or (
            DEFAULT_CUSTOM_MODEL if os.path.isfile(DEFAULT_CUSTOM_MODEL) else DEFAULT_PRETRAINED_MODEL
        )
        log_message(f"Загружаем модель: {model_path} на {device}")
        _model = YOLO(model_path)
        _model.to(device)
    return _model

# ----------------------------
def detect_objects(frame, model, conf_threshold=CONFIDENCE_THRESHOLD):
    """Обычное детектирование (если не требуется только лицо)."""
    input_frame = cv2.resize(frame, DETECTION_SIZE)
    results = model(input_frame, conf=conf_threshold)
    annotated_frame = frame.copy()

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        scale_x = frame.shape[1] / DETECTION_SIZE[0]
        scale_y = frame.shape[0] / DETECTION_SIZE[1]
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_frame,
            f"{cls_name} {conf:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    return annotated_frame, results

# ----------------------------
def auto_filename(extension="mp4"):
    timestamp = datetime.now().strftime("%H-%M-%S")
    today = datetime.now().strftime("%Y-%m-%d")
    dir_path = os.path.join(OUTPUT_ROOT, today)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, f"det_{timestamp}.{extension}")

# ----------------------------
def video_worker(capture, frame_queue):
    """Захват кадров из видеопотока."""
    while not stop_event.is_set():
        ret, frame = capture.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
    capture.release()

# ----------------------------
def infer_worker(frame_queue, result_queue, model, conf_threshold, face_mode=False):
    """Фоновый поток для инференса."""
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            with torch.no_grad():
                if face_mode:
                    annotated, results = detect_objects_for_face(frame, model, conf_threshold)
                else:
                    annotated, results = detect_objects(frame, model, conf_threshold)
            result_queue.put((annotated, results))

# ----------------------------
def adjust_imgsz(fps, objs_count, imgsz, min_size=MIN_DETECTION_SIZE, max_size=MAX_DETECTION_SIZE):
    w, h = imgsz
    if fps < 15:
        if objs_count < 5:
            w = min(int(w * 1.1), max_size[0])
            h = min(int(h * 1.1), max_size[1])
        else:
            w = max(int(w * 0.8), min_size[0])
            h = max(int(h * 0.8), min_size[1])
    elif fps > 25:
        w = max(int(w * 0.9), min_size[0])
        h = max(int(h * 0.9), min_size[1])
    return (w, h)

# ----------------------------
def get_video_writer(save_path, width, height, fps):
    try:
        fourcc = cv2.VideoWriter_fourcc(*"X264")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise Exception("X264 не доступен")
    except:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    return writer

# ----------------------------
def process_camera_sources(cameras, gray=False, save_video=None, conf_threshold=CONFIDENCE_THRESHOLD, face_mode=False):
    global DETECTION_SIZE
    model = get_model()

    for cam_id in cameras:
        try:
            cap = open_camera(cam_id)
        except IOError as e:
            log_message(str(e), "ОШИБКА")
            continue

        frame_queue = Queue(maxsize=2)
        result_queue = Queue(maxsize=2)

        threading.Thread(target=video_worker, args=(cap, frame_queue), daemon=True).start()
        threading.Thread(
            target=infer_worker,
            args=(frame_queue, result_queue, model, conf_threshold, face_mode),
            daemon=True
        ).start()

        writer = None
        if save_video:
            save_path = auto_filename("mp4") if save_video == "auto" else save_video
            h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps_cap = cap.get(cv2.CAP_PROP_FPS) or 25.0
            writer = get_video_writer(save_path, w, h, fps_cap)
            log_message(f"Сохраняем видео: {save_path}, размер: {w}x{h}")

        frame_times = []
        last_fps_log = time.time()

        while True:
            if not result_queue.empty():
                annotated, results = result_queue.get()
                if gray:
                    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
                    annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

                cv2.imshow(f"Камера {cam_id}", annotated)
                if writer:
                    writer.write(annotated)

                frame_times.append(time.time())
                if len(frame_times) > 10:
                    fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                    objs_count = len(results[0].boxes)
                    if time.time() - last_fps_log > 2:
                        log_message(f"Камера {cam_id} | FPS: {fps:.1f} | Объекты: {objs_count}")
                        log_system_load()
                        last_fps_log = time.time()
                    DETECTION_SIZE = adjust_imgsz(fps, objs_count, DETECTION_SIZE)

            if cv2.waitKey(1) & 0xFF == ord("q") or stop_event.is_set():
                log_message(f"Остановка камеры {cam_id}")
                break

        if writer:
            writer.release()
    cv2.destroyAllWindows()

# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv9 / YOLOv8 — Асинхронная система детекции")
    parser.add_argument("--cams", nargs="+", default=["0"], help="Список источников (USB или URL)")
    parser.add_argument("--gray", action="store_true", help="Черно-белый режим")
    parser.add_argument("--save", nargs="?", const="auto", help="Сохранить видео (auto / путь)")
    parser.add_argument("--face", action="store_true", help="Режим распознавания лица")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Порог уверенности")
    args = parser.parse_args()

    # --- Преобразуем источники ---
    sources = []
    for c in args.cams:
        try:
            sources.append(int(c))
        except ValueError:
            sources.append(c)

    # --- Фильтруем только доступные камеры ---
    available_sources = list_available_sources(sources)
    if not available_sources:
        print("[ОШИБКА] Нет доступных источников, выход.")
        exit(1)

    print(f"[INFO] Доступные источники: {available_sources}")

    process_camera_sources(
        available_sources,
        gray=args.gray,
        save_video=args.save,
        conf_threshold=args.conf,
        face_mode=args.face
    )
