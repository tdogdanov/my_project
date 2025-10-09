import argparse
import cv2
import time
import traceback
import torch
from datetime import datetime
from my_cv import (
    read_image,
    show_image,
    convert_to_gray,
    play_video,
    play_camera,
    save_image,
    save_video,
    log_message,
)
from ultralytics import YOLO

# -------- YOLO --------
_model = None
def get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_message(f"Загружаем YOLOv8n на {device}...")
        _model = YOLO("yolov8n.pt").to(device)
    return _model


def detect_objects(frame):
    model = get_model()
    results = model(frame)
    return results[0].plot()


def auto_filename(extension: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"output_{timestamp}.{extension}"


# -------- Видео и камера с детекцией --------
def play_video_with_detection(video_path, show_gray=False, save_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_message(f"Не удалось открыть видео {video_path}", "ERROR")
        return

    prev_time = time.time()
    frame_count = 0
    saved_frames = []
    timestamps = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = detect_objects(frame)
            if show_gray:
                frame = convert_to_gray(frame)

            cv2.imshow("Video Object Detection", frame)

            if save_path:
                saved_frames.append(frame.copy())
                timestamps.append(time.time())

            frame_count += 1
            if frame_count >= 10:
                now = time.time()
                fps = frame_count / (now - prev_time)
                log_message(f"FPS: {fps:.2f}")
                prev_time, frame_count = now, 0

            if cv2.waitKey(25) & 0xFF == ord("q"):
                log_message("Видео остановлено пользователем")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if save_path and saved_frames:
            if len(timestamps) > 1:
                intervals = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
                real_fps = 1 / (sum(intervals) / len(intervals))
            else:
                real_fps = 25
            save_video(saved_frames, save_path, fps=real_fps)


def play_camera_with_detection(camera_index=0, show_gray=False, save_path=None):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log_message(f"Не удалось открыть камеру {camera_index}", "ERROR")
        return

    log_message(f"Камера {camera_index} запущена. Q - выход, Ctrl+C - прерывание.")
    frame_count = 0
    saved_frames = []
    timestamps = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log_message("Не удалось получить кадр с камеры", "ERROR")
                break

            frame = detect_objects(frame)
            if show_gray:
                frame = convert_to_gray(frame)

            cv2.imshow("Camera Object Detection", frame)

            if save_path:
                saved_frames.append(frame.copy())
                timestamps.append(time.time())

            frame_count += 1
            if frame_count >= 10:
                if len(timestamps) > 1:
                    intervals = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
                    fps = 1 / (sum(intervals) / len(intervals))
                    log_message(f"FPS: {fps:.2f}")
                frame_count = 0

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log_message("Камера остановлена пользователем")
                break
    except KeyboardInterrupt:
        log_message("Камера остановлена через Ctrl+C")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if save_path and saved_frames:
            if len(timestamps) > 1:
                intervals = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
                real_fps = 1 / (sum(intervals) / len(intervals))
            else:
                real_fps = 25
            save_video(saved_frames, save_path, fps=real_fps)



def main():
    parser = argparse.ArgumentParser(description="OpenCV + YOLOv8 Object Detection")
    parser.add_argument("--image", type=str, help="Путь к изображению")
    parser.add_argument("--video", type=str, help="Путь к видео")
    parser.add_argument("--camera", type=int, nargs="?", const=0, help="Индекс камеры")
    parser.add_argument("--detect", action="store_true", help="Включить детекцию объектов")
    parser.add_argument("--gray", action="store_true", help="Показать в градациях серого")
    parser.add_argument("--save", type=str, nargs="?", const="auto", help="Сохранить результат (out.jpg/out.mp4)")
    args = parser.parse_args()

    # -------- Изображение --------
    if args.image:
        try:
            img = read_image(args.image)
            if args.detect:
                img = detect_objects(img)
            if args.gray:
                img = convert_to_gray(img)

            save_path = None
            if args.save:
                save_path = args.save if args.save != "auto" else auto_filename("jpg")
                save_image(img, save_path)

            # Показываем всегда
            show_image(img, "Processed Image")

        except Exception as e:
            log_message(f"Ошибка при обработке изображения: {e}", "ERROR")
            traceback.print_exc()

    # -------- Видео --------
    if args.video:
        try:
            save_path = None
            if args.save:
                save_path = args.save if args.save != "auto" else auto_filename("mp4")

            if args.detect:
                play_video_with_detection(args.video, args.gray, save_path)
            else:
                if args.gray:
                    log_message("--gray работает только с --detect для видео", "WARN")
                play_video(args.video)

        except Exception as e:
            log_message(f"Ошибка при обработке видео: {e}", "ERROR")
            traceback.print_exc()

    # -------- Камера --------
    if args.camera is not None:
        try:
            save_path = None
            if args.save:
                save_path = args.save if args.save != "auto" else auto_filename("mp4")

            if args.detect:
                play_camera_with_detection(args.camera, args.gray, save_path)
            else:
                if args.gray:
                    log_message("--gray работает только с --detect для камеры", "WARN")
                play_camera(args.camera)

        except Exception as e:
            log_message(f"Ошибка при работе с камерой: {e}", "ERROR")
            traceback.print_exc()


if __name__ == "__main__":
    main()
