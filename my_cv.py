import cv2
import os
import time
from datetime import datetime

LOG_FILE = "log.txt"
ERROR_FILE = "error.log"

# ANSI-коды для цветов
COLOR_RESET = "\033[0m"
COLOR_INFO = "\033[32m"
COLOR_WARN = "\033[33m"
COLOR_ERROR = "\033[31m"

# очищаем логи при запуске
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=== Новый запуск программы ===\n")
with open(ERROR_FILE, "w", encoding="utf-8") as f:
    f.write("=== Новый запуск программы (ошибки) ===\n")


def log_message(message: str, level: str = "INFO"):
    """Логирование сообщений в консоль и файл"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {message}"
    color = COLOR_INFO if level == "INFO" else COLOR_WARN if level == "WARN" else COLOR_ERROR
    print(f"{color}{line}{COLOR_RESET}")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if level in ("WARN", "ERROR"):
        with open(ERROR_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def resolve_path(file_name, base_folder):
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(base_folder, file_name)


# -------- Изображения --------
def read_image(file_name):
    path = resolve_path(file_name, "images")
    log_message(f"Открываю изображение: {path}")
    img = cv2.imread(path)
    if img is None:
        log_message(f"Файл {path} не найден или не удалось прочитать", "ERROR")
        raise FileNotFoundError(f"Файл {path} не найден или не удалось прочитать")
    return img


def show_image(img, window_name="Image"):
    log_message(f"Показываю изображение: {window_name}")
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_to_gray(img):
    log_message("Конвертирую изображение в оттенки серого")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def save_image(img, file_name="output.jpg"):
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", file_name)
    log_message(f"Сохраняю изображение: {path}")
    cv2.imwrite(path, img)
    return path


# -------- Видео --------
def play_video(file_name):
    path = resolve_path(file_name, "videos")
    log_message(f"Запускаю воспроизведение видео: {path}")
    if not os.path.exists(path):
        log_message(f"Видео {path} не найдено", "ERROR")
        raise FileNotFoundError(f"Видео {path} не найдено")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log_message(f"Не удалось открыть видео: {path}", "ERROR")
        raise ValueError(f"Не удалось открыть видео: {path}")

    prev_time = time.time()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            log_message("Видео закончилось или кадры закончились")
            break
        cv2.imshow("Video", frame)

        frame_count += 1
        if frame_count >= 10:
            now = time.time()
            fps = frame_count / (now - prev_time)
            log_message(f"FPS: {fps:.2f}")
            prev_time, frame_count = now, 0

        if cv2.waitKey(25) & 0xFF == ord("q"):
            log_message("Видео остановлено пользователем")
            break

    cap.release()
    cv2.destroyAllWindows()


def save_video(frames, file_name="output.mp4", fps=25):
    """Сохраняет список кадров в видео с заданным fps"""
    if not frames:
        log_message("Нет кадров для сохранения видео", "ERROR")
        return None

    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", file_name)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    log_message(f"Сохранил видео: {path}")
    return path


# -------- Камера --------
def play_camera(camera_index=0):
    log_message(f"Запускаю камеру с индексом {camera_index}")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log_message(f"Не удалось открыть камеру {camera_index}", "ERROR")
        raise Exception(f"Не удалось открыть камеру {camera_index}")

    log_message("Камера запущена. Нажмите Q для выхода.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log_message("Не удалось получить кадр с камеры", "ERROR")
                break
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log_message("Камера остановлена пользователем")
                break
    except KeyboardInterrupt:
        log_message("Камера остановлена через Ctrl+C")
    finally:
        cap.release()
        cv2.destroyAllWindows()
