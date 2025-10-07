# my_cv.py
import cv2
import os
from datetime import datetime

LOG_FILE = "log.txt"
ERROR_FILE = "error.log"

# ANSI-коды для цветов
COLOR_RESET = "\033[0m"
COLOR_INFO = "\033[32m"    # зелёный
COLOR_WARN = "\033[33m"    # жёлтый
COLOR_ERROR = "\033[31m"   # красный

# очищаем логи при запуске
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("=== Новый запуск программы ===\n")
with open(ERROR_FILE, "w", encoding="utf-8") as f:
    f.write("=== Новый запуск программы (ошибки) ===\n")

def log_message(message: str, level: str = "INFO"):
    """Логирует сообщение в консоль с цветом и в log.txt. Ошибки и WARN пишутся также в error.log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {message}"

    # Цвет для терминала
    if level == "INFO":
        color = COLOR_INFO
    elif level == "WARN":
        color = COLOR_WARN
    elif level == "ERROR":
        color = COLOR_ERROR
    else:
        color = COLOR_RESET

    print(f"{color}{line}{COLOR_RESET}")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    if level in ("ERROR", "WARN"):
        with open(ERROR_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# -----------------------------
# Работа с изображениями
# -----------------------------
def read_image(file_name):
    path = os.path.join("images", file_name)
    log_message(f"Открываю изображение: {path}")
    img = cv2.imread(path)
    if img is None:
        log_message(f"Файл {path} не найден или не удалось прочитать", "ERROR")
        raise FileNotFoundError(f"Файл {path} не найден или не удалось прочитать")
    return img

def show_image(img, window_name="Image"):
    log_message(f"Показываю изображение в окне: {window_name}")
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convert_to_gray(img):
    log_message("Конвертирую изображение в оттенки серого")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# -----------------------------
# Работа с видео
# -----------------------------
def play_video(file_name):
    path = os.path.join("videos", file_name)
    log_message(f"Запускаю воспроизведение видео: {path}")

    if not os.path.exists(path):
        log_message(f"Видео {path} не найдено", "ERROR")
        raise FileNotFoundError(f"Видео {path} не найдено")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        log_message(f"Не удалось открыть видео: {path}", "ERROR")
        raise ValueError(f"Не удалось открыть видео: {path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if frame_count == 0:
                log_message("Не удалось прочитать ни одного кадра", "ERROR")
            else:
                log_message("Видео закончилось или кадры закончились", "WARN")
            break

        cv2.imshow("Video", frame)
        frame_count += 1

        if cv2.waitKey(25) & 0xFF == ord("q"):
            log_message("Видео остановлено пользователем")
            break

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------
# Работа с камерой
# -----------------------------
def play_camera(camera_index=0):
    log_message(f"Запускаю камеру с индексом {camera_index}")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        log_message(f"Не удалось открыть камеру {camera_index}", "ERROR")
        raise Exception(f"Не удалось открыть камеру {camera_index}")

    log_message("Камера запущена. Нажмите Q для выхода.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if frame_count == 0:
                log_message("Не удалось прочитать ни одного кадра с камеры", "ERROR")
            else:
                log_message("Пропущен кадр с камеры", "WARN")
            break

        cv2.imshow("Camera", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            log_message("Камера остановлена пользователем")
            break

    cap.release()
    cv2.destroyAllWindows()
