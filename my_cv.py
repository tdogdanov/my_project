import cv2
import os

# -----------------------------
# Работа с изображениями
# -----------------------------
import cv2
import os

# -----------------------------
# Работа с изображениями
# -----------------------------
def read_image(file_name):
    path = os.path.join('images', file_name)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Файл {path} не найден")
    return img

def show_image(img, window_name='Image'):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Работа с видео
# -----------------------------
def play_video(file_name):
    path = os.path.join('videos', file_name)
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Файл {path} не найден или не удалось открыть")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# Работа с камерой
# -----------------------------
def play_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise Exception(f"Не удалось открыть камеру {camera_index}")

    print("Нажмите 'q', чтобы выйти")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

