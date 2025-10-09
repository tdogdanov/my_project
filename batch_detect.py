# batch_detect_quick.py
import os
import cv2
from main import detect_objects, auto_filename
from my_cv import read_image, save_image, log_message

IMAGES_DIR = "images"

def process_all_images_quick():
    for file_name in os.listdir(IMAGES_DIR):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                log_message(f"Обрабатываю {file_name}...")
                img = read_image(file_name)
                img = detect_objects(img)

                         # Показываем изображение до нажатия Q
                cv2.imshow("Detected", img)
                log_message("Нажмите Q, чтобы закрыть это изображение")
                while True:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                cv2.destroyAllWindows()

                # Сохраняем результат
                save_path = auto_filename("jpg")
                save_image(img, save_path)

            except Exception as e:
                log_message(f"Ошибка с {file_name}: {e}", "ERROR")

if __name__ == "__main__":
    process_all_images_quick()
