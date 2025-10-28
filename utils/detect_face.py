# utils/detect_face.py

import cv2
import torch

def detect_objects_for_face(frame, model, conf_threshold=0.6, imgsz=(640, 384)):
    """
    Обнаружение только класса 'face' (или person→face в кастомной модели).
    Отрисовка рамки и текста FACE 
    """
    input_frame = cv2.resize(frame, imgsz)
    results = model(input_frame, conf=conf_threshold, verbose=False)
    annotated_frame = frame.copy()

    # Проверяем наличие детекций
    if not results or len(results[0].boxes) == 0:
        return annotated_frame, results

    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id].lower()

        # 💡 Берём только "face"
        if cls_name != "face" or conf < conf_threshold:
            continue

        # --- Преобразуем координаты в исходное разрешение ---
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        scale_x = frame.shape[1] / imgsz[0]
        scale_y = frame.shape[0] / imgsz[1]
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

        # --- Отрисовка рамки и текста ---
        color = (0, 255, 0)  # зелёный
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

        text = "FACE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 4
        text_w, text_h = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = x1 + (x2 - x1) // 2 - text_w // 2
        text_y = y2 + text_h + 10

        # Надпись FACE
        cv2.putText(annotated_frame, text, (text_x, text_y), font, font_scale, color, thickness)

        # Уровень уверенности
        cv2.putText(annotated_frame, f"{conf:.2f}", (x1, y1 - 10), font, 0.6, color, 2)

    return annotated_frame, results
