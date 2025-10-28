# utils/ipcams.py
import cv2

def parse_source(src):
    """
    Определяет тип источника: USB (число) или IP-камера (URL)
    """
    try:
        return int(src)  # USB-камера
    except ValueError:
        return src      # IP-камера (Android/iOS, RTSP, HTTP)

def open_camera(source, timeout=2.0):
    """
    Пытается открыть камеру (USB или IP) и возвращает cv2.VideoCapture.
    Если камера недоступна, возвращает None.
    """
    src = parse_source(source)
    cap = cv2.VideoCapture(src)
    
    # Ждем несколько кадров для проверки потока
    start_time = cv2.getTickCount()
    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap  # Камера успешно открыта
        # Проверка таймаута
        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed > timeout:
            break

    cap.release()
    print(f"[ОШИБКА] Не удалось открыть камеру: {source}")
    return None

def list_available_sources(sources):
    """
    Фильтрует список источников и возвращает только доступные.
    sources: список [0, 1, "http://192.168.0.10:8080/video"]
    """
    available = []
    for src in sources:
        cap = open_camera(src)
        if cap:
            available.append(src)
            cap.release()
    return available
