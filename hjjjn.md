yolo_realtime_project/
├── data/
│   └── classes.txt           # Список классов (например, COCO 80 классов)
├── models/
│   └── yolov8n.pt            # Предобученная модель YOLOv8 (или yolov9.pt)
├── videos/                   # Папка для тестовых видео
├── outputs/                  # Папка для сохранения результатов
├── utils/
│   ├── draw_boxes.py         # Функция отрисовки боксов и меток
│   ├── video_stream.py       # Класс для работы с видеопотоком
├── requirements.txt          # Необходимые библиотеки
├── detect_realtime.py        # Основной скрипт детекции
└── README.md
