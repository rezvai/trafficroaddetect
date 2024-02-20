# Импортируем библиотеку для использования полученной модели
from ultralytics import YOLO
# Импортиурем библиотеку opencv
import cv2

# Сохраняем в переменную модель с полученными весами при обучение
model = YOLO(r'')

# Загрузка видео
video_capture = cv2.VideoCapture('video.mp4')

# Получение информации о видео (размеры кадра, частота кадров и т. д.)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Создание объекта VideoWriter для записи результата
output_video = cv2.VideoWriter('output_video.mp4', 
                               cv2.VideoWriter_fourcc(*'mp4v'), 
                               fps, 
                               (frame_width, frame_height))

# Обработка каждого кадра видео
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    # Запись предикта о кадре
    processed_frame = model.predict(frame)
    
    # Запись обработанного кадра в выходное видео
    output_video.write(processed_frame)

# Освобождение ресурсов
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
