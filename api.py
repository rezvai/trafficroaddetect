from ultralytics import YOLO
import cv2

# Загрузка модели YOLO
model = YOLO(r'runs\detect\train7\weights\best.pt')

# Загрузка видео
video_capture = cv2.VideoCapture('video.mp4')

# Получение информации о видео
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_capture.get(cv2.CAP_PROP_FPS)

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
    
    # Предсказание с помощью модели YOLO
    processed_frame = model.predict(frame)
    
    # Получение координат боксов
    boxes = processed_frame[0].boxes.xyxy
    
    # Отображение боксов на изображении
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box.tolist()) 
        cv2.rectangle(processed_frame[0].orig_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Запись обработанного кадра в выходное видео
    output_video.write(processed_frame[0].orig_img)

# Освобождение ресурсов
video_capture.release()
output_video.release()
cv2.destroyAllWindows()