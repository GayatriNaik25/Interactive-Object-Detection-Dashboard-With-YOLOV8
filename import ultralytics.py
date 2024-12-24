import ultralytics
ultralytics.checks()

from ultralytics import YOLO

#load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt',"v8")

#predict on an image
detection_output = model.predict(source=r"D:\cv2 project\yolo\video_sample1.mp4",conf=0.25,save=True)

#Disply tensor array
print(detection_output)

#Disply numpy array
print(detection_output[0].numpy())