import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Perform inference on an image
image_path = 'traffic-car.jpeg'
results = model(image_path)

# Extract bounding boxes, classes, names, and confidences
boxes = results[0].boxes.xyxy.tolist()
classes = results[0].boxes.cls.tolist()
names = results[0].names
confidences = results[0].boxes.conf.tolist()

# Load the image using OpenCV
image = cv2.imread(image_path)

# Iterate through the results and draw the bounding boxes
for box, cls, conf in zip(boxes, classes, confidences):
    x1, y1, x2, y2 = map(int, box)
    confidence = conf
    detected_class = cls
    name = names[int(cls)]
    
    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
# Save or display the image
#cv2.imwrite('output.jpg', image)

small = cv2.resize(image, (0,0), fx=0.3, fy=0.25) 

cv2.imshow('Image', small)
cv2.waitKey(0)
cv2.destroyAllWindows()
