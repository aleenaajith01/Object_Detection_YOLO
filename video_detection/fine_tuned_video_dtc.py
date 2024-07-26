import cv2
from ultralytics import YOLO


model = YOLO(r"C:\Users\Aleena Ajith\OneDrive\Desktop\YOLO\Video\best (1).pt")


cap = cv2.VideoCapture(r"C:\Users\Aleena Ajith\OneDrive\Desktop\YOLO\Object_detection_using_YOLO\video_detection\cityy.mp4")

out = cv2.VideoWriter('VisDrone1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(cap.get(3)), int(cap.get(4))))


while True:
    ret, frame = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(frame)  # only detect car, truck and bus
    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()

    for box, cls in zip(boxes, clss):
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)

        # calculate center
        cx = int(box[0] + box[2]) // 2
        cy = int(box[1] + box[3]) // 2

        # Plot center
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    out.write(frame)

    resized_frame = cv2.resize(frame, (1020, 550))

    cv2.imshow("vehicle distance calculation", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()