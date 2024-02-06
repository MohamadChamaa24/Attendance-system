import cv2
import torch

cap = cv2.VideoCapture(1)
wht = 640
conf_threshold = 0.3

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)
model = model.autoshape()  # automatically shape input to the model size

phone_detected = False  # Flag to check if a cell phone is detected

while True:
    ret, img = cap.read()
    
    # Run inference
    results = model(img)

    # Process results
    for det in results.xyxy[0]:
        classId = int(det[5])
        confidence = det[4]

        if confidence > conf_threshold and classId == 67:  # Class 67 is typically associated with phones
            x, y, w, h = map(int, det[:4])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"Phone: {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

            phone_detected = True  # Set the flag to indicate phone detection

    cv2.imshow("Webcam", img)

    if phone_detected:
        print("Please don't use a cell phone to capture your image.")
        break

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
