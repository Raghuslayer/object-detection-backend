import cv2
import os

# Load YOLO model
MODEL_CFG = "models/yolov3.cfg"
MODEL_WEIGHTS = "models/yolov3.weights"
CLASS_NAMES = "models/coco.names"

net = cv2.dnn.readNetFromDarknet(MODEL_CFG, MODEL_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(CLASS_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objects(image_path, filename):
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Convert image to YOLO format
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Run YOLO detection
    detections = net.forward(output_layers)

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"{classes[class_id]}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    processed_path = f"static/processed/{filename}"
    cv2.imwrite(processed_path, img)
    return processed_path
