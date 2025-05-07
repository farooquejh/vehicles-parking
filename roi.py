import cv2
import json
from ultralytics import YOLO

def main():
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")
    
    # Load image
    image_path = "parking_frame_0018.jpg"  # Replace with your image
    image = cv2.imread(image_path)
    imgsz = 640
    image = cv2.resize(image, (imgsz, imgsz))
    if image is None:
        print("Error: Image not found!")
        return
    
    # Detect cars (class ID 2 in COCO dataset)
    results = model(image)[0]
    car_class_id = 2
    detected_cars = []
    
    for box in results.boxes:
        if int(box.cls[0]) == car_class_id:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_cars.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for cars
    
    # Let user manually draw ROIs (parking spots)
    manual_rois = []
    while True:
        roi = cv2.selectROI("Draw Parking Spot (Press ENTER to confirm, ESC to stop)", 
                           image, fromCenter=False, showCrosshair=True)
        if roi[2] > 0 and roi[3] > 0:  # If width & height are valid
            manual_rois.append({
                "x": int(roi[0]), "y": int(roi[1]),
                "width": int(roi[2]), "height": int(roi[3])
            })
            # Draw green rectangle for ROI
            cv2.rectangle(image, (roi[0], roi[1]), 
                          (roi[0]+roi[2], roi[1]+roi[3]), (0, 255, 0), 2)
        else:
            break  # Exit on ESC
    
    # Save ROIs and detected cars to JSON
    output_data = {
        "manual_rois": manual_rois,
        "detected_cars": detected_cars
    }
    
    with open("parking_rois.json", "w") as f:
        json.dump(output_data, f, indent=4)
    
    # Save and display the result
    cv2.imwrite("parking_with_rois.jpg", image)
    cv2.imshow("Parking Spots & Cars", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Saved {len(manual_rois)} parking spots to 'parking_rois.json'")

if __name__ == "__main__":
    main()