import cv2
import json
import os
from ultralytics import YOLO

def box_inside(roi, car_box):
    # Convert ROI (x,y,w,h) to (x1,y1,x2,y2)
    rx1, ry1 = roi['x'], roi['y']
    rx2, ry2 = rx1 + roi['width'], ry1 + roi['height']
    
    # Car box is already in (x1,y1,x2,y2)
    cx1, cy1, cx2, cy2 = car_box['x1'], car_box['y1'], car_box['x2'], car_box['y2']
    
    # Check for overlap
    overlap_x = max(0, min(rx2, cx2) - max(rx1, cx1))
    overlap_y = max(0, min(ry2, cy2) - max(ry1, cy1))
    overlap_area = overlap_x * overlap_y
    roi_area = roi['width'] * roi['height']
    
    # Consider it inside if the overlap is significant (more than 15% of ROI area)
    return overlap_area > (0.40 * roi_area)

def main():
    # Load YOLOv8 model
    model = YOLO("yolov8l.pt")
    
    # Load configuration from JSON
    try:
        with open("parking_config.json", "r") as f:
            config = json.load(f)
            image_path = config["image_path"]
            original_dimensions = config.get("original_dimensions", {"width": None, "height": None})
            standard_dimensions = config.get("standard_dimensions", {"width": 1280, "height": 720})
            
            standard_width = standard_dimensions["width"]
            standard_height = standard_dimensions["height"]
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading config: {e}")
        # Default values
        image_path = "frameaj211_110.jpg"
        standard_width = 1028
        standard_height = 1028
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Load image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Failed to load image from '{image_path}'!")
        return
    
    # Get original dimensions
    original_height, original_width = original_image.shape[:2]
    print(f"Original image dimensions: {original_width}x{original_height} pixels")
    
    # Resize image to match the standardized size used during ROI definition
    image = cv2.resize(original_image, (standard_width, standard_height),interpolation=cv2.INTER_LINEAR)
    resized_height, resized_width = image.shape[:2]
    print(f"Resized to standard dimensions: {resized_width}x{resized_height} pixels")
    
    # Load ROIs from JSON - these are in the standardized resolution
    try:
        with open("parking_rois.json", "r") as f:
            data = json.load(f)
            manual_rois = data["manual_rois"]
    except FileNotFoundError:
        print("Error: 'parking_rois.json' not found. Run the ROI definition script first.")
        return
    
    # Make a debug copy to verify ROIs match
    debug_image = image.copy()
    
    # Draw the loaded ROIs in green on debug image
    for i, roi in enumerate(manual_rois):
        x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_image, f"Spot {i+1}", (x + 5, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save debug image to verify ROIs are loaded correctly
    cv2.imwrite("debug_rois.jpg", debug_image)
    print("Saved debug image showing loaded ROIs to 'debug_rois.jpg'")
    
    # Detect cars
    results = model(image)[0]
    car_class_id = 2
    detected_cars = []
    
    for box in results.boxes:
        if int(box.cls[0]) == car_class_id:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_cars.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
    
    # Check each ROI for cars
    for i, roi in enumerate(manual_rois):
        rx, ry, rw, rh = roi["x"], roi["y"], roi["width"], roi["height"]
        
        # Check if any car is inside
        occupied = False
        for car in detected_cars:
            if box_inside(roi, car):
                occupied = True
                # Draw car in red
                cv2.rectangle(image, (car["x1"], car["y1"]), 
                             (car["x2"], car["y2"]), (0, 0, 255), 2)
        
        # Always draw blue bounding box for ROI
        cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)  # Blue for ROI
        
        # Add spot number for reference
        cv2.putText(image, f"Spot {i+1}", (rx + 5, ry + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw additional green bounding box for empty spots
        if not occupied:
            # Draw green bounding box slightly inside the blue one for empty spots
            offset = 5
            cv2.rectangle(image, (rx + offset, ry + offset), 
                         (rx + rw - offset, ry + rh - offset), (0, 255, 0), 2)  # Green for empty
            cv2.putText(image, "Empty", (rx + 5, ry + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Green text
        else:
            cv2.putText(image, "Occupied", (rx + 5, ry + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Red text
    
    # Save and display result
    cv2.imwrite("parking_occupancy.jpg", image)
    
    # Display image with OpenCV (will scale to fit screen)
    cv2.namedWindow("Parking Occupancy", cv2.WINDOW_NORMAL)
    cv2.imshow("Parking Occupancy", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Parking analysis saved to 'parking_occupancy.jpg'")

if __name__ == "__main__":
    main()