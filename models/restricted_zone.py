from ultralytics import YOLO
import cv2
import numpy as np
from playsound import playsound
import threading

class restricted_zone_detection:
    """
    This class detects persons entering restricted areas using YOLO person detection.
    It monitors the entire camera POV and triggers alerts when persons are detected.
    """
    
    def __init__(self, model_path="yolov8n.pt", conf=0.5, sound_path="./static/audio/fire_alarm.mp3"):
        """
        Initialize the restricted zone detection model.
        
        Args:
            model_path: Path to YOLO model (uses YOLOv8 nano for person detection)
            conf: Confidence threshold for person detection
            sound_path: Path to alert sound file
        """
        self.model = YOLO(model_path)
        self.confidence = conf
        self.sound_path = sound_path
        self.person_class_id = 0  # COCO dataset class ID for 'person'
        
    def play_alert_sound(self):
        """Play alert sound in a separate thread to avoid blocking"""
        try:
            threading.Thread(target=playsound, args=(self.sound_path,), daemon=True).start()
        except Exception as e:
            print(f"Error playing restricted zone alert sound: {e}")

    def process(self, img, flag=True):
        """
        Process the image to detect persons in the restricted area (entire camera POV).
        
        Args:
            img: Input image/frame from camera
            flag: Boolean flag to enable/disable detection
            
        Returns:
            tuple: (detection_found, bounding_boxes_list)
        """
        if not flag:
            return (False, [])

        bb_boxes = []
        
        try:
            # Run YOLO detection
            results = self.model(img, verbose=False)
            
            # Process detections
            for box in results[0].boxes:
                # Check if detection is a person and meets confidence threshold
                if (int(box.cls[0]) == self.person_class_id and 
                    float(box.conf[0]) > self.confidence):
                    
                    # Extract bounding box coordinates
                    bb = list(map(int, box.xyxy[0]))
                    bb_boxes.append(bb)
                    
                    # Draw bounding box on the image
                    x1, y1, x2, y2 = bb
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # Add warning text
                    cv2.putText(img, "RESTRICTED AREA BREACH!", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 0, 255), 2)
                    
                    # Add confidence score
                    confidence_text = f"Person: {float(box.conf[0]):.2f}"
                    cv2.putText(img, confidence_text, 
                              (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 0, 255), 1)

            # If persons detected, trigger alert
            if len(bb_boxes) > 0:
                found = True
                self.play_alert_sound()
                
                # Add overall warning overlay
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (img.shape[1], 50), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                cv2.putText(img, f"ALERT: {len(bb_boxes)} PERSON(S) IN RESTRICTED ZONE", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                found = False
                
        except Exception as e:
            print(f"Error in restricted zone detection: {e}")
            found = False
            
        return (found, bb_boxes)

    def draw_zone_overlay(self, img):
        """
        Draw a semi-transparent overlay to indicate the entire frame is a restricted zone.
        
        Args:
            img: Input image
            
        Returns:
            img: Image with zone overlay
        """
        overlay = img.copy()
        
        # Draw border around entire frame
        cv2.rectangle(overlay, (5, 5), (img.shape[1]-5, img.shape[0]-5), (0, 255, 255), 3)
        
        # Add corner markers
        corner_size = 30
        corners = [
            (10, 10), (img.shape[1]-40, 10),
            (10, img.shape[0]-40), (img.shape[1]-40, img.shape[0]-40)
        ]
        
        for corner in corners:
            cv2.rectangle(overlay, corner, 
                         (corner[0] + corner_size, corner[1] + corner_size), 
                         (0, 255, 255), -1)
        
        # Add zone label
        cv2.putText(overlay, "RESTRICTED ZONE - MONITORING ACTIVE", 
                   (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 255), 2)
        
        # Blend overlay with original image
        cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)
        
        return img