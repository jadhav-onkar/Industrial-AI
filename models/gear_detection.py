from ultralytics import YOLO
import cv2
from playsound import playsound
import threading

class gear_detection():
    """
    this class will be used to detect safety gear on people.

    Args:
    model_path: path to model.
    conf: minimum confidence to consider detection.
    
    """
    def __init__(self, model_path, conf=0.85, sound_path="./static/audio/fire_alarm.mp3"):
        self.model = YOLO(model_path)
        self.confidence = conf
        self.sound_path = sound_path

    def play_alert_sound(self):
        """Play alert sound in a separate thread to avoid blocking"""
        try:
            threading.Thread(target=playsound, args=(self.sound_path,), daemon=True).start()
        except Exception as e:
            print(f"Error playing safety gear alert sound: {e}")

    def process(self, img, flag=True):
        """
        this function processes the cv2 frame and returns the
        bounding boxes
        """
        if not flag:
            return (False, [])

        bb_boxes = []
        result = self.model(img, verbose=False)

        for box in result[0].boxes:
            if((int(box.cls[0]) in [2,3,4]) and (float(box.conf[0]) > self.confidence)):
                bb = list(map(int, box.xyxy[0]))
                bb_boxes.append(bb)

        if(len(bb_boxes)):
            found = True
            self.play_alert_sound()  # ✅ Added: Play alert sound when safety gear violation detected
        else:
            found = False
        return (found, bb_boxes)