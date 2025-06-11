import cv2
import numpy as np
import math
from playsound import playsound
import threading

# Try to import mediapipe with more detailed error handling
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    
    # Constants
    STRAIGHT_ARM_THRESHOLD = 160
    VERTICAL_THRESH = 20
    HORIZONTAL_THRESH = 25
    MIN_VISIBILITY = 0.7

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    print("✅ MediaPipe successfully loaded for pose detection")
    
except ImportError as e:
    print(f"❌ MediaPipe import failed: {e}")
    print("📝 Note: MediaPipe may not be compatible with this environment")
    print("🔄 Falling back to dummy pose detection")
    MEDIAPIPE_AVAILABLE = False
    mp_pose = None
    mp_drawing = None
    pose = None
except Exception as e:
    print(f"❌ MediaPipe initialization failed: {e}")
    print("🔄 Falling back to dummy pose detection")
    MEDIAPIPE_AVAILABLE = False
    mp_pose = None
    mp_drawing = None
    pose = None

def play_pose_alert_sound():
    """Play pose alert sound in a separate thread to avoid blocking"""
    try:
        sound_path = "./static/audio/fire_alarm.mp3"
        threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()
    except Exception as e:
        print(f"Error playing pose alert sound: {e}")

def calculate_angle(a, b, c):
    if not MEDIAPIPE_AVAILABLE:
        return 0
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    angle = np.degrees(np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0]))
    angle = np.abs(angle)
    return 360 - angle if angle > 180 else angle

def get_arm_angle_vertical(shoulder, elbow):
    if not MEDIAPIPE_AVAILABLE:
        return 0
    shoulder, elbow = np.array(shoulder), np.array(elbow)
    vec = elbow - shoulder
    angle_rad = math.atan2(vec[0], -vec[1])
    return math.degrees(angle_rad)

def detect_l_pose(frame):
    if not MEDIAPIPE_AVAILABLE:
        # Return original frame with clear status message
        cv2.putText(frame, "POSE DETECTION: DISABLED", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(frame, "MediaPipe not available in this environment", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, "Pose detection requires MediaPipe library", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        return frame, False
    
    try:
        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        detected = False
        
        # Add status indicator
        cv2.putText(frame, "POSE DETECTION: ACTIVE", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            try:
                ls, rs = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                le, re = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value], lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                lw, rw = lm[mp_pose.PoseLandmark.LEFT_WRIST.value], lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                visible = all(p.visibility > MIN_VISIBILITY for p in [ls, rs, le, re, lw, rw])
                if visible:
                    ls_c, rs_c = [ls.x, ls.y], [rs.x, rs.y]
                    le_c, re_c = [le.x, le.y], [re.x, re.y]
                    lw_c, rw_c = [lw.x, lw.y], [rw.x, rw.y]

                    left_angle = calculate_angle(ls_c, le_c, lw_c)
                    right_angle = calculate_angle(rs_c, re_c, rw_c)
                    left_vert = get_arm_angle_vertical(ls_c, le_c)
                    right_vert = get_arm_angle_vertical(rs_c, re_c)

                    is_ls = left_angle > STRAIGHT_ARM_THRESHOLD
                    is_rs = right_angle > STRAIGHT_ARM_THRESHOLD
                    is_lv = abs(left_vert) <= VERTICAL_THRESH or abs(abs(left_vert) - 180) <= VERTICAL_THRESH
                    is_rv = abs(right_vert) <= VERTICAL_THRESH or abs(abs(right_vert) - 180) <= VERTICAL_THRESH
                    is_lh = abs(abs(left_vert) - 90) <= HORIZONTAL_THRESH
                    is_rh = abs(abs(right_vert) - 90) <= HORIZONTAL_THRESH

                    if (is_lv and is_rh and is_ls and is_rs) or (is_rv and is_lh and is_rs and is_ls):
                        detected = True
                        play_pose_alert_sound()  # Play alert sound when L-pose detected
                        cv2.putText(frame, "L POSE DETECTED - EMERGENCY ALERT!", 
                                   (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        cv2.putText(frame, "HELP REQUESTED", 
                                   (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error processing landmarks: {e}")
                cv2.putText(frame, "Pose processing error", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if mp_drawing and results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        return frame, detected
    
    except Exception as e:
        print(f"Error in pose detection: {e}")
        cv2.putText(frame, "POSE DETECTION: ERROR", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Error: {str(e)[:50]}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame, False

# Class wrapper to match gear/fire interface
class pose_detection:
    def __init__(self, sound_path="./static/audio/fire_alarm.mp3"):
        self.available = MEDIAPIPE_AVAILABLE
        self.sound_path = sound_path
        
        if self.available:
            print("✅ Pose detection initialized successfully")
        else:
            print("⚠️  Pose detection initialized in fallback mode (MediaPipe unavailable)")

    def play_alert_sound(self):
        """Play alert sound in a separate thread to avoid blocking"""
        try:
            threading.Thread(target=playsound, args=(self.sound_path,), daemon=True).start()
        except Exception as e:
            print(f"Error playing pose alert sound: {e}")

    def process(self, img, flag=False):
        if not flag:
            return False, []
        
        if not self.available:
            # Add visual indicator that pose detection is disabled
            cv2.putText(img, "POSE DETECTION: UNAVAILABLE", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return False, []
        
        try:
            frame, detected = detect_l_pose(img)
            if detected:
                self.play_alert_sound()  # Play alert sound
                # Return full frame as bounding box for consistency
                return True, [[0, 0, img.shape[1], img.shape[0]]]
            else:
                return False, []
        except Exception as e:
            print(f"Error in pose_detection.process: {e}")
            cv2.putText(img, "Pose detection error", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return False, []