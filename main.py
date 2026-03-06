import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from pygame import mixer
import os
import time
import csv
from datetime import datetime

# -------------------------------------------------
# Base Directory
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------
# Output Directories
# -------------------------------------------------
LOG_DIR   = os.path.join(BASE_DIR, "logs")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

# -------------------------------------------------
# CSV Log File
# -------------------------------------------------
log_date   = datetime.now().strftime("%Y-%m-%d")
LOG_FILE   = os.path.join(LOG_DIR, f"drowsiness_log_{log_date}.csv")
log_exists = os.path.isfile(LOG_FILE)
log_csv    = open(LOG_FILE, "a", newline="")
log_writer = csv.writer(log_csv)
if not log_exists:
    log_writer.writerow(["Timestamp", "Event", "Duration_sec", "Clip_File"])

def log_event(event, duration=0, clip_file=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_writer.writerow([timestamp, event, round(duration, 1), clip_file])
    log_csv.flush()
    print(f"[LOG] {timestamp} | {event} | {duration}s | {clip_file}")

# -------------------------------------------------
# Load Alarm Sound
# -------------------------------------------------
mixer.init()
alarm_path = os.path.join(BASE_DIR, "alarm", "alarm.wav")
mixer.music.load(alarm_path)

# -------------------------------------------------
# Load CNN Model
# -------------------------------------------------
model = tf.keras.models.load_model(os.path.join(BASE_DIR, "cnn_model", "eye_model.h5"))

# -------------------------------------------------
# MediaPipe FaceMesh Setup
# -------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------------------------
# Webcam
# -------------------------------------------------
cap      = cv2.VideoCapture(0)
FRAME_W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -------------------------------------------------
# Video Clip Recording
# -------------------------------------------------
video_writer    = None
is_recording    = False
recording_start = None
clip_name_curr  = ""

def start_recording(alert_type):
    global video_writer, is_recording, recording_start, clip_name_curr
    if not is_recording:
        clip_name_curr = datetime.now().strftime(f"{alert_type}_%Y%m%d_%H%M%S.mp4")  # ← .mp4
        clip_path      = os.path.join(CLIPS_DIR, clip_name_curr)
        fourcc         = cv2.VideoWriter_fourcc(*"mp4v")                               # ← mp4v
        video_writer   = cv2.VideoWriter(clip_path, fourcc, 20.0, (FRAME_W, FRAME_H))
        is_recording    = True
        recording_start = time.time()

def stop_recording():
    global video_writer, is_recording, recording_start, clip_name_curr
    if is_recording and video_writer:
        duration       = round(time.time() - recording_start, 1)
        saved_name     = clip_name_curr
        video_writer.release()
        video_writer    = None
        is_recording    = False
        recording_start = None
        clip_name_curr  = ""
        return duration, saved_name
    return 0, ""

# -------------------------------------------------
# Landmark Indexes
# -------------------------------------------------
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 78
MOUTH_RIGHT  = 308

LEFT_EAR  = 234
RIGHT_EAR = 454

# -------------------------------------------------
# Thresholds
# -------------------------------------------------
DROWSY_FRAME_THRESHOLD = 20
YAWN_FRAME_THRESHOLD   = 15
YAWN_RATIO_THRESHOLD   = 0.06
HEAD_TILT_THRESHOLD    = 25
TILT_FRAME_THRESHOLD   = 30

# -------------------------------------------------
# Counters & State Tracking
# -------------------------------------------------
drowsy_frame_count = 0
yawn_frame_count   = 0
tilt_frame_count   = 0

was_drowsy  = False
was_yawning = False
was_tilting = False

drowsy_start = None
yawn_start   = None
tilt_start   = None

# -------------------------------------------------
# FPS
# -------------------------------------------------
fps_start_time  = time.time()
fps_frame_count = 0
fps             = 0

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def extract_eye(frame, landmarks, eye_points, padding=14):
    h, w, _ = frame.shape
    coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]
    x1 = max(min(p[0] for p in coords) - padding, 0)
    x2 = min(max(p[0] for p in coords) + padding, w)
    y1 = max(min(p[1] for p in coords) - padding, 0)
    y2 = min(max(p[1] for p in coords) + padding, h)
    eye = frame[y1:y2, x1:x2]
    if eye.size == 0:
        return None
    eye = cv2.resize(eye, (24, 24))
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = eye / 255.0
    eye = eye.reshape(1, 24, 24, 1)
    return eye

def draw_eye_box(frame, landmarks, eye_points, state="Awake", padding=14):
    h, w, _ = frame.shape
    coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]
    x1 = max(min(p[0] for p in coords) - padding, 0)
    x2 = min(max(p[0] for p in coords) + padding, w)
    y1 = max(min(p[1] for p in coords) - padding, 0)
    y2 = min(max(p[1] for p in coords) + padding, h)
    color = (0, 0, 255) if state == "Drowsy" else (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

def detect_yawn(landmarks, h, w):
    top    = np.array([landmarks[MOUTH_TOP].x * w,    landmarks[MOUTH_TOP].y * h])
    bottom = np.array([landmarks[MOUTH_BOTTOM].x * w, landmarks[MOUTH_BOTTOM].y * h])
    left   = np.array([landmarks[MOUTH_LEFT].x * w,   landmarks[MOUTH_LEFT].y * h])
    right  = np.array([landmarks[MOUTH_RIGHT].x * w,  landmarks[MOUTH_RIGHT].y * h])
    mouth_open  = np.linalg.norm(top - bottom)
    mouth_width = np.linalg.norm(left - right)
    if mouth_width == 0:
        return False, 0.0
    ratio = mouth_open / mouth_width
    return ratio > YAWN_RATIO_THRESHOLD, round(ratio, 3)

def detect_head_tilt(landmarks, h, w):
    left_ear  = np.array([landmarks[LEFT_EAR].x * w,  landmarks[LEFT_EAR].y * h])
    right_ear = np.array([landmarks[RIGHT_EAR].x * w, landmarks[RIGHT_EAR].y * h])
    angle = np.degrees(np.arctan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0]))
    return abs(angle) > HEAD_TILT_THRESHOLD, round(angle, 1)

def draw_hud(frame, fps, latency_ms, state, yawn_detected, tilt_detected, tilt_angle, recording):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 205), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"FPS     : {fps}",         (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, f"Latency : {latency_ms} ms",(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    sc = (0, 0, 255) if state == "Drowsy" else (0, 255, 0) if state == "Awake" else (255, 255, 0)
    cv2.putText(frame, f"State   : {state}",        (10, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 1)

    yc = (0, 0, 255) if yawn_detected else (0, 255, 0)
    cv2.putText(frame, f"Yawning : {'YES' if yawn_detected else 'NO'}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yc, 1)

    tc = (0, 0, 255) if tilt_detected else (0, 255, 0)
    tt = f"YES ({tilt_angle}d)" if tilt_detected else f"NO  ({tilt_angle}d)"
    cv2.putText(frame, f"Tilt    : {tt}",           (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tc, 1)

    if state == "Drowsy" or yawn_detected or tilt_detected:
        cv2.putText(frame, "!! ALERT !!",           (10, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # Red REC dot
    if recording:
        cv2.circle(frame, (288, 15), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (252, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# -------------------------------------------------
# Main Loop
# -------------------------------------------------
print("Driver Drowsiness Detection Started. Press ESC to quit.")
print(f"Logs  saved to → {LOG_DIR}")
print(f"Clips saved to → {CLIPS_DIR}")

while True:
    frame_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    state         = "No Face"
    yawn_detected = False
    tilt_detected = False
    tilt_angle    = 0.0
    any_alert     = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # --- Extract Eyes ---
            left_eye  = extract_eye(frame, landmarks, LEFT_EYE)
            right_eye = extract_eye(frame, landmarks, RIGHT_EYE)

            # --- Drowsiness ---
            if left_eye is not None and right_eye is not None:
                both       = np.vstack([left_eye, right_eye])
                preds      = model.predict(both, verbose=0)
                prediction = (preds[0][0] + preds[1][0]) / 2

                if prediction > 0.5:
                    drowsy_frame_count += 1
                    if drowsy_frame_count >= DROWSY_FRAME_THRESHOLD:
                        state     = "Drowsy"
                        any_alert = True
                        if not mixer.music.get_busy():
                            mixer.music.play()
                        if not was_drowsy:
                            was_drowsy   = True
                            drowsy_start = time.time()
                            start_recording("DROWSY")
                else:
                    if was_drowsy:
                        duration = round(time.time() - drowsy_start, 1)
                        dur, clip = stop_recording()
                        log_event("DROWSY", duration, clip)
                        was_drowsy = False
                    drowsy_frame_count = 0
                    state = "Awake"
                    mixer.music.stop()

            # --- Yawning ---
            yawning, _ = detect_yawn(landmarks, h, w)
            if yawning:
                yawn_frame_count += 1
                if yawn_frame_count >= YAWN_FRAME_THRESHOLD:
                    yawn_detected = True
                    any_alert     = True
                    if not mixer.music.get_busy():
                        mixer.music.play()
                    if not was_yawning:
                        was_yawning = True
                        yawn_start  = time.time()
                        start_recording("YAWN")
            else:
                if was_yawning:
                    duration = round(time.time() - yawn_start, 1)
                    dur, clip = stop_recording()
                    log_event("YAWN", duration, clip)
                    was_yawning = False
                yawn_frame_count = 0

            # --- Head Tilt ---
            tilt_raw, tilt_angle = detect_head_tilt(landmarks, h, w)
            if tilt_raw:
                tilt_frame_count += 1
                if tilt_frame_count >= TILT_FRAME_THRESHOLD:
                    tilt_detected = True
                    any_alert     = True
                    if not mixer.music.get_busy():
                        mixer.music.play()
                    if not was_tilting:
                        was_tilting = True
                        tilt_start  = time.time()
                        start_recording("TILT")
            else:
                if was_tilting:
                    duration = round(time.time() - tilt_start, 1)
                    dur, clip = stop_recording()
                    log_event("HEAD_TILT", duration, clip)
                    was_tilting = False
                tilt_frame_count = 0

            # --- Draw Eye Boxes ---
            draw_eye_box(frame, landmarks, LEFT_EYE,  state, padding=14)
            draw_eye_box(frame, landmarks, RIGHT_EYE, state, padding=14)

    # --- Record frame ---
    if is_recording and video_writer:
        video_writer.write(frame)

    # --- FPS ---
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1.0:
        fps             = fps_frame_count
        fps_frame_count = 0
        fps_start_time  = time.time()

    # --- Latency ---
    latency_ms = int((time.time() - frame_start) * 1000)

    # --- Draw HUD ---
    draw_hud(frame, fps, latency_ms, state, yawn_detected, tilt_detected, tilt_angle, is_recording)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------------------------
# Cleanup
# -------------------------------------------------
if is_recording:
    dur, clip = stop_recording()
    log_event("SESSION_END", dur, clip)

log_csv.close()
cap.release()
cv2.destroyAllWindows()
print("Session ended. Logs and clips saved!")
















# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp
# from pygame import mixer
# import os
# import time

# # -------------------------------------------------
# # Base Directory
# # -------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # -------------------------------------------------
# # Load Alarm Sound
# # -------------------------------------------------
# mixer.init()
# alarm_path = os.path.join(BASE_DIR, "alarm", "alarm.wav")
# mixer.music.load(alarm_path)

# # -------------------------------------------------
# # Load CNN Model
# # -------------------------------------------------
# model = tf.keras.models.load_model(os.path.join(BASE_DIR, "cnn_model", "eye_model.h5"))

# # -------------------------------------------------
# # MediaPipe FaceMesh Setup
# # -------------------------------------------------
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # -------------------------------------------------
# # Webcam
# # -------------------------------------------------
# cap = cv2.VideoCapture(0)

# # -------------------------------------------------
# # Eye Landmark Indexes (MediaPipe)
# # -------------------------------------------------
# LEFT_EYE  = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# # Mouth landmarks for yawning detection
# MOUTH_TOP    = 13
# MOUTH_BOTTOM = 14
# MOUTH_LEFT   = 78
# MOUTH_RIGHT  = 308

# # Head tilt landmarks
# LEFT_EAR  = 234
# RIGHT_EAR = 454

# # -------------------------------------------------
# # Thresholds
# # -------------------------------------------------
# DROWSY_FRAME_THRESHOLD = 10    # consecutive frames before drowsy alarm
# YAWN_FRAME_THRESHOLD   = 15    # consecutive frames before yawn alarm
# YAWN_RATIO_THRESHOLD   = 0.06  # mouth open ratio
# HEAD_TILT_THRESHOLD    = 30    # degrees
# TILT_FRAME_THRESHOLD = 30
# # -------------------------------------------------
# # Counters
# # -------------------------------------------------
# drowsy_frame_count = 0
# yawn_frame_count   = 0
# tilt_frame_count = 0

# # -------------------------------------------------
# # FPS Tracking
# # -------------------------------------------------
# fps_start_time  = time.time()
# fps_frame_count = 0
# fps             = 0

# # -------------------------------------------------
# # Extract Eye Region for CNN
# # -------------------------------------------------
# def extract_eye(frame, landmarks, eye_points, padding=14):
#     h, w, _ = frame.shape
#     coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]

#     x1 = max(min(p[0] for p in coords) - padding, 0)
#     x2 = min(max(p[0] for p in coords) + padding, w)
#     y1 = max(min(p[1] for p in coords) - padding, 0)
#     y2 = min(max(p[1] for p in coords) + padding, h)

#     eye = frame[y1:y2, x1:x2]
#     if eye.size == 0:
#         return None

#     eye = cv2.resize(eye, (24, 24))
#     eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
#     eye = eye / 255.0
#     eye = eye.reshape(1, 24, 24, 1)
#     return eye

# # -------------------------------------------------
# # Draw Bigger Eye Box
# # -------------------------------------------------
# def draw_eye_box(frame, landmarks, eye_points, state="Awake", padding=14):
#     h, w, _ = frame.shape
#     coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]

#     x1 = max(min(p[0] for p in coords) - padding, 0)
#     x2 = min(max(p[0] for p in coords) + padding, w)
#     y1 = max(min(p[1] for p in coords) - padding, 0)
#     y2 = min(max(p[1] for p in coords) + padding, h)

#     color = (0, 0, 255) if state == "Drowsy" else (0, 255, 0)
#     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

# # -------------------------------------------------
# # Yawn Detection using Mouth Aspect Ratio
# # -------------------------------------------------
# def detect_yawn(landmarks, h, w):
#     top    = np.array([landmarks[MOUTH_TOP].x * w,    landmarks[MOUTH_TOP].y * h])
#     bottom = np.array([landmarks[MOUTH_BOTTOM].x * w, landmarks[MOUTH_BOTTOM].y * h])
#     left   = np.array([landmarks[MOUTH_LEFT].x * w,   landmarks[MOUTH_LEFT].y * h])
#     right  = np.array([landmarks[MOUTH_RIGHT].x * w,  landmarks[MOUTH_RIGHT].y * h])

#     mouth_open  = np.linalg.norm(top - bottom)
#     mouth_width = np.linalg.norm(left - right)

#     if mouth_width == 0:
#         return False, 0.0

#     ratio = mouth_open / mouth_width
#     return ratio > YAWN_RATIO_THRESHOLD, round(ratio, 3)

# # -------------------------------------------------
# # Head Tilt Detection
# # -------------------------------------------------
# def detect_head_tilt(landmarks, h, w):
#     left_ear  = np.array([landmarks[LEFT_EAR].x * w,  landmarks[LEFT_EAR].y * h])
#     right_ear = np.array([landmarks[RIGHT_EAR].x * w, landmarks[RIGHT_EAR].y * h])

#     dx = right_ear[0] - left_ear[0]
#     dy = right_ear[1] - left_ear[1]

#     angle = np.degrees(np.arctan2(dy, dx))
#     tilted = abs(angle) > HEAD_TILT_THRESHOLD
#     return tilted, round(angle, 1)

# # -------------------------------------------------
# # Draw HUD Panel (top-left)
# # -------------------------------------------------
# def draw_hud(frame, fps, latency_ms, state, yawn_detected, tilt_detected, tilt_angle):
#     # Semi-transparent black background
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (285, 180), (0, 0, 0), -1)
#     cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

#     # FPS
#     cv2.putText(frame, f"FPS     : {fps}", (10, 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

#     # Latency
#     cv2.putText(frame, f"Latency : {latency_ms} ms", (10, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

#     # Drowsy State
#     state_color = (0, 0, 255) if state == "Drowsy" else (0, 255, 0) if state == "Awake" else (255, 255, 0)
#     cv2.putText(frame, f"State   : {state}", (10, 80),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 1)

#     # Yawn
#     yawn_color = (0, 0, 255) if yawn_detected else (0, 255, 0)
#     yawn_text  = "YES" if yawn_detected else "NO"
#     cv2.putText(frame, f"Yawning : {yawn_text}", (10, 110),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, yawn_color, 1)

#     # Head Tilt
#     tilt_color = (0, 0, 255) if tilt_detected else (0, 255, 0)
#     tilt_text  = f"YES ({tilt_angle}deg)" if tilt_detected else f"NO  ({tilt_angle}deg)"
#     cv2.putText(frame, f"Tilt    : {tilt_text}", (10, 140),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, tilt_color, 1)

#     # Alert
#     if state == "Drowsy" or yawn_detected or tilt_detected:
#         cv2.putText(frame, "!! ALERT !!", (10, 170),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# # -------------------------------------------------
# # Main Loop
# # -------------------------------------------------
# while True:
#     frame_start = time.time()

#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
#     rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)

#     state         = "No Face"
#     yawn_detected = False
#     tilt_detected = False
#     tilt_angle    = 0.0

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             landmarks = face_landmarks.landmark

#             # ✅ Extract eyes BEFORE drawing
#             left_eye  = extract_eye(frame, landmarks, LEFT_EYE)
#             right_eye = extract_eye(frame, landmarks, RIGHT_EYE)

#             # --- Drowsiness Detection ---
#             if left_eye is not None and right_eye is not None:
#                 both  = np.vstack([left_eye, right_eye])
#                 preds = model.predict(both, verbose=0)
#                 prediction = (preds[0][0] + preds[1][0]) / 2

#                 if prediction > 0.5:
#                     drowsy_frame_count += 1
#                     if drowsy_frame_count >= DROWSY_FRAME_THRESHOLD:
#                         state = "Drowsy"
#                         if not mixer.music.get_busy():
#                             mixer.music.play()
#                 else:
#                     drowsy_frame_count = 0
#                     state = "Awake"
#                     mixer.music.stop()

#             # --- Yawn Detection ---
#             yawning, yawn_ratio = detect_yawn(landmarks, h, w)
#             if yawning:
#                 yawn_frame_count += 1
#                 if yawn_frame_count >= YAWN_FRAME_THRESHOLD:
#                     yawn_detected = True
#                     if not mixer.music.get_busy():
#                         mixer.music.play()
#             else:
#                 yawn_frame_count = 0

#             # --- Head Tilt Detection ---
#             tilt_detected, tilt_angle = detect_head_tilt(landmarks, h, w)
#             if tilt_detected:
#                 tilt_frame_count += 1
#                 if tilt_frame_count >= TILT_FRAME_THRESHOLD:
#                     if not mixer.music.get_busy():
#                         mixer.music.play()
#             else:
#                 tilt_frame_count = 0  # reset if head straightens
           
#             # --- Draw bigger eye boxes ---
#             draw_eye_box(frame, landmarks, LEFT_EYE,  state, padding=14)
#             draw_eye_box(frame, landmarks, RIGHT_EYE, state, padding=14)

#     # --- FPS Calculation ---
#     fps_frame_count += 1
#     elapsed = time.time() - fps_start_time
#     if elapsed >= 1.0:
#         fps             = fps_frame_count
#         fps_frame_count = 0
#         fps_start_time  = time.time()

#     # --- Latency Calculation ---
#     latency_ms = int((time.time() - frame_start) * 1000)

#     # --- Draw HUD ---
#     draw_hud(frame, fps, latency_ms, state, yawn_detected, tilt_detected, tilt_angle)

#     cv2.imshow("Driver Drowsiness Detection", frame)
#     # Add this line temporarily after prediction calculation:
#     print(f"Prediction: {prediction:.3f} | Frames: {drowsy_frame_count}")

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
#         break

# cap.release()
# cv2.destroyAllWindows()


















# import cv2
# import numpy as np
# import tensorflow as tf
# import mediapipe as mp
# from pygame import mixer
# import os

# # -------------------------------------------------
# # Base Directory (fixes relative path issues)
# # -------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # -------------------------------------------------
# # Load Alarm Sound
# # -------------------------------------------------
# mixer.init()
# alarm_path = os.path.join(BASE_DIR, "alarm", "alarm.wav")
# mixer.music.load(alarm_path)

# # -------------------------------------------------
# # Load CNN Model
# # -------------------------------------------------
# model = tf.keras.models.load_model(os.path.join(BASE_DIR, "cnn_model", "eye_model.h5"))

# # -------------------------------------------------
# # MediaPipe FaceMesh Setup
# # -------------------------------------------------
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # -------------------------------------------------
# # Webcam
# # -------------------------------------------------
# cap = cv2.VideoCapture(0)

# # -------------------------------------------------
# # Eye Landmark Indexes (MediaPipe)
# # -------------------------------------------------
# LEFT_EYE  = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# # -------------------------------------------------
# # Extract Eye for CNN Prediction
# # -------------------------------------------------
# def extract_eye(frame, landmarks, eye_points, padding=5):
#     h, w, _ = frame.shape
#     coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]

#     x1 = max(min(p[0] for p in coords) - padding, 0)
#     x2 = min(max(p[0] for p in coords) + padding, w)
#     y1 = max(min(p[1] for p in coords) - padding, 0)
#     y2 = min(max(p[1] for p in coords) + padding, h)

#     eye = frame[y1:y2, x1:x2]

#     if eye.size == 0:
#         return None

#     eye = cv2.resize(eye, (24, 24))
#     eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
#     eye = eye / 255.0
#     eye = eye.reshape(1, 24, 24, 1)

#     return eye

# # -------------------------------------------------
# # Draw Box Around Eye (Green = Awake, Red = Drowsy)
# # -------------------------------------------------
# def draw_eye_box(frame, landmarks, eye_points, state="Awake", padding=5):
#     h, w, _ = frame.shape
#     coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]

#     x1 = max(min(p[0] for p in coords) - padding, 0)
#     x2 = min(max(p[0] for p in coords) + padding, w)
#     y1 = max(min(p[1] for p in coords) - padding, 0)
#     y2 = min(max(p[1] for p in coords) + padding, h)

#     color = (0, 0, 255) if state == "Drowsy" else (0, 255, 0)
#     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

# # -------------------------------------------------
# # Main Loop
# # -------------------------------------------------
# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)

#     state = "No Face"

#     if results.multi_face_landmarks:

#         for face_landmarks in results.multi_face_landmarks:

#             landmarks = face_landmarks.landmark

#             # ✅ Extract eyes BEFORE any drawing
#             left_eye  = extract_eye(frame, landmarks, LEFT_EYE)
#             right_eye = extract_eye(frame, landmarks, RIGHT_EYE)

#             if left_eye is not None and right_eye is not None:

#                 # ✅ Batch predict both eyes in one call (faster)
#                 both = np.vstack([left_eye, right_eye])
#                 preds = model.predict(both, verbose=0)
#                 prediction = (preds[0][0] + preds[1][0]) / 2

#                 print("Prediction:", prediction)

#                 if prediction > 0.7:
#                     state = "Drowsy"
#                     if not mixer.music.get_busy():
#                         mixer.music.play()
#                 else:
#                     state = "Awake"
#                     mixer.music.stop()

#             # ✅ Draw clean boxes around eyes (NO dotted mesh)
#             draw_eye_box(frame, landmarks, LEFT_EYE, state)
#             draw_eye_box(frame, landmarks, RIGHT_EYE, state)

#     # ✅ Status text on screen
#     color = (0, 0, 255) if state == "Drowsy" else (0, 255, 0) if state == "Awake" else (255, 255, 0)
#     cv2.putText(frame, state, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#     cv2.imshow("Driver Drowsiness Detection", frame)

#     if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
#         break

# cap.release()
# cv2.destroyAllWindows()
