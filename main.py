import cv2
import mediapipe as mp
import pyautogui
import time
import math
from collections import deque

# ---------------- CONFIG ----------------
pyautogui.FAILSAFE = False
ACTION_DELAY = 0.7
TILT_THRESHOLD = 15

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face.FaceMesh(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# ---------------- ACTION TIMER ----------------
last_action = {
    "LEFT": 0,
    "RIGHT": 0,
    "JUMP": 0,
    "ROLL": 0
}

def can_act(action):
    if time.time() - last_action[action] > ACTION_DELAY:
        last_action[action] = time.time()
        return True
    return False

# ---------------- GESTURE FUNCTIONS ----------------
def is_fist(lm):
    tips = [8, 12, 16, 20]
    return all(lm[tip].y > lm[tip - 2].y for tip in tips)

def is_two_fingers(lm):
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y
    return index_up and middle_up and ring_down and pinky_down

# ---------------- HEAD TILT ----------------
angle_buffer = deque(maxlen=7)

def get_head_tilt(face_lm):
    left_eye = face_lm.landmark[33]
    right_eye = face_lm.landmark[263]
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    angle = math.degrees(math.atan2(dy, dx))
    angle_buffer.append(angle)
    return sum(angle_buffer) / len(angle_buffer)

# ---------------- MAIN LOOP ----------------
print("🎮 Gesture Control Started (ESC to Exit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    # -------- HAND CONTROL --------
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_lm, hand_info in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness
        ):
            lm = hand_lm.landmark
            label = hand_info.classification[0].label

            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            # LEFT / RIGHT movement (fist)
            if is_fist(lm):
                if label == "Right" and can_act("RIGHT"):
                    pyautogui.press("right")
                    cv2.putText(frame, "RIGHT", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                elif label == "Left" and can_act("LEFT"):
                    pyautogui.press("left")
                    cv2.putText(frame, "LEFT", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # JUMP → Right hand two fingers
            elif is_two_fingers(lm) and label == "Right" and can_act("JUMP"):
                pyautogui.press("up")
                cv2.putText(frame, "JUMP", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # ROLL → Left hand two fingers
            elif is_two_fingers(lm) and label == "Left" and can_act("ROLL"):
                pyautogui.press("down")
                cv2.putText(frame, "ROLL", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # -------- HEAD CONTROL --------
    if face_results.multi_face_landmarks:
        for face_lm in face_results.multi_face_landmarks:
            angle = get_head_tilt(face_lm)

            if angle > TILT_THRESHOLD and can_act("RIGHT"):
                pyautogui.press("right")
                cv2.putText(frame, "HEAD RIGHT", (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

            elif angle < -TILT_THRESHOLD and can_act("LEFT"):
                pyautogui.press("left")
                cv2.putText(frame, "HEAD LEFT", (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("Temple Run - Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
