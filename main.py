import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import speech_recognition as sr
import threading

# ================= VOLUME CONTROL (WINDOWS SAFE) =================
def volume_up():
    pyautogui.press("volumeup")

def volume_down():
    pyautogui.press("volumedown")

# ================= VOICE CONTROL THREAD =================
def voice_control():
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        r.adjust_for_ambient_noise(source)

    while True:
        try:
            with mic as source:
                print("🎤 Listening...")
                audio = r.listen(source, phrase_time_limit=3)

            command = r.recognize_google(audio).lower()
            print("Voice:", command)

            if "volume up" in command:
                volume_up()

            elif "volume down" in command:
                volume_down()

        except sr.UnknownValueError:
            pass
        except Exception as e:
            print("Voice error:", e)

threading.Thread(target=voice_control, daemon=True).start()

# ================= HAND TRACKING =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
smoothening = 3
speed_multiplier = 1.7
tip_ids = [4, 8, 12, 16, 20]

# ================= MAIN LOOP =================
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = img.shape
            lm_list = [(int(lm.x * w), int(lm.y * h))
                       for lm in hand_landmarks.landmark]

            if lm_list:
                # Mouse movement
                x1, y1 = lm_list[8]   # Index
                x0, y0 = lm_list[4]   # Thumb

                screen_x = np.interp(x1, (0, w), (0, screen_w))
                screen_y = np.interp(y1, (0, h), (0, screen_h))

                curr_x = prev_x + (screen_x - prev_x) / smoothening * speed_multiplier
                curr_y = prev_y + (screen_y - prev_y) / smoothening * speed_multiplier

                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Click
                if abs(x1 - x0) < 40 and abs(y1 - y0) < 40:
                    pyautogui.click()
                    pyautogui.sleep(0.25)

                # Finger detection
                fingers = []
                for i in range(1, 5):
                    fingers.append(
                        lm_list[tip_ids[i]][1] <
                        lm_list[tip_ids[i] - 2][1]
                    )

                # Scroll up (index + middle)
                if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
                    pyautogui.scroll(60)

                # Scroll down (index only)
                if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3]:
                    pyautogui.scroll(-60)

    cv2.imshow("Virtual Mouse + Voice Volume", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
