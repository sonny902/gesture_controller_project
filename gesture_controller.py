import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.85)

cap = cv2.VideoCapture(0)
print('Show gestures — press Q to quit')

def fingers_up(landmarks, handedness):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    # Thumb
    if handedness == 'Right':
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)
    # Other fingers
    for tip in tips[1:]:
        fingers.append(1 if landmarks[tip].y < landmarks[tip-2].y else 0)
    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            lm = hand_landmarks.landmark
            hand_label = handedness.classification[0].label
            f = fingers_up(lm, hand_label)
            count = sum(f)
            
            if f == [0,0,0,0,0]:
                gesture = "FIST"
            elif f == [0,1,0,0,0]:
                gesture = "INDEX FINGER"
            elif f == [0,1,1,0,0]:
                gesture = "TWO FINGERS"
            elif f == [0,1,1,1,0]:
                gesture = "THREE FINGERS"
            elif f == [0,1,1,1,1]:
                gesture = "FOUR FINGERS"
            elif f == [1,1,1,1,1]:
                gesture = "OPEN PALM"
            elif f == [1,0,0,0,0]:
                gesture = "THUMBS UP"
            else:
                gesture = f"OTHER {f}"
            
            print(f"Gesture: {gesture}")
            cv2.putText(frame, gesture, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow('Gesture Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()