import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to capture frame")
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            landmark_list = []

            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

            print(landmark_list)

    cv2.imshow("Collect Data", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()