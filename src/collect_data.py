import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)

# CSV file path
csv_file = "dataset/gestures.csv"

# Create CSV file with header if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["label"]

        for i in range(21):
            header.append(f"x{i+1}")
            header.append(f"y{i+1}")

        writer.writerow(header)

# Gesture labels
label_map = {
    ord('y'): "YES",
    ord('n'): "NO",
    ord('s'): "STOP",
    ord('o'): "OK",
    ord('h'): "HELLO"
}

print("\n===== DATA COLLECTION STARTED =====")
print("Press keys to save gestures:")
print("y -> YES")
print("n -> NO")
print("s -> STOP")
print("o -> OK")
print("h -> HELLO")
print("ESC -> Exit")
print("===================================\n")

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to capture frame")
        continue

    # Flip image for natural view
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    landmarks = None

    if result.multi_hand_landmarks:

        cv2.putText(
            frame,
            "HAND DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        for hand_landmarks in result.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

    else:

        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.putText(
        frame,
        "Y:YES N:NO S:STOP O:OK H:HELLO",
        (20, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Collect Gesture Data", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save data when key is pressed
    if key in label_map:

        if landmarks is not None:

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([label_map[key]] + landmarks)

            print(f"Saved: {label_map[key]}")

        else:
            print("No hand detected. Sample not saved.")

    elif key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()