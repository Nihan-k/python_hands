import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
drawing_utils = mp.tasks.vision.drawing_utils

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
THUMB_IP = 3
THUMB_TIP = 4


def count_fingers(hand_landmarks, handedness):
    fingers = []
    for i in range(1, 5):
        if hand_landmarks[FINGER_TIPS[i]].y < hand_landmarks[FINGER_PIPS[i]].y:
            fingers.append(1)
        else:
            fingers.append(0)

    if handedness == "Right":
        if hand_landmarks[THUMB_TIP].x < hand_landmarks[THUMB_IP].x:
            fingers.insert(0, 1)
        else:
            fingers.insert(0, 0)
    else:
        if hand_landmarks[THUMB_TIP].x > hand_landmarks[THUMB_IP].x:
            fingers.insert(0, 1)
        else:
            fingers.insert(0, 0)

    return sum(fingers)


def main():
    model_path = "hand_landmarker.task"
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    frame_idx = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_idx += 1
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            timestamp = int(frame_idx * (1000 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp)

            if result.hand_landmarks:
                for idx, hand_landmarks in enumerate(result.hand_landmarks):
                    handedness = result.handedness[idx][0].category_name
                    count = count_fingers(hand_landmarks, handedness)

                    h, w, _ = frame.shape
                    x = int(hand_landmarks[0].x * w)
                    y = int(hand_landmarks[0].y * h)

                    drawing_utils.draw_landmarks(frame, hand_landmarks)

                    cv2.putText(
                        frame,
                        f"Fingers: {count}",
                        (x - 40, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3,
                    )

            cv2.imshow("Finger Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
