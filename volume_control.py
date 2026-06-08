import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
drawing_utils = mp.tasks.vision.drawing_utils

THUMB_TIP = 4
THUMB_IP = 3
THUMB_MCP = 2

STEP = 0.05
COOLDOWN = 10


def get_volume_interface():
    return AudioUtilities.GetSpeakers().EndpointVolume


def is_thumbs_up(hand_landmarks):
    return hand_landmarks[THUMB_TIP].y < hand_landmarks[THUMB_IP].y


def is_thumbs_down(hand_landmarks):
    return hand_landmarks[THUMB_TIP].y > hand_landmarks[THUMB_MCP].y


def main():
    volume = get_volume_interface()
    current_vol = volume.GetMasterVolumeLevelScalar()

    model_path = "hand_landmarker.task"
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    frame_idx = 0
    cooldown = 0

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

            if cooldown > 0:
                cooldown -= 1

            if result.hand_landmarks and cooldown == 0:
                hand = result.hand_landmarks[0]

                if is_thumbs_up(hand):
                    current_vol = min(1.0, current_vol + STEP)
                    volume.SetMasterVolumeLevelScalar(current_vol, None)
                    cooldown = COOLDOWN
                    label = "VOL +"
                elif is_thumbs_down(hand):
                    current_vol = max(0.0, current_vol - STEP)
                    volume.SetMasterVolumeLevelScalar(current_vol, None)
                    cooldown = COOLDOWN
                    label = "VOL -"
                else:
                    label = ""

                if label:
                    drawing_utils.draw_landmarks(frame, hand)
                    cv2.putText(
                        frame,
                        label,
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        4,
                    )

            vol_percent = int(current_vol * 100)
            cv2.putText(
                frame,
                f"Volume: {vol_percent}%",
                (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Volume Control", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
