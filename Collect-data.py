from CodeMediapipe import *
import os
import numpy as np
import mediapipe as mp
import cv2
# Path for the exported data
DATA_PATH = os.path.join('Dataset')
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# words to classify
words = np.array(['Hi', 'How', 'You', 'I', 'Ok'])

# no of videos
videos = 30

# no of frames per video
frames = 30
for word in words:
    for sequence in range(videos):
        try:
            os.makedirs(os.path.join(DATA_PATH, word, str(sequence)))
        except:
            pass

# Capturing dataframes using camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Set mediapipe model to collect keypoints of your video frames
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # The words we have to train
    for word in words:
        # The videos per each word
        for vid in range(videos):
            # The number of frames per videos captured
            for frame_no in range(frames):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                # Apply the collection Logic here
                if frame_no == 0:
                    cv2.putText(image, 'STARTING COLLECTION',
                                (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting the frames for the video number'.format(word, vid),
                                (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                else:
                    cv2.putText(image, 'Collecting the frames for the video number'.format(word, vid),
                                (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen as a selfi image
                cv2.imshow('OpenCV Feed', cv2.flip(image, 1))

                # Extracting keypoints from frames
                keypoints = extract_keypoint(results)
                kp_Path = os.path.join(
                    DATA_PATH, word, str(vid), str(frame_no))
                np.save(kp_Path, keypoints)

                # Quit your code when you want ...
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()
