import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
mp_holistic = mp.solutions.holistic  # Holistic modelq
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


# drawing pipelines on you face, body , left-hand, right-hand
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks,
                              mp_holistic.FACEMESH_TESSELATION)  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(
                                  color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# # Set mediapipe model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():

#         # Read feed
#         ret, frame = cap.read()

#         # Make detections
#         image, results = mediapipe_detection(frame, holistic)
#         print(results)

#         # Draw landmarks
#         draw_styled_landmarks(image, results)

#         # Show to screen as a selfi image
#         cv2.imshow('OpenCV Feed', cv2.flip(image, 1))

#         # Break gracefully
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# # len(results.right_hand_landmarks.landmark)
# # len(results.left_hand_landmarks.landmark)
# # len(results.face_landmarks.landmark)
# # len(results.pose_landmarks.landmark)


# Methods to exract the keypoints
# converting all the landmarks to arrays containing values of x,y,z
def extract_keypoint(results):
    pose = np.array([[r.x, r.y, r.z, r.visibility]
                     for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

    # in case didn't get any keypoints of the hand then will assign an array holding zeros
    l_hand = np.array([[r.x, r.y, r.z]
                       for r in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)

    # l_hand
    # in case didn't get the landmarks of the hands
    r_hand = np.array([[r.x, r.y, r.z]
                       for r in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    # flatten the face landmarks
    face = np.array([[r.x, r.y, r.z]
                     for r in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    # len(pose)
    return np.concatenate([face, pose, l_hand, r_hand])
# face
# len(pose)
# pose.shape
# np.zeros(21*3)
