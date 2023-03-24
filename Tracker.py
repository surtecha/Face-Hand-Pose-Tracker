import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Face landmarks
        mp_draw.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                               mp_draw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                               mp_draw.DrawingSpec(color=(80, 255, 121), thickness=1, circle_radius=1)
                               )

        #Right hand landmarks
        mp_draw.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(240, 15, 0), thickness=2, circle_radius=4),
                               mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                              )

        #Left hand landmarks
        mp_draw.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(15, 255, 0), thickness=2, circle_radius=4),
                               mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                               )

        #Pose landmarks
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()