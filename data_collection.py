import mediapipe as mp
import numpy as np
import cv2

# Establishing connection to the webcam camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Entering the file name for collected data
name = input("Enter the name of the data: ")

# Using holistic solution from Media Pipe library
holistic = mp.solutions.holistic
hands = mp.solutions.hands
hol = holistic.Holistic()
drawing = mp.solutions.drawing_utils

row_collection = []
data_size = 0

while True:
    lst = []  # List for storing all the landmarks as numpy array

    _, frm = cap.read()

    # Flipping the frame from left to right
    frm = cv2.flip(frm, 1)

    # Converting color from BGR to RGB
    res = hol.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Storing Landmark data
    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

    def extract_hand_landmarks(hand_landmarks):
        if hand_landmarks:
            for i in hand_landmarks.landmark:
                lst.append(i.x - hand_landmarks.landmark[8].x)
                lst.append(i.y - hand_landmarks.landmark[8].y)
        else:
            for _ in range(42):
                lst.append(0.0)

    extract_hand_landmarks(res.left_hand_landmarks)
    extract_hand_landmarks(res.right_hand_landmarks)

    row_collection.append(lst)
    data_size += 1

    # Drawing landmarks on the frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Show image back to screen
    cv2.imshow("window", frm)

    if cv2.waitKey(1) == 27 or data_size > 99:
        cv2.destroyAllWindows()  # Close the image show frame
        cap.release()  # Releasing WebCam
        break

# Optimizing the data collected
np.save(rf"C:\Users\vinay\Desktop\music_rec_frontend\{name}.npy", np.array(row_collection))
print(np.array(row_collection).shape)
