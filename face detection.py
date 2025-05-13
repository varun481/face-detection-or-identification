import cv2
import face_recognition
import numpy as np

# Load image (your face)
known_image = face_recognition.load_image_file("VARUNIMAGE.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

#pre-trained cascade classifier for eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# give accessthe webcam
cap = cv2.VideoCapture(0)

# Check webcam is opened or not
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frames
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (face_location, face_encoding) in zip(face_locations, face_encodings):
        # Compare face with the known face
        matches = face_recognition.compare_faces([known_face_encoding], face_encoding)

        if matches[0]:
            # lable the rectangle shape that face detected
            (top, right, bottom, left) = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, "This is You", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Region of interest for eyes within the detected face
            roi_gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            roi_color = frame[top:bottom, left:right]

            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30), maxSize=(100, 100))

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    
    cv2.imshow('Face and Eye Recognition', frame)

    # if exit  'esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

#webcam and close windows
cap.release()
cv2.destroyAllWindows()

