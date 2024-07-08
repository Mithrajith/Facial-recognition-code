import cv2
import numpy as np
import os
import telepot

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Telegram bot configuration
bot_token = '6962362747:AAHqrgsCOrP7loBlMBeUPb3MA8ML3Bvi_18'
chat_id = 5807052979
bot = telepot.Bot(bot_token)

# Load known faces and train the recognizer
known_faces_dir = 'D:\my project\IOT Project\known_faces_dir'
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Preparing training data
faces = []
labels = []

def prepare_training_data():
    label_id = 0
    for filename in os.listdir(known_faces_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(known_faces_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            face = face_detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in face:
                faces.append(image[y:y + h, x:x + w])
                labels.append(label_id)
        label_id += 1

    face_recognizer.train(faces, np.array(labels))

prepare_training_data()

# Function to send alert
def send_alert(video_path):
    with open(video_path, 'rb') as video_file:
        bot.sendVideo(chat_id, video=video_file)
    bot.sendMessage(chat_id, "Unknown person detected!")

# Recognize faces in the webcam feed
while True:
    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(face_roi)

        if confidence < 50:  # You can adjust the confidence threshold
            name = "Known"
        else:
            name = "Unknown"
            # Save the video with the unknown person
            video_path = 'unknown.mp4'
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame.shape[1], frame.shape[0]))
            for _ in range(5 * 20):  # Save 5 seconds of video
                video_writer.write(frame)
                ret, frame = video_capture.read()
            video_writer.release()
            # Send alert
            send_alert(video_path)

        # Draw a box around the face and label it
        color = (0, 255, 0) if name == "Known" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

# Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()