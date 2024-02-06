import os
import cv2
import numpy as np
import streamlit as st
from predict import FaceClassifier
import sqlite3
import torch

def resize_embedding(embedding, target_size):
    current_size = len(embedding)
    
    if current_size < target_size:
        # Padding the embedding with zeros to match the target size
        padding_size = target_size - current_size
        embedding = np.concatenate((embedding, np.zeros(padding_size, dtype=np.float32)))

    elif current_size > target_size:
        # Truncate the embedding to match the target size
        embedding = embedding[:target_size]

    return embedding

def detect_phone_and_capture(video_capture, classifier, rows, embedding_size, similarity_threshold):
    st.title("Student Attendance System")

    conf_threshold = 0.3  # Define conf_threshold here

    phone_detected = False

    # Create a button to capture the image
    st.write("Click the button to capture an image.")
    capture_button = st.button("Capture Image")

    if capture_button:
        success, captured_frame = video_capture.read()

        if success:
            # Run object detection
            results = model(captured_frame)

            for det in results.xyxy[0]:
                classId = int(det[5])
                confidence = det[4]

                if confidence > conf_threshold and classId == 67:  # Class 67 is typically associated with phones
                    st.error("Please don't use a cell phone to capture your image.")
                    phone_detected = True  # Set the flag to indicate phone detection
                    break

            if not phone_detected:
                # Save the captured image to the 'application_data/input_image' directory
                input_image_path = 'application_data/input_image/captured_image.jpg'
                cv2.imwrite(input_image_path, captured_frame)

                # Display the captured image
                st.image(captured_frame, channels="BGR", use_column_width=True)

                # Get the embedding of the captured image
                input_embedding = classifier.get_face_embedding(captured_frame)

                # Resize the input embedding to match the expected size
                input_embedding = resize_embedding(input_embedding, embedding_size)

                # Compare similarity with embeddings in the database
                verified_students = []

                for row in rows:
                    if len(row) >= 2:
                        student_name = row[0]
                        binary_data = row[1]
                        data_count = len(binary_data) // np.float32().itemsize
                        database_embedding = np.frombuffer(binary_data, dtype=np.float32, count=data_count)

                        # Resize the database embedding to match the expected size
                        database_embedding = resize_embedding(database_embedding, embedding_size)

                        similarity_score = classifier.are_same_person(database_embedding, input_embedding)

                        if similarity_score > similarity_threshold:
                            verified_students.append(student_name)
                    else:
                        st.write(f"Invalid row format: {row}")

                # Display verification results
                st.header("Verification Results")
                if verified_students:
                    st.write("Verified Students:")
                    for student_name in verified_students:
                        st.write(student_name)
                else:
                    st.write("No students verified.")

                # Release the video capture when the image is captured
                video_capture.release()

        cv2.destroyAllWindows()

def main():
    # Initialize video capture and other shared variables
    video_capture = cv2.VideoCapture(1, cv2.CAP_MSMF)
    classifier = FaceClassifier()
    similarity_threshold = 0.8
    embedding_size = 128  # Set this to the expected size of embeddings in the database

    # Load embeddings from the database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name,embedding FROM students")
    rows = cursor.fetchall()

    detect_phone_and_capture(video_capture, classifier, rows, embedding_size, similarity_threshold)

if __name__ == "__main__":
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)
    model = model.autoshape()  # automatically shape input to the model size

    # Start the Streamlit app
    main()