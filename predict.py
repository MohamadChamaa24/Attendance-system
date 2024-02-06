import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN, extract_face
import sqlite3

class FaceClassifier:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def get_face_embedding(self, img_path_or_array):
        # Check if img_path_or_array is a string (file path) or a numpy array
        if isinstance(img_path_or_array, str):
            img_array = cv2.imread(img_path_or_array)
        elif isinstance(img_path_or_array, np.ndarray):
            img_array = img_path_or_array
        else:
            raise ValueError("Invalid input: img_path_or_array should be a string (file path) or a numpy array.")

        img = self.preprocess_image_array(img_array)

        if img is not None:
            img = torch.tensor(img.transpose(0, 3, 1, 2), dtype=torch.float).to(self.device)
            embeddings = self.embedder(img)
            return embeddings[0].detach().cpu().numpy()
        else:
            return None  # No face detected in the image.

    def preprocess_image_array(self, img_array):
        try:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            boxes, _ = self.mtcnn.detect(img_rgb)

            if boxes is not None:
                faces = extract_face(img_rgb, boxes[0])

                if faces is not None:
                    faces = cv2.resize(faces.permute(1, 2, 0).numpy(), (160, 160))
                    faces = faces / 255.0
                    faces = np.expand_dims(faces, axis=0)
                    return faces
                else:
                    return None  # No valid face extracted
            else:
                return None  # No face detected in the image.
        except Exception as e:
            raise RuntimeError(f"Error during image preprocessing: {e}")

    def are_same_person(self, embedding1, embedding2, threshold=0.8):
        if embedding1 is None or embedding2 is None:
            # Handle the case where either embedding is None
            return 0.0  # or any other appropriate value
        # Normalize the embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)

        # Calculate cosine similarity
        similarity_score = np.dot(embedding1, embedding2)

        return similarity_score > threshold

if __name__ == '__main__':
    database_path = 'database.db'

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding FROM students")
    rows = cursor.fetchall()
    conn.close()

    input_image_path = 'application_data/input_image/captured_image.jpg'  # Change this to the correct path
    input_embedding = FaceClassifier().get_face_embedding(input_image_path)

    if input_embedding is not None:
        classifier = FaceClassifier()
        print(f"Size of input_embedding: {input_embedding.shape}")
        # Set a similarity threshold for verification
        similarity_threshold = 0.8

        # Loop over all rows in the database
        for row in rows:
            if len(row) >= 2:
                
                database_embedding = np.frombuffer(sqlite3.Binary(row[1]), dtype=np.float32)

                # Calculate similarity
                similarity_score = classifier.are_same_person(database_embedding, input_embedding)

                # Check if similarity score is above the threshold
                if similarity_score > similarity_threshold:
                    print(f"Student verified.")
                    break  # Break the loop after the first verification
            else:
                print(f"Invalid row format: {row}")
        else:
            # This block will be executed if the loop completes without a break
            print("Captured image not verified for any student.")
    else:
        print("Face not detected in the input image.")
