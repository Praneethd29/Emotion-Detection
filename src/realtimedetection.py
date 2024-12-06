import cv2
import numpy as np
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.saving import register_keras_serializable

# Register Sequential as a serializable object
from tensorflow.keras.models import Sequential

@register_keras_serializable()
class CustomSequential(Sequential):
    pass

try:
    # Load model from JSON
    with open("facialemotionmodel.json", "r") as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json, custom_objects={'Sequential': CustomSequential})
    model.load_weights("facialemotionmodel.h5")
    print("Model loaded from JSON and weights.")
except Exception as e:
    print("Failed to load JSON model, trying .h5 format...")
    model = load_model("facialemotionmodel.h5")
    print("Model loaded from .h5 file.")

# Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Preprocessing function
def extract_features(image):
    feature = np.array(image, dtype=np.float32).reshape(1, 48, 48, 1)
    return feature / 255.0

# Open webcam for real-time emotion detection
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

try:
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error accessing webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (48, 48))
            img = extract_features(face_img)

            # Predict emotion
            pred = model.predict(img, verbose=0)
            prediction_label = labels[pred.argmax()]

            # Display prediction
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    webcam.release()
    cv2.destroyAllWindows()
