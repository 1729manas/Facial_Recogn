

from flask import Flask, request, jsonify, send_file
import face_recognition_api
import cv2
import os
import pickle
import numpy as np
import base64
import io
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)


fname = 'classifier.pkl'
if os.path.isfile(fname):
    with open(fname, 'rb') as f:
        le, clf = pickle.load(f)
else:
    raise FileNotFoundError(f"Classifier '{fname}' does not exist.")


emotion_classifier = ... 
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.route("/process_and_predict", methods=["POST"])
def process_and_predict():
   
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    
    base64_image = data["image"].split(",")[-1]
    try:
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"error": "Failed to decode image data", "message": str(e)}), 400

    
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    predictions = []
    if len(face_encodings) > 0:
        closest_distances = clf.kneighbors(face_encodings, n_neighbors=1)
        is_recognized = [closest_distances[0][i][0] <= 0.5 for i in range(len(face_locations))]
        predictions = [
            (le.inverse_transform([int(pred)])[0].title(), loc) if rec else ("Unknown", loc)
            for pred, loc, rec in zip(clf.predict(face_encodings), face_locations, is_recognized)
        ]

   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

      
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = emotion_classifier.predict(roi)[0]
            emotion_label = emotion_labels[prediction.argmax()]

            
            label_position = (x, y - 10)
            cv2.putText(image, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

  
    for name, (top, right, bottom, left) in predictions:
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    _, img_encoded = cv2.imencode(".jpg", image)
    img_bytes = img_encoded.tobytes()

    return send_file(io.BytesIO(img_bytes), mimetype="image/jpeg")

if __name__ == '__main__':
    app.run(debug=True)



@app.route('/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
