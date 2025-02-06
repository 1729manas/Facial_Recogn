import base64
from flask import Flask, Response, jsonify, render_template, request, send_file
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import io
from PIL import Image
import os
import pickle
import face_recognition_api


app = Flask(__name__)


face_classifier = cv2.CascadeClassifier(r"../HaarcascadeclassifierCascadeClassifier.xml")
classifier = load_model(r"../model.h5")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

fname = 'classifier.pkl'
if os.path.isfile(fname):
    with open(fname, 'rb') as f:
        le, clf = pickle.load(f)
else:
    raise FileNotFoundError(f"Classifier '{fname}' does not exist")


@app.route("/predict_emotion", methods=["POST"])
def predict_emotion():
    # Retrieve base64 image data from the request JSON
    data = request.get_json()

    if "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    # Remove the prefix if it exists (for safety)
    base64_image = data["image"].split(",")[
        -1
    ] 

    # Decode the base64 image data
    try:
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"error": "Failed to decode image data", "message": str(e)}), 400

    # Convert to grayscale and perform face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    # Process each detected face
    for x, y, w, h in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Prepare the face region for emotion prediction
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Make prediction
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]

            # Put the emotion label text above the rectangle
            label_position = (x, y - 10)
            cv2.putText(
                image,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )


    _, img_encoded = cv2.imencode(".jpg", image)
    img_bytes = img_encoded.tobytes()




    return send_file(io.BytesIO(img_bytes), mimetype="image/jpeg")


@app.route('/recognize',methods=['POST'])
def recongize():

    data = request.get_json()
    base64_image = data["image"].split(",")[
        -1
    ]  
    try:
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"error": "Failed to decode image data", "message": str(e)}), 400

    frame = image


    small_frame = image

    face_locations = face_recognition_api.face_locations(small_frame)
    face_encodings = face_recognition_api.face_encodings(small_frame, face_locations)
    face_names = []
    predictions = []
    if len(face_encodings) > 0:
        closest_distances = clf.kneighbors(face_encodings, n_neighbors=1)
        is_recognized = [closest_distances[0][i][0] <= 0.5 for i in range(len(face_locations))]

        predictions = [(le.inverse_transform([int(pred)])[0].title(), loc) if rec else ("Unknown", loc) for pred, loc, rec in zip(clf.predict(face_encodings), face_locations, is_recognized)]

    print(predictions)
    for name, (top, right, bottom, left) in predictions:
        print(name, top, right, bottom, left)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 2, bottom - 6), font, 0.5, (255, 255, 255), 1)
    

    _, img_encoded = cv2.imencode(".jpg", frame)
    img_bytes = img_encoded.tobytes()


    return send_file(io.BytesIO(img_bytes), mimetype="image/jpeg")


@app.route('/emotion_and_face', methods = ['POST'])
def emotion_and_face():
    data = request.get_json()
    base64_image = data["image"].split(",")[-1]

    try:
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"error": "Failed to decode image data", "message": str(e)}), 400


    small_frame = image

    face_locations = face_recognition_api.face_locations(small_frame)
    face_encodings = face_recognition_api.face_encodings(small_frame, face_locations)
    face_predictions = []
    if len(face_encodings) > 0:
        closest_distances = clf.kneighbors(face_encodings, n_neighbors=1)
        is_recognized = [closest_distances[0][i][0] <= 0.5 for i in range(len(face_locations))]

        face_predictions = [(le.inverse_transform([int(pred)])[0].title(), loc) if rec else ("Unknown", loc) for pred, loc, rec in zip(clf.predict(face_encodings), face_locations, is_recognized)]

    for name, (top, right, bottom, left) in face_predictions:
        pass




    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

   
    for x, y, w, h in faces:
       
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

      
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

        
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]


            label_position = (x, y - 10)
            cv2.putText(
                image,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    for name, (top, right, bottom, left) in face_predictions:
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 2, bottom - 6), font, 0.5, (255, 255, 255), 1)

    _, img_encoded = cv2.imencode(".jpg", image)
    img_bytes = img_encoded.tobytes()


    return send_file(io.BytesIO(img_bytes), mimetype="image/jpeg") 





@app.route('/')
def home():
    return render_template('index.html')

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)