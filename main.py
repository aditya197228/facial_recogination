from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import datetime
import os
import sqlite3
import pandas as pd
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- Utility Functions ---
def decode_base64_image(base64_string):
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def preprocess_face(img, box):
    x, y, w, h = box
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# Load FaceNet & MTCNN
face_detector = MTCNN()
embedder = FaceNet()  # Auto-downloads model on first use

# File paths
knn_model_path = 'model/knn_model.pkl'
embeddings_path = 'model/embeddings.npy'
labels_path = 'model/labels.npy'
os.makedirs('model', exist_ok=True)

# Load or initialize KNN
if os.path.exists(knn_model_path):
    knn = joblib.load(knn_model_path)
else:
    knn = KNeighborsClassifier(n_neighbors=3)

# --- Register a Student ---
@app.route('/register', methods=['POST'])
def api_register():
    data = request.json
    name = data['name']
    images = data['images']  # List of base64 encoded images

    embeddings = []
    for base64_img in images:
        img = decode_base64_image(base64_img)
        result = face_detector.detect_faces(img)
        if not result:
            continue
        box = result[0]['box']
        face = preprocess_face(img, box)
        embedding = embedder.embeddings(face)[0]
        embeddings.append(embedding)

    if not embeddings:
        return {"status": "Face not detected"}

    # Load existing embeddings and labels
    X = list(np.load(embeddings_path, allow_pickle=True)) if os.path.exists(embeddings_path) else []
    y = list(np.load(labels_path, allow_pickle=True)) if os.path.exists(labels_path) else []

    for emb in embeddings:
        X.append(emb)
        y.append(name)

    knn.fit(X, y)
    joblib.dump(knn, knn_model_path)
    np.save(embeddings_path, X)
    np.save(labels_path, y)

    return {"status": "Registered"}

# --- Recognize a Student ---
@app.route('/recognize', methods=['POST'])
def api_recognize():
    base64_img = request.json['image']
    img = decode_base64_image(base64_img)
    results = face_detector.detect_faces(img)
    if not results:
        return {"name": "Unknown", "timestamp": None}

    box = results[0]['box']
    face = preprocess_face(img, box)
    embedding = embedder.embeddings(face)[0]

    label = knn.predict([embedding])[0]
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    os.makedirs('database', exist_ok=True)
    conn = sqlite3.connect('database/attendance.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            name TEXT,
            timestamp TEXT
        )
    """)
    cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (label, timestamp))
    conn.commit()
    conn.close()

    return {"name": label, "timestamp": timestamp}

# --- Attendance Analytics ---
@app.route('/analytics', methods=['GET'])
def api_analytics():
    conn = sqlite3.connect('database/attendance.db')
    df = pd.read_sql_query("SELECT * FROM attendance", conn)
    if df.empty:
        conn.close()
        return []
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    stats = df.groupby('name').agg({
        'timestamp': ['count', lambda x: x.dt.hour.mean(), lambda x: x.max() - x.min()]
    })
    stats.columns = ['days_present', 'avg_entry_hour', 'presence_span']
    stats = stats.reset_index()
    conn.close()
    return stats.to_dict(orient='records')

# --- Truancy Risk Prediction ---
@app.route('/predict_truancy', methods=['GET'])
def api_truancy():
    conn = sqlite3.connect('database/attendance.db')
    df = pd.read_sql_query("SELECT * FROM attendance", conn)
    conn.close()
    if df.empty:
        return []

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    stats = df.groupby('name').agg({
        'timestamp': ['count', lambda x: x.dt.hour.mean()]
    })
    stats.columns = ['days_present', 'avg_entry_hour']
    stats = stats.reset_index()

    # Assign risk
    stats['risk'] = stats['days_present'].apply(lambda x: 'High' if x < 3 else ('Medium' if x < 5 else 'Low'))

    clf = RandomForestClassifier()
    clf.fit(stats[['days_present', 'avg_entry_hour']], stats['risk'])
    pred = clf.predict(stats[['days_present', 'avg_entry_hour']])
    stats['predicted_risk'] = pred

    return stats[['name', 'predicted_risk']].to_dict(orient='records')

# --- Root and Favicon Handlers ---
@app.route('/', methods=['GET'])
def index():
    return "✅ Face Recognition Attendance API is running."

@app.route('/favicon.ico')
def favicon():
    return '', 204

# --- Run Flask App ---
if __name__ == '__main__':
    app.run(debug=True)
