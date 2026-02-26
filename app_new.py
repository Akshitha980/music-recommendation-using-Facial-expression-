import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import ast
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import mediapipe as mp
import random

app = Flask(__name__)
app.secret_key = 'your_very_secret_key_here'  # Required for session management
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU if needed
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable OneDNN optimization for consistent behavior

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Load pretrained model and labels (update paths as necessary)
model = tf.keras.models.load_model("models/model.h5", compile=False)
labels = np.load("models/labels.npy")

# Initialize Mediapipe holistic model once globally
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load Music Recommendations from CSV
music_df = pd.read_csv("music_recommendations.csv")

# ---------- Database initialization remains unchanged ----------

def init_db():
    conn = sqlite3.connect('users.db', timeout=10,  check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        role TEXT NOT NULL
    )
    ''')
    
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if cursor.fetchone() is None:
        admin_password = generate_password_hash("admin123")
        cursor.execute("INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)",
                     ("admin", admin_password, "admin@example.com", "admin"))
    
    conn.commit()
    conn.close()

init_db()

# ---------- Decorators remain unchanged ----------

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        elif session.get('role') != 'admin':
            flash('You need admin privileges to access this page', 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

# ---------- Music Recommendation function unchanged ----------

def get_music_recommendation(emotion):
    music_df["seeds"] = music_df["seeds"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
    )
    filtered = music_df[music_df["seeds"].apply(lambda seeds: emotion.lower() in [s.lower() for s in seeds])]
    if not filtered.empty:
        return filtered["lastfm_url"].tolist()
    else:
        return ["No recommendation found for this emotion."]

# ---------- New landmarks processing functions ----------

def normalize_landmarks(landmarks):
    if not landmarks:
        return [0.0] * 42  # 21 landmarks × 2
    xs = [landmark.x for landmark in landmarks.landmark]
    ys = [landmark.y for landmark in landmarks.landmark]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = xmax - xmin if (xmax - xmin) != 0 else 1.0
    height = ymax - ymin if (ymax - ymin) != 0 else 1.0
    
    normalized = []
    for landmark in landmarks.landmark:
        normalized.append((landmark.x - xmin) / width)
        normalized.append((landmark.y - ymin) / height)
    return normalized

def process_landmarks(landmarks, reference_landmark=None):
    if landmarks:
        if reference_landmark:
            lst = []
            for lm in landmarks.landmark:
                lst.append(lm.x - reference_landmark.x)
                lst.append(lm.y - reference_landmark.y)
            return lst
        else:
            return normalize_landmarks(landmarks)
    else:
        return [0.0] * 42

def prepare_input_features(face_landmarks, left_hand_landmarks, right_hand_landmarks):
    face_features = process_landmarks(face_landmarks, 
                                     face_landmarks.landmark[1] if face_landmarks else None)
    left_hand_features = process_landmarks(left_hand_landmarks,
                                          left_hand_landmarks.landmark[8] if left_hand_landmarks else None)
    right_hand_features = process_landmarks(right_hand_landmarks,
                                           right_hand_landmarks.landmark[8] if right_hand_landmarks else None)
    combined_features = face_features + left_hand_features + right_hand_features
    
    if len(combined_features) < 1020:
        combined_features = np.pad(combined_features, (0, 1020 - len(combined_features)), 'constant')
    elif len(combined_features) > 1020:
        combined_features = combined_features[:1020]
        
    return np.array(combined_features).reshape(1, 1020).astype(np.float32)

# ---------- Replace emotion prediction functions with Mediapipe version ----------

def predict_emotion(image):
    """Predict emotion from a BGR OpenCV image, returns label string or error string."""
    if image is None:
        return "Error: Image not found."
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    
    # If no face landmarks detected, return no face message
    if not results.face_landmarks:
        return "No face detected."
    
    features = prepare_input_features(
        results.face_landmarks,
        results.left_hand_landmarks,
        results.right_hand_landmarks
    )
    
    pred = model.predict(features, verbose=0)
    emotion_label = labels[np.argmax(pred)]
    return emotion_label

def predict_emotion_from_file(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Image not found."
    return predict_emotion(image)

# ---------- The rest of your routes remain unchanged ----------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, password, role FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            session['role'] = user[2]
            
            flash(f'Welcome back, {username}!', 'success')
            if user[2] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        role = 'user'
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            role = 'admin'
        
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        if cursor.fetchone():
            conn.close()
            flash('Username or email already exists', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)",
                     (username, hashed_password, email, role))
        conn.commit()
        conn.close()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/admin')
@admin_required
def admin_dashboard():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, role FROM users")
    users = cursor.fetchall()
    conn.close()
    
    return render_template('admin_dashboard.html', users=users)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    if user_id == session.get('user_id'):
        flash('You cannot delete your own account while logged in', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    
    flash('User has been deleted', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/edit_user/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def edit_user(user_id):
    try:
        with sqlite3.connect('users.db', timeout=10, check_same_thread=False) as conn:
            cursor = conn.cursor()
            if request.method == 'POST':
                username = request.form['username']
                email = request.form['email']
                role = request.form['role']

                cursor.execute("""
                    UPDATE users SET username = ?, email = ?, role = ? WHERE id = ?
                """, (username, email, role, user_id))

                conn.commit()
                flash('User updated successfully!', 'success')
                return redirect(url_for('admin_dashboard'))

            cursor.execute("SELECT id, username, email, role FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()

            if not user:
                flash('User not found.', 'danger')
                return redirect(url_for('admin_dashboard'))

            return render_template('edit_user.html', user=user)

    except sqlite3.OperationalError as e:
        flash('Database error: ' + str(e), 'danger')
        return redirect(url_for('admin_dashboard'))

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('home'))

@app.route('/home')
@login_required
def home():
    return render_template('index.html', username=session.get('username'))

@app.route('/recommend_music', methods=['POST'])
@login_required
def handle_music_request():
    if 'file' not in request.files:
        return render_template('index.html', error="No file selected", username=session.get('username'))

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected", username=session.get('username'))

    file_path = os.path.join("static/uploads", file.filename)
    os.makedirs("static/uploads", exist_ok=True)
    file.save(file_path)

    detected_emotion = predict_emotion_from_file(file_path)

    if "Error" in detected_emotion or "No face" in detected_emotion:
        return render_template('index.html', error=detected_emotion, username=session.get('username'))
    


    print(detected_emotion)

    music_links = get_music_recommendation(detected_emotion)
    if detected_emotion == "surprise":
        music_links = get_music_recommendation("lush")


    return render_template('index.html', detected_emotion=detected_emotion, music_links=music_links, username=session.get('username'))

@app.route('/webcam')
@login_required
def webcam_page():
    return render_template('webcam.html', username=session.get('username'))

@app.route('/capture', methods=['POST'])
@login_required
def capture():
    image_data = request.form['image_data']
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    detected_emotion = predict_emotion(open_cv_image)

    if "Error" in detected_emotion or "No face" in detected_emotion:
        return {"error": detected_emotion}

    music_links = get_music_recommendation(detected_emotion)

    return {"emotion": detected_emotion, "music_links": music_links}

@app.route('/profile')
@login_required
def profile():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT username, email FROM users WHERE id = ?", (session.get('user_id'),))
    user = cursor.fetchone()
    conn.close()
    
    return render_template('profile.html', username=user[0], email=user[1])

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    username = request.form['username']
    email = request.form['email']
    
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE (username = ? OR email = ?) AND id != ?", 
                 (username, email, session.get('user_id')))
    if cursor.fetchone():
        conn.close()
        flash('Username or email already in use', 'danger')
        return redirect(url_for('profile'))
    
    cursor.execute("UPDATE users SET username = ?, email = ? WHERE id = ?",
                 (username, email, session.get('user_id')))
    
    if request.form['password'] and request.form['password'].strip():
        password = generate_password_hash(request.form['password'])
        cursor.execute("UPDATE users SET password = ? WHERE id = ?", 
                     (password, session.get('user_id')))
    
    conn.commit()
    conn.close()
    
    session['username'] = username
    
    flash('Profile updated successfully', 'success')
    return redirect(url_for('profile'))

# Webcam video feed (unchanged)
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True, use_reloader=False)