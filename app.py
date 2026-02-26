import os
import numpy as np
import cv2
import dlib
import pandas as pd
import tensorflow as tf
import ast  # For safely converting strings to lists
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_very_secret_key_here'  # Required for session management
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU if needed

# Load Pretrained Model and Labels
model = load_model("models/model.h5", compile=False)
labels = np.load("models/labels.npy")

# Load Dlib's Face Detector & Landmark Predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Load Music Recommendations from CSV
music_df = pd.read_csv("music_recommendations.csv")

# Database initialization
def init_db():
    conn = sqlite3.connect('users.db', timeout=10,  check_same_thread=False)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        role TEXT NOT NULL
    )
    ''')
    
    # Create default admin user if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if cursor.fetchone() is None:
        admin_password = generate_password_hash("admin123")
        cursor.execute("INSERT INTO users (username, password, email, role) VALUES (?, ?, ?, ?)",
                     ("admin", admin_password, "admin@example.com", "admin"))
    
    conn.commit()
    conn.close()

# Call database initialization
init_db()

# Authentication decorators
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

def get_music_recommendation(emotion):
    """Fetch recommended music based on detected emotion from CSV."""
    # Convert 'seeds' column to lists safely
    music_df["seeds"] = music_df["seeds"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
    )

    # Filter songs where the detected emotion is in the 'seeds' list
    filtered = music_df[music_df["seeds"].apply(lambda seeds: emotion.lower() in [s.lower() for s in seeds])]

    if not filtered.empty:
        return filtered["lastfm_url"].tolist()  # Return a list of matching URLs
    else:
        return ["No recommendation found for this emotion."]

def process_landmarks(landmarks, reference_point=None):
    """Convert landmarks into a relative coordinate vector."""
    lst = []
    for lm in landmarks:
        lst.append(lm[0] - reference_point[0] if reference_point else lm[0])
        lst.append(lm[1] - reference_point[1] if reference_point else lm[1])
    return lst

def predict_emotion(image):
    """Detect face landmarks and predict emotion from image array."""
    if image is None:
        return "Error: Image not found."

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return "No face detected."

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = [(p.x, p.y) for p in landmarks.parts()]
        face_landmarks = process_landmarks(landmarks, landmarks[30])  # Reference: Nose tip (index 30)

        feature_vector = np.array(face_landmarks, dtype=np.float32)
        feature_vector = np.pad(feature_vector, (0, max(0, 1020 - len(feature_vector))), mode='constant')[:1020]

        feature_vector = feature_vector.reshape(1, -1)
        pred = model.predict(feature_vector)
        emotion_label = labels[np.argmax(pred)]

        return emotion_label

def predict_emotion_from_file(image_path):
    """Detect face landmarks and predict emotion from file path."""
    image = cv2.imread(image_path)
    return predict_emotion(image)

# Authentication Routes
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
        role = 'user'  # Default role is user
        
        # Check if it's the first user, make them admin
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            role = 'admin'
        
        # Check if username or email already exists
        cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        if cursor.fetchone():
            conn.close()
            flash('Username or email already exists', 'danger')
            return redirect(url_for('register'))
        
        # Hash password and save user
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

# Admin Routes
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
    # Don't allow admins to delete themselves
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

            # GET request
            cursor.execute("SELECT id, username, email, role FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()

            if not user:
                flash('User not found.', 'danger')
                return redirect(url_for('admin_dashboard'))

            return render_template('edit_user.html', user=user)

    except sqlite3.OperationalError as e:
        flash('Database error: ' + str(e), 'danger')
        return redirect(url_for('admin_dashboard'))

# Main Application Routes
@app.route('/')
def index():
    # Redirect to login if not authenticated
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

    music_links = get_music_recommendation(detected_emotion)  # Returns a list of music links

    return render_template('index.html', detected_emotion=detected_emotion, music_links=music_links, username=session.get('username'))

@app.route('/webcam')
@login_required
def webcam_page():
    """Render the webcam capture page."""
    return render_template('webcam.html', username=session.get('username'))

@app.route('/capture', methods=['POST'])
@login_required
def capture():
    """Process the captured image from webcam."""
    image_data = request.form['image_data']
    # Remove the prefix 'data:image/jpeg;base64,'
    image_data = image_data.split(',')[1]
    
    # Convert base64 to image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    
    # Convert PIL Image to OpenCV format
    open_cv_image = np.array(image) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR
    
    # Process image for emotion detection
    detected_emotion = predict_emotion(open_cv_image)
    
    if "Error" in detected_emotion or "No face" in detected_emotion:
        return {"error": detected_emotion}
    
    # Get music recommendations
    music_links = get_music_recommendation(detected_emotion)
    
    return {"emotion": detected_emotion, "music_links": music_links}

# User profile routes
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
    
    # Check if username or email is already taken by another user
    cursor.execute("SELECT id FROM users WHERE (username = ? OR email = ?) AND id != ?", 
                 (username, email, session.get('user_id')))
    if cursor.fetchone():
        conn.close()
        flash('Username or email already in use', 'danger')
        return redirect(url_for('profile'))
    
    # Update user information
    cursor.execute("UPDATE users SET username = ?, email = ? WHERE id = ?",
                 (username, email, session.get('user_id')))
    
    # Update password if provided
    if request.form['password'] and request.form['password'].strip():
        password = generate_password_hash(request.form['password'])
        cursor.execute("UPDATE users SET password = ? WHERE id = ?", 
                     (password, session.get('user_id')))
    
    conn.commit()
    conn.close()
    
    # Update session with new username
    session['username'] = username
    
    flash('Profile updated successfully', 'success')
    return redirect(url_for('profile'))

# Live Video Feed Route
def generate_frames():
    """Generate frames from webcam."""
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
@login_required
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Make sure uploads directory exists
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True, use_reloader=False)