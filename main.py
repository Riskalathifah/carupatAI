from flask import Flask, request, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import pickle
import cv2
import dlib
import mediapipe as mp
import math
import os

app = Flask(__name__)

with open ('modelnew.pkl','rb') as file:
    model = pickle.load(file)

def logic(actualSize, gender, size):
    if gender == 'male':
        if size == 'sni':
            if actualSize < 30:
                return 'Maaf size Anda belum tersedia'
            elif 30 <= actualSize < 34:
                return 'XXS'
            elif 34 <= actualSize < 36:
                return 'XS'
            elif 36 <= actualSize < 38:
                return 'S'
            elif 38 <= actualSize < 42:
                return 'M'
            elif 42 <= actualSize < 47:
                return 'L'
            elif 47 <= actualSize < 54:
                return 'XL'
            elif 54 <= actualSize <= 58:
                return 'XXL'
            elif actualSize > 58:
                return 'Maaf size Anda belum tersedia'
        elif size == 'us':
            if actualSize < 30:
                return 'Maaf size Anda belum tersedia'
            elif 30 <= actualSize < 32:
                return 'XXS'
            elif 32 <= actualSize < 34:
                return 'XS'
            elif 34 <= actualSize < 38:
                return 'S'
            elif 38 <= actualSize < 40:
                return 'M'
            elif 40 <= actualSize < 46:
                return 'L'
            elif 46 <= actualSize < 48:
                return 'XL'
            elif 48 <= actualSize <= 50:
                return 'XXL'
            elif actualSize > 50:
                return 'Maaf size Anda belum tersedia'
        elif size == 'eu':
            if actualSize < 40:
                return 'Maaf size Anda belum tersedia'
            elif 40 <= actualSize < 42:
                return 'XXS'
            elif 42 <= actualSize < 44:
                return 'XS'
            elif 44 <= actualSize < 48:
                return 'S'
            elif 40 <= actualSize < 44:
                return 'M'
            elif 44 <= actualSize < 48:
                return 'L'
            elif 48 <= actualSize < 50:
                return 'XL'
            elif 50 <= actualSize <= 52:
                return 'XXL'
            elif actualSize > 52:
                return 'Maaf size Anda belum tersedia'
    if gender == 'female':
        if size == 'sni':
            if actualSize < 30:
                return 'Maaf size Anda belum tersedia'
            elif 30 <= actualSize < 33:
                return 'XXS'
            elif 33 <= actualSize < 38:
                return 'XS'
            elif 38 <= actualSize < 42:
                return 'S'
            elif 42 <= actualSize < 46:
                return 'M'
            elif 46 <= actualSize < 50:
                return 'L'
            elif 50 <= actualSize < 56:
                return 'XL'
            elif 56 <= actualSize <= 60:
                return 'XXL'
            elif actualSize > 60:
                return 'Maaf size Anda belum tersedia'
        elif size == 'us':
            if actualSize < 30:
                return 'Maaf size Anda belum tersedia'
            elif 30 <= actualSize < 32:
                return 'XXS'
            elif 32 <= actualSize < 34:
                return 'XS'
            elif 34 <= actualSize < 38:
                return 'S'
            elif 38 <= actualSize < 40:
                return 'M'
            elif 40 <= actualSize < 46:
                return 'L'
            elif 46 <= actualSize < 48:
                return 'XL'
            elif 48 <= actualSize <= 50:
                return 'XXL'
            elif actualSize > 50:
                return 'Maaf size Anda belum tersedia'
        elif size == 'eu':
            if actualSize < 32:
                return 'Maaf size Anda belum tersedia'
            elif 32 <= actualSize < 34:
                return 'XXS'
            elif 34 <= actualSize < 36:
                return 'XS'
            elif 36 <= actualSize < 38:
                return 'S'
            elif 38 <= actualSize < 40:
                return 'M'
            elif 40 <= actualSize < 44:
                return 'L'
            elif 44 <= actualSize < 48:
                return 'XL'
            elif 48 <= actualSize <= 54:
                return 'XXL'
            elif actualSize > 54:
                return 'Maaf size Anda belum tersedia'

@app.route('/')
def index ():
    return render_template('index.html')

@app.route('/Carupat.AI')
def predictionpage():
    return render_template('carupat2.html')

@app.route('/process', methods=['GET','POST'])
def hitung():
    if request.method == 'POST':
        data={
            'lb': request.form.get('panjangBahu'),
            'pb' : request.form.get('panjangBadan'),
            'pl' : request.form.get('panjangLengan'),
            'ld' : request.form.get('lingkarDada'),
            'gender' : request.form.get('gender'),
            'size' : request.form.get('size')
        }
        columns=['ld','pb','lb']
        df = pd.DataFrame([data],columns=columns)
        result = model.predict(df)
        actualSize = int(result)
        
        print('======= actual size ========')
        print(actualSize)

        print('====== df ========')
        print(df)

        estimated_size= logic(actualSize, data.get('gender'), data.get('size'))
        print('====== estimated_size ========')
        print(estimated_size)

        return render_template('carupat2.html', estimatedSize=estimated_size, panjangBahu=data.get('lb'), panjangBadan=data.get('pb'), panjangLengan=data.get('pl'), lingkarDada=data.get('ld'), gender=data.get('gender'), size=data.get('size'), fileName=request.form.get('fileName'), clothingType=request.form.get('clothingType'))
    else:
        return render_template('carupat2.html')

############################ MODEL V1 ############################
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def detect_landmarks(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("model_predictor_body_landmark.dat")

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if not faces:
        print("No face detected.")
        return None

    face = faces[0]
    landmarks = predictor(gray, face)

    right_shoulder = (landmarks.part(12).x, landmarks.part(12).y)
    left_shoulder = (landmarks.part(4).x, landmarks.part(4).y)
    right_wrist = (landmarks.part(16).x, landmarks.part(16).y)
    right_hip = (landmarks.part(8).x, landmarks.part(8).y)

    return right_shoulder, left_shoulder, right_wrist, right_hip

@app.route('/upload', methods=['POST'])
def upload():
    result = {'panjang_bahu': 0, 'panjang_badan': 0, 'panjang_lengan': 0, 'lingkar_dada': 0}
    if 'file' not in request.files:
        return jsonify(result)

    file = request.files['file']
    if file.filename == '':
        return jsonify(result)

    file_name = secure_filename(file.filename)
    file_path = os.path.join('static', 'uploads', file_name)
    file.save(file_path)

    landmarks = detect_landmarks(file_path)

    if landmarks:
        right_shoulder, left_shoulder, right_wrist, right_hip = landmarks
        result['panjang_bahu'] = calculate_distance(right_shoulder, left_shoulder) - 27 # biar engga kepanjangan
        result['panjang_badan'] = calculate_distance(right_shoulder, right_hip) - 40 # biar engga kepanjangan
        result['panjang_lengan'] = calculate_distance(right_shoulder, right_wrist) - 40 # biar engga kepanjangan
        result['lingkar_dada'] = (result['panjang_bahu'] * 2.5)  # biar engga kepanjangan
        result['file_name'] = file_name

    return jsonify(result)
############################ END #################################

############################ MODEL V2 ############################
def calculate_distance_v2(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def detect_reference_shoulder_width(image_path):
    mp_pose = mp.solutions.pose

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

            shoulder_width_px = calculate_distance(left_shoulder, right_shoulder)

            return shoulder_width_px

    return None

def detect_landmarks_v2(image_path, panjang_bahu):
    reference_shoulder_width_cm = panjang_bahu

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y

            shoulder_width_px = calculate_distance(left_shoulder, right_shoulder)
            body_length_px = calculate_distance(left_shoulder, left_hip) + calculate_distance(right_shoulder, right_hip)
            sleeve_length_px = calculate_distance(left_shoulder, left_wrist)
            chest_size_px = calculate_distance(left_shoulder, right_hip)

            pixels_per_cm = shoulder_width_px / reference_shoulder_width_cm

            shoulder_width_cm = shoulder_width_px / pixels_per_cm
            body_length_cm = body_length_px / pixels_per_cm
            sleeve_length_cm = sleeve_length_px / pixels_per_cm
            chest_size_cm = chest_size_px / pixels_per_cm

            print("Shoulder Width:", shoulder_width_cm, "cm")
            print("Body Length:", body_length_cm, "cm")
            print("Sleeve Length:", sleeve_length_cm, "cm")
            print("Chest Size:", chest_size_cm, "cm")

    output_image_path = os.path.join('static/enchanced', os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)
    print("Output image saved at:", output_image_path)

    return shoulder_width_cm, body_length_cm, sleeve_length_cm, chest_size_cm

@app.route('/upload-v2', methods=['POST'])
def upload_v2():
    result = {'panjang_bahu': 0, 'panjang_badan': 0, 'panjang_lengan': 0, 'lingkar_dada': 0}
    if 'file' not in request.files:
        return jsonify(result)

    file = request.files['file']
    if file.filename == '':
        return jsonify(result)

    # panjang_bahu = request.form['panjang_bahu']
    # panjang_bahu_float = float(panjang_bahu)

    file_name = secure_filename(file.filename)
    file_path = os.path.join('static', 'uploads', file_name)
    file.save(file_path)

    reference_shoulder_width_px = detect_reference_shoulder_width(file_path)
    print('=====how much px======')
    print(reference_shoulder_width_px)

    print('=====how much cm======')
    # shoulder_width_cm = (reference_shoulder_width_px / (reference_shoulder_width_px * float(0.0264583333))) * 2 # estimatedly
    shoulder_width_cm = (reference_shoulder_width_px * float(0.0264583333)) * 10000 # estimatedly
    print(shoulder_width_cm)

    # landmarks = detect_landmarks_v2(file_path, panjang_bahu_float)
    landmarks = detect_landmarks_v2(file_path, shoulder_width_cm)
    if landmarks:
        shoulder_width_cm, body_length_cm, sleeve_length_cm, chest_size_cm = landmarks
        result['panjang_bahu'] = shoulder_width_cm
        result['panjang_badan'] = body_length_cm
        result['panjang_lengan'] = sleeve_length_cm
        result['lingkar_dada'] = chest_size_cm
        result['file_name'] = file_name

    return jsonify(result)
############################ END #################################

if __name__ == '__main__':
    app.run(debug=True)