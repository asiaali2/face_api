from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import os
from PIL import Image
import io

app = Flask(__name__)

EMPLOYEE_FOLDER = "images/employees"

# 🔥 كاش للوجوه
known_encodings = {}
known_names = {}


# 🔷 تحميل كل الموظفين مرة واحدة
def load_all_employees():
    print("🔄 Loading employee faces...")
    for file in os.listdir(EMPLOYEE_FOLDER):
        if file.endswith(".jpg") or file.endswith(".png"):
            employee_id = os.path.splitext(file)[0]
            path = os.path.join(EMPLOYEE_FOLDER, file)

            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings[employee_id] = encodings[0]
                known_names[employee_id] = employee_id
                print(f"✅ Loaded {employee_id}")
            else:
                print(f"⚠️ No face found in {file}")


# تحميل عند تشغيل السيرفر
load_all_employees()


@app.route('/verify_face', methods=['POST'])
def verify_face():
    try:
        employee_id = request.form.get('employee_id')
        file = request.files.get('image')

        if not employee_id or not file:
            return jsonify({"status": "error", "message": "Missing data"}), 400

        # 🔴 تحقق من وجود الموظف
        if employee_id not in known_encodings:
            return jsonify({"status": "error", "message": "Employee not found"}), 404

        known_encoding = known_encodings[employee_id]

        # 🔷 قراءة الصورة
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(img)

        # 🔷 استخراج الوجه
        encodings = face_recognition.face_encodings(img)

        if len(encodings) == 0:
            return jsonify({"status": "fail", "message": "No face detected"})

        unknown_encoding = encodings[0]

        # 🔷 المقارنة
        distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]

        # 🔥 حد الأمان (تقدر تعدله)
        THRESHOLD = 0.5

        match = distance < THRESHOLD

        return jsonify({
            "status": "success",
            "match": bool(match),
            "confidence": float(1 - distance),
            "distance": float(distance)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# 🔷 إضافة موظف جديد (مهم 🔥)
@app.route('/add_employee', methods=['POST'])
def add_employee():
    try:
        employee_id = request.form.get('employee_id')
        file = request.files.get('image')

        if not employee_id or not file:
            return jsonify({"status": "error"}), 400

        path = os.path.join(EMPLOYEE_FOLDER, f"{employee_id}.jpg")

        # حفظ الصورة
        file.save(path)

        # استخراج الوجه
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            os.remove(path)
            return jsonify({"status": "fail", "message": "No face detected"})

        # 🔥 إضافة للكاش مباشرة
        known_encodings[employee_id] = encodings[0]

        return jsonify({"status": "success", "message": "Employee added"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# 🔷 إعادة تحميل الوجوه بدون إعادة تشغيل السيرفر
@app.route('/reload', methods=['GET'])
def reload_faces():
    known_encodings.clear()
    load_all_employees()
    return jsonify({"status": "reloaded"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)