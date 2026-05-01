from flask import Flask, request, jsonify
from deepface import DeepFace
import os
from PIL import Image
import io

app = Flask(__name__)

EMPLOYEE_FOLDER = "images/employees"

# 🔷 تأكد المجلد موجود
os.makedirs(EMPLOYEE_FOLDER, exist_ok=True)


# 🔷 التحقق من الوجه باستخدام DeepFace
@app.route('/verify_face', methods=['POST'])
def verify_face():
    try:
        employee_id = request.form.get('employee_id')
        file = request.files.get('image')

        if not employee_id or not file:
            return jsonify({"status": "error", "message": "Missing data"}), 400

        employee_path = os.path.join(EMPLOYEE_FOLDER, f"{employee_id}.jpg")

        # 🔴 تحقق من وجود الموظف
        if not os.path.exists(employee_path):
            return jsonify({"status": "error", "message": "Employee not found"}), 404

        # 🔷 حفظ الصورة المرسلة مؤقتًا
        temp_path = "temp.jpg"
        file.save(temp_path)

        # 🔥 المقارنة
        result = DeepFace.verify(
            img1_path=employee_path,
            img2_path=temp_path,
            enforce_detection=False
        )

        return jsonify({
            "status": "success",
            "match": bool(result["verified"]),
            "distance": float(result["distance"])
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# 🔷 إضافة موظف جديد
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

        return jsonify({"status": "success", "message": "Employee added"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# 🔷 إعادة تحميل (ما تحتاجها فعليًا الآن)
@app.route('/reload', methods=['GET'])
def reload_faces():
    return jsonify({"status": "reloaded"})


if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
