# app.py

from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import os
import uuid
from model_service import predict  # Import hàm dự đoán đã được tải sẵn

app = Flask(__name__)

# Cấu hình thư mục để lưu ảnh upload
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='Không có file nào được chọn.')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='Chưa chọn file nào.')

        if file:
            # Tạo tên file duy nhất
            file_ext = os.path.splitext(file.filename)[1]
            filename = str(uuid.uuid4()) + file_ext
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Gọi hàm dự đoán
            predictions = predict(filepath)

            # Trả về kết quả + đường dẫn ảnh hợp lệ
            return render_template('index.html',
                                   predictions=predictions,
                                   image_file=url_for('static', filename='uploads/' + filename))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
