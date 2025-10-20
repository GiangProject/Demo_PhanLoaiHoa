from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import os
from model_loader import predict  # Hàm dự đoán

app = Flask(__name__)

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
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            predictions = predict(filepath)

            # ✅ Quan trọng: dùng dấu '/' thay vì '\'
            image_file = f"uploads/{filename}"

            return render_template('index.html', 
                                   predictions=predictions, 
                                   image_file=image_file)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
