import os
from flask import Flask, render_template, request
from face_recognition_service import get_image_with_landmarks

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './'


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", result={})
    else:
        image = request.files["image"]
        path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(path)
        result_from_landmarks = get_image_with_landmarks(path)
        os.remove(path)

        return render_template("index.html", result=result_from_landmarks)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
