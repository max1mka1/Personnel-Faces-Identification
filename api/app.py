import os
import logging
from typing import Dict, List, Tuple
from flask import Flask, render_template, request, send_from_directory
from face_recognition_service import check_person_access
from config.app_config import AppConfig
import werkzeug.datastructures as datastructures

logging.basicConfig(filename='app.log', encoding='utf-8', level=logging.DEBUG)

def register_extensions(app: Flask):
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


def create_app(config: object) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config)
    register_extensions(app)
    # app.config['UPLOAD_FOLDER'] = './uploads/'

    return app


app = create_app(AppConfig)


@app.route('/', methods=["GET", "POST"])
def index():
    """
    Flask web-form ifentification process
    """
    if request.method == "GET":
        return render_template("index.html", result={})
    elif request.method == 'POST':
        filestorage_image: datastructures.FileStorage = request.files["image"]
        person_id: str = request.form.get('person_id')  # запрос к данным формы и получение ID сотрудника

        if person_id:
            result = check_person_access(filestorage_image=filestorage_image,
                                         person_id=person_id,
                                         root_path=app.root_path,
                                         only_json_return=False)

            return render_template("index.html", result=result)
        else:
            return render_template("index.html", result={'error': 'Вы не ввели идентификационный номер сотрудника!'})


@app.route('/get_id', methods=["POST"])
def get_id():
    """
    API method for person ifentification
    :return:        json-file with required params
    """
    if request.method == 'POST':
        filestorage_image: datastructures.FileStorage = request.files["image"]
        person_id: str = request.args.get('username')

        json_result = check_person_access(filestorage_image=filestorage_image,
                                     person_id=person_id,
                                     root_path=app.root_path,
                                     only_json_return=True)

        return json_result


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                          'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
