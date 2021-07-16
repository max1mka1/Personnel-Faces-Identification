import os
import cv2
import uuid
import dlib
import base64
import pickle
import logging
import numpy as np
from urllib.parse import quote
from typing import Dict, List, Tuple, Any
from imutils import resize, face_utils
from sklearn.metrics.pairwise import cosine_similarity

import face_recognition
import werkzeug.datastructures as datastructures

logging.basicConfig(level=logging.DEBUG)
logging.info(f'Dlib uses CUDA: {dlib.DLIB_USE_CUDA}')

MAX_FACES: int = 1
IM_HEIGHT: int = 512
LANDMARKS_COUNT: int = 68
ACCURACY_THRESHOLD: float = 0.90
LANDMARKS_COLOR: Tuple[int, int, int] = (0, 0, 255)

MODEL_USED = "./models/shape_predictor_68_face_landmarks_GTX.dat"
CNN_DETECTOR_MODEL_WHEIGHTS = './models/mmod_human_face_detector.dat'

face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_DETECTOR_MODEL_WHEIGHTS)
predictor = dlib.shape_predictor(MODEL_USED)


def load_database() -> Dict[str, Dict[str, str]]:
    """
    Just simpliest DB, no time for real DB...
    :return:
    """
    database = {'004': {'Имя сотрудника': 'Прохоренко И.П.',
                        'Должность': 'Сталевар',
                        'Идентификатор должности': '238-046-958'},
                '066': {'Имя сотрудника': 'Тиньгаев М.И.',
                        'Должность': 'Инженер компьютерного зрения',
                        'Идентификатор должности': '777-777-777'},
                '079': {'Имя сотрудника': 'Прохоренко В.П.',
                        'Должность': 'Сталевар',
                        'Идентификатор должности': '555-555-555'},
                '116': {'Имя сотрудника': 'Прохоренко Н.И.',
                        'Должность': 'Сталевар',
                        'Идентификатор должности': '444-444-444'},
                '140': {'Имя сотрудника': 'Прохоренко К.И.',
                        'Должность': 'Сталевар',
                        'Идентификатор должности': '333-333-333'},
                '168': {'Имя сотрудника': 'Прохоренко Н.Н.',
                        'Должность': 'Сталевар',
                        'Идентификатор должности': '222-222-222'}
                }

    return database


def decode_and_resize_image(image: datastructures.FileStorage) -> Tuple[np.array, np.array]:
    """
    Function that converts FileStorage file into cv2 numpy array
    :param image:       datastructures.FileStorage image file
    :return:            cv2 resized original image and gray image of np.array type
    """
    image: np.array = np.fromstring(image.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    image = resize(image, height=IM_HEIGHT)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_clone = image.copy()

    return image, image_gray, image_clone


def image_to_text(image: np.array):
    """

    :param image:
    :return:
    """
    retval, buffer = cv2.imencode('.jpg', image)
    image_as_text = base64.b64encode(buffer)

    return image_as_text


def check_person_access(filestorage_image: datastructures.FileStorage,
                        person_id: str,
                        root_path: str,
                        only_json_return: bool = False) -> Dict[str, str]:
    """
    Function to check if there is a person on image and
    to check if person have access is granted.
    :param image:           np.array image from camera
    :param person_id:       Person ID number
    :return:                Dict with image and person info, otherwise error
    """

    faces_bboxes = None
    image_gray: np.array = None
    image_clone: np.array = None
    authorized_staff_path: str = None
    database: Dict = None
    database_path: str = os.path.join(root_path, 'database', 'authorized_staff_db.pickle')
    successful_recognitions_path: str = os.path.join(root_path, 'uploads', 'successful')
    unsuccessful_recognitions_path: str = os.path.join(root_path, 'uploads', 'unsuccessful')
    image_as_text: None
    identification_error_info: str = None        #
    all_identification_errors: List = []
    recognition_error: bool = False         # Ошибка идентификации
    recognition_confidence: float = 0.0
    faces_count: int = 0
    image_uuid: str = None


    image_report: Dict[str, Any] = {'Ошибка идентификации:': 'Нет',
                                    'Уверенность системы в схожести фотографий:': recognition_confidence,
                                    'Количество лиц в кадре:': faces_count,
                                    'Порог точности:': ACCURACY_THRESHOLD}

    try:
        image, image_gray, image_clone = decode_and_resize_image(filestorage_image)
    except:
        identification_error_info = 'Ошибка во время чтения изображения! Проверьте изображение...'
        all_identification_errors.append(identification_error_info)
        logging.info(identification_error_info)

    # Формируем базу данных только если предоставили ID для сравнения
    if not os.path.exists(database_path):
        database = load_database()
        # Директория с эталонными фотографиями сотрудников
        authorized_staff_path = os.path.join(root_path, 'data', 'authorized_staff_base')
        for image_name in os.listdir(authorized_staff_path):
            auth_person_id = image_name.split('_')[0]
            image_path = os.path.join(authorized_staff_path, image_name)

            authorized_image = cv2.imread(image_path)
            authorized_image = cv2.cvtColor(authorized_image, cv2.COLOR_BGR2RGB)
            authorized_image = resize(authorized_image, height=IM_HEIGHT)
            authorized_image_gray = cv2.cvtColor(authorized_image, cv2.COLOR_BGR2GRAY)
            authorized_face_bboxes = face_recognition.face_locations(authorized_image, model='cnn')
            database[auth_person_id]['embedding'] = face_recognition.face_encodings(authorized_image, authorized_face_bboxes)[0]

        # Save dict with embeddings into pickle file
        with open(database_path, 'wb') as handle:
            pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Open database with authorized persons embeddings
        with open(database_path, 'rb') as handle:
            database = pickle.load(handle)

    try:
        faces_bboxes = face_detector(image_gray, 1)
        faces_detected = cnn_face_detector(image_gray, 1)

        # loop over detected faces
        for face in faces_detected:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y

            # draw box over face
            cv2.rectangle(image_clone, (x, y), (x + w, y + h), (0, 0, 255), 2)

    except:
        identification_error_info = 'Ошибка во время поиска лиц'
        all_identification_errors.append(identification_error_info)
        logging.info(identification_error_info)

    faces_count = len(faces_bboxes)
    if faces_bboxes:
        if len(faces_bboxes) > MAX_FACES:
            identification_error_info = 'Больше одного лица в кадре'
            all_identification_errors.append(identification_error_info)
            logging.info(identification_error_info)
    else:
        identification_error_info = 'Нет лиц в кадре'
        all_identification_errors.append(identification_error_info)
        logging.info(identification_error_info)

    any_face_was_found = len(faces_bboxes) > 0

    if any_face_was_found:
        for rect in faces_bboxes:
            shape = predictor(image_gray, rect)
            shape = face_utils.shape_to_np(shape)

            for point in range(1, LANDMARKS_COUNT):
                coords = shape.tolist()[point]
                cv2.circle(image_clone, (coords[0], coords[1]), 1, LANDMARKS_COLOR, thickness=1)
    else:
        identification_error_info = 'Нет лиц в кадре'
        all_identification_errors.append(identification_error_info)
        logging.info(identification_error_info)

    image_face_bboxes = face_recognition.face_locations(image, model='cnn')
    image_embedding = face_recognition.face_encodings(image, image_face_bboxes)[0]
    image_uuid = uuid.uuid4().hex
    if person_id in database.keys():

        recognition_confidence = cosine_similarity(image_embedding.reshape(1, -1),
                                                   database[person_id]['embedding'].reshape(1, -1))
        if recognition_confidence > ACCURACY_THRESHOLD:
            recognition_error = False
            uploaded_image_name = os.path.join(successful_recognitions_path, f'{image_uuid}.jpg')
            if person_id in database.keys():
                image_report.update(database[person_id])
        else:
            recognition_error = True
            # Save image
            uploaded_image_name = os.path.join(unsuccessful_recognitions_path, f'{image_uuid}.jpg')
            identification_error_info = 'Другой человек в кадре'  # низкая степень схожести, но id в базе есть
            all_identification_errors.append(identification_error_info)
            identification_error_info = 'Степень уверенности ниже пороговой'
            all_identification_errors.append(identification_error_info)
            logging.info(identification_error_info)
    else:
        identification_error_info = 'Идентификационный номер отсутствует в БД'
        all_identification_errors.append(identification_error_info)
        uploaded_image_name = os.path.join(unsuccessful_recognitions_path, f'{image_uuid}.jpg')
        logging.info(identification_error_info)
    cv2.imwrite(uploaded_image_name, image)

    image_report['Количество лиц в кадре:'] = faces_count
    image_report['Ошибка идентификации:'] = 'Да' if recognition_error else 'Нет'
    image_report['Уверенность системы в схожести фотографий:'] = round(float(recognition_confidence), 2)

    if identification_error_info:
        image_report['Причина ошибки:'] = '; '.join(all_identification_errors)

    image_as_text = image_to_text(image_clone)
    image_report.update({'result_image': f'data:image/png;base64,{quote(image_as_text)}',
                         'Идентификатор файла': image_uuid})
    if 'embedding' in image_report.keys():
        del image_report['embedding']

    for key, val in image_report.items():
        if key != 'result_image':
            logging.info(f'{key} {val}')

    if only_json_return:
        del image_report['result_image']
        return image_report
    else:
        return image_report

