import sys
import time
import cv2
import dlib
import base64
from imutils import resize, face_utils
from urllib.parse import quote
import face_recognition


print(f'Dlib uses CUDA: {dlib.DLIB_USE_CUDA}')

MODEL_USED = "./models/shape_predictor_68_face_landmarks_GTX.dat"
CNN_DETECTOR_MODEL_WHEIGHTS = './models/mmod_human_face_detector.dat'

face_detector = dlib.get_frontal_face_detector()
# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_DETECTOR_MODEL_WHEIGHTS)
predictor = dlib.shape_predictor(MODEL_USED)


def load_database():
    """

    :return:
    """

def get_image_with_landmarks(file_path: str):
    """

    :param file_path:
    :return:
    """
    rects = None
    gray = None
    clone = None

    try:
        image = cv2.imread(file_path, 1)
        image = resize(image, height=240)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clone = image.copy()

        start = time.time()
        rects = face_detector(gray, 1)
        end = time.time()
        print(f"face_detector : {end - start}.2f", flush=True)

        start = time.time()
        faces_cnn = cnn_face_detector(gray, 1)
        end = time.time()
        print(f"CNN : {end - start}.2f", flush=True)
        print(f'faces_cnn: {faces_cnn}', flush=True)

        # loop over detected faces
        for face in faces_cnn:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y

            # draw box over face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    except Exception:
        return {'error': 'Error while reading the image'}

    any_face_was_found = len(rects) > 0
    if any_face_was_found:
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for point in range(1, 68):
                coords = shape.tolist()[point]
                cv2.circle(clone, (coords[0], coords[1]), 1, (0, 0, 255), thickness=2)
    else:
        return {'error': 'No face was detected in the image provided'}

    retval, buffer = cv2.imencode('.jpg', clone)
    image_as_text = base64.b64encode(buffer)

    return {'image_with_landmarks': 'data:image/png;base64,{}'.format(quote(image_as_text))}