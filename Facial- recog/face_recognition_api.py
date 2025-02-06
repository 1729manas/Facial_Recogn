import scipy.misc
import dlib
import numpy as np

from PIL import Image

face_detector = dlib.get_frontal_face_detector()

predictor_model = './models/shape_predictor_68_face_landmarks.dat'
pose_predictor = dlib.shape_predictor(predictor_model)

face_recognition_model = './models/dlib_face_recognition_resnet_model_v1.dat'
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def _rect_to_tuple(rect):

    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _tuple_to_rect(rect):

    return dlib.rectangle(rect[3], rect[0], rect[1], rect[2])


def _trim_rect_tuple_to_bounds(rect, image_shape):

    return max(rect[0], 0), min(rect[1], image_shape[1]), min(rect[2], image_shape[0]), max(rect[3], 0)


def face_distance(face_encodings, face_to_compare):

    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)



def load_image_file(filename, mode='RGB'):
    
    img = Image.open(filename)
    

    img_array = np.array(img)

    
    if img_array.shape[0] > 800:
        baseheight = 500
        w = (baseheight / img_array.shape[0])
        p = int(img_array.shape[1] * w)
        img = img.resize((p, baseheight))
    elif img_array.shape[1] > 800:
        baseheight = 500
        w = (baseheight / img_array.shape[1])
        p = int(img_array.shape[0] * w)
        img = img.resize((baseheight, p))

    
    return np.array(img)


def _raw_face_locations(img, number_of_times_to_upsample=1):

    return face_detector(img, number_of_times_to_upsample)


def face_locations(img, number_of_times_to_upsample=1):

    return [_trim_rect_tuple_to_bounds(_rect_to_tuple(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample)]


def _raw_face_landmarks(face_image, face_locations=None):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_tuple_to_rect(face_location) for face_location in face_locations]

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def face_landmarks(face_image, face_locations=None):

    landmarks = _raw_face_landmarks(face_image, face_locations)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    
    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } for points in landmarks_as_tuples]


def face_encodings(face_image, known_face_locations=None, num_jitters=1):

    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)

    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):

    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
