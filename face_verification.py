from deepface import DeepFace
import cv2
import os
import logging
from utils import file_exists, read_yaml

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

config_path = "config.yaml"
config = read_yaml(config_path)

artifacts = config['artifacts']
# Normalize configured cascade path (supports Windows-style backslashes in config)
configured_cascade = artifacts.get('HAARCASCADE_PATH', '')
configured_cascade = configured_cascade.replace('\\', os.sep).replace('/', os.sep)
cascade_path = os.path.abspath(configured_cascade)

# Normalize intermediate/output dir and ensure it exists
output_path = artifacts.get('INTERMIDEIATE_DIR', 'data/02_intermediate_data')
output_path = output_path.replace('\\', os.sep).replace('/', os.sep)
os.makedirs(os.path.join(os.getcwd(), output_path), exist_ok=True)

def _load_cascade(path: str):
    """Load cascade classifier with fallback to OpenCV bundled cascades."""
    logging.info(f"Loading cascade from: {path}")
    try:
        face_cascade = cv2.CascadeClassifier(path)
        if getattr(face_cascade, 'empty', lambda: False)():
            logging.warning(f"Cascade at {path} is empty")
            raise RuntimeError("Empty cascade")
        logging.info(f"Loaded cascade successfully")
        return face_cascade
    except Exception:
        # Fallback to bundled cascade
        fallback = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        logging.info(f"Using bundled cascade: {fallback}")
        face_cascade = cv2.CascadeClassifier(fallback)
        if getattr(face_cascade, 'empty', lambda: False)():
            logging.error("Failed to load any Haar cascade")
            return None
        return face_cascade

def detect_and_extract_face(img):
    logging.info("Extracting face")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = _load_cascade(cascade_path)
    if face_cascade is None:
        logging.error("No cascade classifier available")
        return None

    try:
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    except Exception as e:
        logging.error(f"detectMultiScale failed: {e}")
        return None

    max_area = 0
    largest_face = None
    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            max_area = area
            largest_face = (x, y, w, h)

    if largest_face is not None:
        (x, y, w, h) = largest_face
        new_w = int(w * 1.50)
        new_h = int(h * 1.50)
        new_x = max(0, x - int((new_w - w) / 2))
        new_y = max(0, y - int((new_h - h) / 2))
        extracted_face = img[new_y:new_y+new_h, new_x:new_x+new_w]
        
        current_wd = os.getcwd()
        filename = os.path.join(current_wd, output_path, "extracted_face.jpg")

        if os.path.exists(filename):
            os.remove(filename)

        cv2.imwrite(filename, extracted_face)
        logging.info(f"Face saved: {filename}")
        return filename
    else:
        logging.warning("No face detected")
        return None


def deepface_face_comparison(image1_path, image2_path):
    """Attempts face verification and returns (is_verified: bool, details: list).

    details is a list of dicts, one per backend attempted, each containing
    - backend: str
    - success: bool
    - result: verification dict if success else None
    - error: exception string if failed
    """
    logging.info("Verifying the images....")
    details = []

    # Ensure files exist (normalize separators)
    image1_path = image1_path.replace('\\', os.sep).replace('/', os.sep) if isinstance(image1_path, str) else image1_path
    image2_path = image2_path.replace('\\', os.sep).replace('/', os.sep) if isinstance(image2_path, str) else image2_path

    img1_exists = file_exists(image1_path) if isinstance(image1_path, str) else True
    img2_exists = file_exists(image2_path) if isinstance(image2_path, str) else True

    if not (img1_exists and img2_exists):
        logging.warning(f"One or both image paths do not exist: {image1_path}, {image2_path}")
        details.append({
            'backend': None,
            'success': False,
            'result': None,
            'error': 'One or both image paths do not exist'
        })
        return False, details

    # Try verification with two detector backends for robustness
    backends = ['opencv', 'mtcnn']
    for backend in backends:
        record = {'backend': backend, 'success': False, 'result': None, 'error': None}
        try:
            logging.info(f"Attempting DeepFace.verify with detector_backend='{backend}'")
            verification = DeepFace.verify(img1_path=image1_path, img2_path=image2_path, detector_backend=backend)
            logging.info(f"DeepFace result ({backend}): {verification}")
            record['result'] = verification
            record['success'] = bool(isinstance(verification, dict) and verification.get('verified'))
        except Exception as e:
            logging.warning(f"DeepFace.verify with backend '{backend}' failed: {e}")
            record['error'] = str(e)

        details.append(record)
        if record['success']:
            logging.info("Faces are verified as the same person")
            return True, details

    logging.info("Faces are not verified as the same person by any backend")
    return False, details


def get_face_embeddings(image_path):
    logging.info(f"Retrieving face embeddings from image: {image_path}")

    if not file_exists(image_path):
        logging.warning(f"Image path does not exist: {image_path}")
        return None
    
    embedding_objs = DeepFace.represent(img_path=image_path, model_name="Facenet")
    embedding = embedding_objs[0]["embedding"]

    if len(embedding) > 0:
        logging.info("Face embeddings retrieved successfully")
        return embedding
    else:
        logging.warning("Failed to retrieve face embeddings")
        return None