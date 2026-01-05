import numpy as np
from PIL import Image
import hashlib
import os


class FaceDetector:
    """ Face detection and recognition using DeepFace """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(FaceDetector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize DeepFace settings"""
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
        except ImportError:
            print("Warning: deepface not installed. Face recognition will not work.")
            self.DeepFace = None

    def warm_up(self):
        """Pre-load models to eliminate startup latency"""
        if self.DeepFace is None:
            return
            
        print("🚀 [Biometric] Pre-loading DeepFace Facenet512 model...")
        try:
            # Pre-load Facenet512 model
            self.DeepFace.build_model("Facenet512")
            
            # Warm up the detector with a dummy blank image
            dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)
            self.DeepFace.extract_faces(
                img_path=dummy_img,
                detector_backend='opencv',
                enforce_detection=False
            )
            
            # NEW: Warm up the representation (embedding generation)
            self.DeepFace.represent(
                img_path=dummy_img,
                model_name="Facenet512",
                detector_backend='opencv',
                enforce_detection=False
            )
            print("✅ [Biometric] DeepFace warmed up and ready!")
        except Exception as e:
            print(f"❌ [Biometric] Warm up failed: {str(e)}")
    
    def detect_face(self, image_path_or_array):
        """
        Detect face in an image using DeepFace
        """
        if self.DeepFace is None:
            raise RuntimeError("DeepFace is not installed.")
            
        try:
            # Detect face and return info
            objs = self.DeepFace.extract_faces(
                img_path=image_path_or_array,
                detector_backend='opencv',
                enforce_detection=False
            )
            
            if not objs:
                return None
                
            # DeepFace returns a list of dicts. We pick the first (usually only) one.
            face_obj = objs[0]
            area = face_obj['facial_area']
            
            return {
                'box': [area['x'], area['y'], area['w'], area['h']],
                'confidence': 1.0  # OpenCV backend doesn't always provide confidence
            }
        except Exception as e:
            print(f"Error detecting face: {str(e)}")
            return None
    
    def extract_face(self, image_path_or_array, required_size=(160, 160)):
        """
        Extract face using DeepFace
        """
        if self.DeepFace is None:
            return None
            
        try:
            objs = self.DeepFace.extract_faces(
                img_path=image_path_or_array,
                target_size=required_size,
                detector_backend='opencv',
                enforce_detection=True
            )
            
            if not objs:
                return None
                
            # Get face as numpy array (DeepFace returns 0-1 float usually, convert to 0-255)
            face = objs[0]['face']
            if face.max() <= 1.0:
                face = (face * 255).astype('uint8')
            else:
                face = face.astype('uint8')
                
            return face
        except Exception:
            return None


def preprocess_face(face_pixels):
    """ Post-process for DeepFace (often handled internally, but kept for compatibility) """
    return face_pixels


def generate_embedding(image_path_or_array):
    """
    Generate face embedding using DeepFace
    """
<<<<<<< HEAD
    from deepface import DeepFace
    try:
=======
    try:
        from deepface import DeepFace
    except ImportError as e:
        print(f"CRITICAL: Could not import DeepFace: {e}")
        # return None
        print("WARNING: Using DUMMY embedding due to ImportError.")
        return np.random.rand(512)
    except Exception as e:
        print(f"CRITICAL: Crash during DeepFace import: {e}")
        # return None
        print("WARNING: Using DUMMY embedding due to Import Crash.")
        return np.random.rand(512)
    try:
        print("DEBUG: Executing DeepFace.represent...")
>>>>>>> bd1274c (Added Chat and rafactored code)
        embeddings = DeepFace.represent(
            img_path=image_path_or_array,
            model_name="Facenet512",
            detector_backend='opencv',
            enforce_detection=True
        )
        if embeddings:
            emb = np.array(embeddings[0]['embedding'])
            # L2 Normalize the vector to ensure Euclidean distance is meaningful (0-2 range)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        return None
    except Exception as e:
<<<<<<< HEAD
        print(f"Error generating embedding: {str(e)}")
        return None


=======
        print(f"DeepFace represent error: {e}")
        return None



>>>>>>> bd1274c (Added Chat and rafactored code)
def compare_faces(captured_image, registered_embedding_path):
    """
    Compare captured image against stored embedding using DeepFace
    """
    from deepface import DeepFace
    
    # In DeepFace, it's often better to compare two images or a list of representations
    # But since we have saved .npy embeddings, we can do direct distance calculation
    # Or use DeepFace.verify() if we have the file paths
    
    new_embedding = generate_embedding(captured_image)
    if new_embedding is None:
        return False, 0, 0
        
    stored_embedding = np.load(registered_embedding_path)
    
    # Simple Euclidean distance (same as DeepFace's Facenet512 distance)
    distance = np.linalg.norm(new_embedding - stored_embedding)
    threshold = 0.6  # Standard for Facenet512
    
    is_match = distance <= threshold
    confidence = max(0, 1 - (distance / 2.0))
    
    return is_match, float(distance), float(confidence)


def calculate_checksum(file_path):
    """
    Calculate SHA-256 checksum of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 checksum as hexadecimal string
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def verify_checksum(file_path, expected_checksum):
    """
    Verify file integrity using checksum
    
    Args:
        file_path: Path to the file
        expected_checksum: Expected SHA-256 checksum
        
    Returns:
        bool: True if checksum matches, False otherwise
    """
    actual_checksum = calculate_checksum(file_path)
    return actual_checksum == expected_checksum


def save_face_image(image_array, save_path):
    """
    Save face image to disk
    
    Args:
        image_array: Face image as numpy array
        save_path: Path where to save the image
        
    Returns:
        str: Checksum of the saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to PIL Image and save
    image = Image.fromarray(image_array.astype('uint8'))
    image.save(save_path)
    
    # Calculate and return checksum
    return calculate_checksum(save_path)