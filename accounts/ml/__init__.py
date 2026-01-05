# ml/__init__.py
"""
Machine Learning module for biometric authentication
"""

from .face_utils import FaceDetector, preprocess_face

__all__ = ['FaceDetector', 'preprocess_face']