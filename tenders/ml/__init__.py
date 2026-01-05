"""
Tender Anomaly Detection ML Module
===================================
Contains the machine learning pipeline for detecting anomalous tenders.
"""

from .evaluator import TenderAnomalyEvaluator
from .text_extractor import extract_text_from_file

__all__ = ['TenderAnomalyEvaluator', 'extract_text_from_file']
