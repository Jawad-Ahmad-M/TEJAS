"""
Tender Anomaly Evaluator
========================
Complete ML pipeline for detecting anomalous tenders.
Refactored from notebook code - uses Isolation Forest model.

Required Columns:
- Title
- Authority
- Object_Description
- CPV
- Estimated_Value
- Award_Criteria
- Conditions
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

<<<<<<< HEAD
=======
# Try to import torch for Autoencoder
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

>>>>>>> bd1274c (Added Chat and rafactored code)
warnings.filterwarnings('ignore')

# Get the directory where this module is located
MODULE_DIR = Path(__file__).parent
MODELS_DIR = MODULE_DIR / 'models'


<<<<<<< HEAD
class TenderAnomalyEvaluator:
    """
    Complete anomaly detection pipeline for tenders.
    Uses Isolation Forest as the primary detection model.
=======
if TORCH_AVAILABLE:
    class Autoencoder(nn.Module):
        """Deep learning-based reconstruction error detection."""
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim)
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


class TenderAnomalyEvaluator:
    """
    Complete ensemble anomaly detection pipeline for tenders.
    Combines Isolation Forest, LOF, One-Class SVM, and Autoencoder.
>>>>>>> bd1274c (Added Chat and rafactored code)
    """
    
    REQUIRED_COLUMNS = [
        'Title', 'Authority', 'Object_Description', 'CPV',
        'Estimated_Value', 'Award_Criteria', 'Conditions'
    ]
    
<<<<<<< HEAD
    # Anomaly thresholds
    THRESHOLDS = {
        'EXTREME': 0.9,
        'HIGH': 0.7,
        'MEDIUM': 0.5,
        'LOW': 0.3
    }
    
    def __init__(self, model_path=None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Optional path to pre-trained Isolation Forest model.
                       If not provided, a new model will be trained when needed.
        """
        self.isolation_forest = None
        self.scaler = None
        self.sentence_model = None
        self.model_path = model_path or (MODELS_DIR / 'isolation_forest.pkl')
        self.scaler_path = MODELS_DIR / 'scaler.pkl'
        
        # Load model if exists
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if they exist."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.isolation_forest = pickle.load(f)
                print(f"✓ Loaded Isolation Forest model from {self.model_path}")
            
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✓ Loaded scaler from {self.scaler_path}")
        except Exception as e:
            print(f"Warning: Could not load models: {e}")
=======
    # Anomaly thresholds (Adjusted: 0.6 is now HIGH risk)
    THRESHOLDS = {
        'EXTREME': 0.8,
        'HIGH': 0.6,
        'MEDIUM': 0.4,
        'LOW': 0.2
    }
    
    def __init__(self, models_dir=None, eager_load=False):
        """
        Initialize the evaluator and load pre-trained models.
        """
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        
        # Models
        self.isolation_forest = None
        self.lof = None
        self.oc_svm = None
        self.autoencoder = None
        self.scaler = None
        self.sentence_model = None
        
        # Paths
        self.paths = {
            'iso': self.models_dir / 'isolation_forest.pkl',
            'lof': self.models_dir / 'lof.pkl',
            'svm': self.models_dir / 'one_class_svm.pkl',
            'ae': self.models_dir / 'autoencoder.pth',
            'scaler': self.models_dir / 'scaler.pkl'
        }
        
        # Load models
        self._load_models()
        
        if eager_load:
            self._load_sentence_model()
    
    def _load_models(self):
        """Load all available pre-trained models."""
        # 1. Isolation Forest
        if self.paths['iso'].exists():
            try:
                with open(self.paths['iso'], 'rb') as f:
                    self.isolation_forest = pickle.load(f)
                print(f"✓ Loaded Isolation Forest")
            except Exception as e:
                print(f"Error loading Isolation Forest: {e}")
        
        # 2. LOF
        if self.paths['lof'].exists():
            try:
                with open(self.paths['lof'], 'rb') as f:
                    self.lof = pickle.load(f)
                print(f"✓ Loaded LOF")
            except Exception as e:
                print(f"Error loading LOF: {e}")
                
        # 3. One-Class SVM
        if self.paths['svm'].exists():
            try:
                with open(self.paths['svm'], 'rb') as f:
                    self.oc_svm = pickle.load(f)
                print(f"✓ Loaded One-Class SVM")
            except Exception as e:
                print(f"Error loading OC-SVM: {e}")
                
        # 4. Autoencoder
        if TORCH_AVAILABLE and self.paths['ae'].exists():
            try:
                # Assuming 768 dimensions from Sentence-BERT
                self.autoencoder = Autoencoder(input_dim=768)
                self.autoencoder.load_state_dict(torch.load(self.paths['ae'], map_location=torch.device('cpu')))
                self.autoencoder.eval()
                print(f"✓ Loaded Autoencoder")
            except Exception as e:
                print(f"Error loading Autoencoder: {e}")
        
        # 5. Scaler
        if self.paths['scaler'].exists():
            try:
                with open(self.paths['scaler'], 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✓ Loaded Scaler")
            except Exception as e:
                print(f"Error loading Scaler: {e}")
>>>>>>> bd1274c (Added Chat and rafactored code)
    
    def _load_sentence_model(self):
        """Lazy load the sentence transformer model."""
        if self.sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
<<<<<<< HEAD
                print("✓ Loaded Sentence-BERT model: all-mpnet-base-v2")
=======
>>>>>>> bd1274c (Added Chat and rafactored code)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )
        return self.sentence_model
    
    def validate_input(self, data):
        """
        Validate that input data has all required columns.
        
        Args:
            data: dict with tender data
            
        Returns:
            tuple: (is_valid, list of missing columns)
        """
        missing = []
        for col in self.REQUIRED_COLUMNS:
            if col not in data or not data[col]:
                missing.append(col)
        
        return len(missing) == 0, missing
    
    # ==========================================
    # STEP 1: PREPROCESSING
    # ==========================================
    
    def preprocess(self, data):
        """
        Step 1: Preprocess tender data.
        
        Args:
            data: dict with tender columns
            
        Returns:
            dict: Preprocessed data with additional features
        """
        processed = data.copy()
        
        # 1.1 Parse Estimated Value
        processed['Estimated_Value_Numeric'] = self._parse_estimated_value(
            data.get('Estimated_Value', '')
        )
        
        # 1.2 Extract CPV Hierarchy
        cpv_levels = self._extract_cpv_levels(data.get('CPV', ''))
        processed.update(cpv_levels)
        
        # 1.3 Clean Text Fields
        processed['Title_Clean'] = self._clean_text(data.get('Title', ''))
        processed['Object_Description_Clean'] = self._clean_text(data.get('Object_Description', ''))
        processed['Award_Criteria_Clean'] = self._clean_text(data.get('Award_Criteria', ''))
        processed['Conditions_Clean'] = self._clean_text(data.get('Conditions', ''))
        
        # 1.4 Create Combined Text
        processed['Combined_Text'] = self._combine_text(processed)
        
        # 1.5 Calculate Text Statistics
        processed['Title_Length'] = len(processed['Title_Clean'])
        processed['Description_Length'] = len(processed['Object_Description_Clean'])
        processed['Combined_Text_Length'] = len(processed['Combined_Text'])
        processed['Word_Count'] = len(processed['Combined_Text'].split())
        
        # 1.6 Log transform of value
        if processed['Estimated_Value_Numeric'] and processed['Estimated_Value_Numeric'] > 0:
            processed['Log_Estimated_Value'] = np.log1p(processed['Estimated_Value_Numeric'])
        else:
            processed['Log_Estimated_Value'] = 0
        
        return processed
    
    def _parse_estimated_value(self, value_str):
        """Parse estimated value from text to number."""
        if not value_str or pd.isna(value_str):
            return None
        
        value_str = str(value_str).upper()
        
        # Remove currency symbols
        value_str = re.sub(r'[€$£¥]', '', value_str)
        
        # Handle range (take average)
        if '-' in value_str or 'TO' in value_str:
            numbers = re.findall(r'[\d,]+\.?\d*', value_str)
            if len(numbers) >= 2:
                nums = [float(n.replace(',', '')) for n in numbers[:2]]
                return sum(nums) / 2
        
        # Extract numerical value
        value_str = re.sub(r'[^\d.]', '', value_str)
        
        try:
            return float(value_str) if value_str else None
        except:
            return None
    
    def _extract_cpv_levels(self, cpv_code):
        """Extract hierarchical levels from CPV code."""
        result = {
            'CPV_Level_2': 0,
            'CPV_Level_4': 0,
            'CPV_Level_6': 0,
            'CPV_Level_8': 0
        }
        
        if not cpv_code or pd.isna(cpv_code):
            return result
        
        cpv_str = re.sub(r'[^0-9]', '', str(cpv_code))
        
        if len(cpv_str) < 8:
            cpv_str = cpv_str.ljust(8, '0')
        elif len(cpv_str) > 8:
            cpv_str = cpv_str[:8]
        
        try:
            result['CPV_Level_2'] = int(cpv_str[:2]) if cpv_str[:2].isdigit() else 0
            result['CPV_Level_4'] = int(cpv_str[:4]) if cpv_str[:4].isdigit() else 0
            result['CPV_Level_6'] = int(cpv_str[:6]) if cpv_str[:6].isdigit() else 0
            result['CPV_Level_8'] = int(cpv_str) if cpv_str.isdigit() else 0
        except:
            pass
        
        return result
    
    def _clean_text(self, text):
        """Clean and normalize text."""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip
        text = text.strip()
        
        return text
    
    def _combine_text(self, data):
        """Combine all text fields for analysis."""
        parts = []
        
        if data.get('Title_Clean'):
            parts.append(f"Title: {data['Title_Clean']}")
        if data.get('Object_Description_Clean'):
            parts.append(f"Description: {data['Object_Description_Clean']}")
        if data.get('Award_Criteria_Clean'):
            parts.append(f"Award Criteria: {data['Award_Criteria_Clean']}")
        if data.get('Conditions_Clean'):
            parts.append(f"Conditions: {data['Conditions_Clean']}")
        
        return " | ".join(parts)
    
    # ==========================================
    # STEP 2: FEATURE ENGINEERING
    # ==========================================
    
    def engineer_features(self, preprocessed_data):
        """
        Step 2: Engineer features for model.
        
        Args:
            preprocessed_data: dict from preprocess()
            
        Returns:
            numpy array: Feature vector for model
        """
        # 2.1 Generate Text Embeddings
        model = self._load_sentence_model()
        combined_text = preprocessed_data.get('Combined_Text', '')
        
        if combined_text:
            embeddings = model.encode([combined_text], convert_to_numpy=True)[0]
        else:
            embeddings = np.zeros(768)  # all-mpnet-base-v2 has 768 dimensions
        
        # 2.2 Create Numerical Features
        numerical = self._create_numerical_features(preprocessed_data)
        
        # 2.3 Combine All Features
        feature_vector = np.concatenate([embeddings, numerical])
        
        return feature_vector
    
    def _create_numerical_features(self, data):
        """Create numerical features from preprocessed data."""
        # Default values for missing data
        estimated_value = data.get('Estimated_Value_Numeric', 0) or 0
        log_value = data.get('Log_Estimated_Value', 0) or 0
        
        features = np.array([
            estimated_value,
            log_value,
            0,  # Value_Z_Score_CPV - requires historical data
            data.get('Title_Length', 0),
            data.get('Description_Length', 0),
            data.get('Word_Count', 0),
            data.get('Combined_Text_Length', 0),
            1,  # Authority_Frequency - new authority
            1,  # CPV_Frequency - requires historical data
            data.get('CPV_Level_2', 0),
            data.get('CPV_Level_4', 0),
            estimated_value / (data.get('Word_Count', 1) + 1),  # Value_Per_Word
            data.get('Description_Length', 0) / (data.get('Word_Count', 1) + 1),  # Complexity_Ratio
        ], dtype=np.float64)
        
        # Scale if scaler exists
        if self.scaler:
            try:
                features = self.scaler.transform(features.reshape(1, -1))[0]
            except:
                pass  # Use unscaled if error
        
        return features
    
    # ==========================================
    # STEP 3: ANOMALY EVALUATION
    # ==========================================
    
    def evaluate(self, data):
        """
<<<<<<< HEAD
        Evaluate a tender for anomalies.
        
        Args:
            data: dict with tender columns (raw input)
            
        Returns:
            dict: {
                'anomaly_score': float (0-1, higher = more anomalous),
                'is_anomaly': bool,
                'category': str (EXTREME/HIGH/MEDIUM/LOW/NORMAL),
                'explanation': str
            }
=======
        Evaluate a tender for anomalies using an ensemble of models.
>>>>>>> bd1274c (Added Chat and rafactored code)
        """
        # Validate input
        is_valid, missing = self.validate_input(data)
        if not is_valid:
            return {
                'anomaly_score': None,
                'is_anomaly': None,
                'category': 'ERROR',
                'explanation': f"Missing required columns: {', '.join(missing)}"
            }
        
        # Preprocess
        preprocessed = self.preprocess(data)
        
        # Engineer features
        features = self.engineer_features(preprocessed)
<<<<<<< HEAD
        
        # Run Isolation Forest
        if self.isolation_forest is None:
            return self._evaluate_without_model(preprocessed)
        
        try:
            from sklearn.preprocessing import MinMaxScaler
            
            # Get raw score (lower = more anomalous)
            raw_score = self.isolation_forest.decision_function(features.reshape(1, -1))[0]
            
            # Normalize to 0-1 (1 = most anomalous)
            # Note: decision_function returns negative for anomalies
            # We invert so higher = more anomalous
            anomaly_score = max(0, min(1, 0.5 - raw_score))
            
            # Get prediction (-1 = anomaly, 1 = normal)
            prediction = self.isolation_forest.predict(features.reshape(1, -1))[0]
            is_anomaly = prediction == -1
            
            # Categorize
            category = self._categorize_score(anomaly_score)
            
            # Generate explanation
            explanation = self._generate_explanation(preprocessed, anomaly_score, is_anomaly)
            
            return {
                'anomaly_score': round(anomaly_score, 4),
                'is_anomaly': is_anomaly,
                'category': category,
                'explanation': explanation
            }
        
=======
        X = features.reshape(1, -1)
        
        # If no models are loaded, use heuristics
        if all(m is None for m in [self.isolation_forest, self.lof, self.oc_svm, self.autoencoder]):
            return self._evaluate_without_model(preprocessed)
        
        try:
            # 1. Get individual scores
            scores = {}
            flagged = {}
            model_explanations = {}
            
            # Isolation Forest
            if self.isolation_forest:
                raw_score = self.isolation_forest.decision_function(X)[0]
                scores['iso'] = max(0, min(1, 0.5 - raw_score))
                flagged['iso'] = self.isolation_forest.predict(X)[0] == -1
                if flagged['iso']:
                    model_explanations['iso'] = "Global outlier detection: Features deviate significantly from the training distribution."
                elif scores['iso'] > 0.4:
                    model_explanations['iso'] = "Suspicious global patterns detected."
            
            # LOF
            if self.lof:
                raw_lof = self.lof.decision_function(X)[0]
                scores['lof'] = max(0, min(1, 0.5 - raw_lof))
                flagged['lof'] = self.lof.predict(X)[0] == -1
                if flagged['lof']:
                    model_explanations['lof'] = "Local density outlier: Significantly different from tenders in similar categories/regions."
                elif scores['lof'] > 0.4:
                    model_explanations['lof'] = "Slightly unusual compared to nearby neighbors."
                
            # One-Class SVM
            if self.oc_svm:
                raw_svm = self.oc_svm.decision_function(X)[0]
                scores['svm'] = max(0, min(1, 0.5 - raw_svm))
                flagged['svm'] = self.oc_svm.predict(X)[0] == -1
                if flagged['svm']:
                    model_explanations['svm'] = "Boundary violation: Data point falls outside the learned 'normal' cluster boundary."
                elif scores['svm'] > 0.4:
                    model_explanations['svm'] = "Close to the boundary of normality."
                
            # Autoencoder
            if TORCH_AVAILABLE and self.autoencoder:
                embeddings = X[:, :768]
                with torch.no_grad():
                    X_torch = torch.FloatTensor(embeddings)
                    reconstructed = self.autoencoder(X_torch)
                    mse = torch.mean((X_torch - reconstructed) ** 2, dim=1).item()
                    scores['ae'] = max(0, min(1, mse * 500))
                    if scores['ae'] > 0.5:
                        model_explanations['ae'] = f"High reconstruction error (MSE: {mse:.6f}): Neural network failed to compress/rebuild this pattern safely."
            
            # Statistical (heuristic)
            stat_score, stat_reasons = self._get_statistical_score_details(preprocessed)
            scores['stat'] = stat_score
            if stat_reasons:
                model_explanations['stat'] = " | ".join(stat_reasons)
            
            # 2. Compute Ensemble Score
            # ... (weights logic remains the same)
            
            # 2. Compute Ensemble Score
            # Define weights (from research)
            # Optimized weights based on hyper-ensemble analysis
            # Golden Combination: ISO + LOF + AE
            weights = {
                'iso': 0.33,
                'lof': 0.33,
                'ae': 0.34,
                'svm': 0.0,
                'stat': 0.0
            }
            
            # Redistribute weights if models are missing
            available_weights = 0
            for key in weights:
                if key == 'stat' or (key == 'iso' and self.isolation_forest) or \
                   (key == 'lof' and self.lof) or (key == 'svm' and self.oc_svm) or \
                   (key == 'ae' and self.autoencoder):
                    available_weights += weights[key]
                else:
                    weights[key] = 0
            
            # Normalize weights to sum to 1
            if available_weights > 0:
                for key in weights:
                    weights[key] /= available_weights
            
            # Compute weighted ensemble score
            ensemble_score = sum(scores[k] * weights[k] for k in weights)
            
            # Determine if anomaly (using consensus or optimized threshold)
            # Threshold adjusted to 0.60 to block MEDIUM+ risk tenders per user request
            is_anomaly = ensemble_score >= 0.60 or sum(flagged.values()) >= 2
            
            # Categorize
            category = self._categorize_score(ensemble_score)
            
            # Generate explanation
            explanation = self._generate_ensemble_explanation(preprocessed, scores, flagged, ensemble_score)
            
            return {
                'anomaly_score': round(float(ensemble_score), 4),
                'is_anomaly': bool(is_anomaly),
                'category': str(category),
                'explanation': str(explanation),
                'model_details': {
                    k: {
                        'score': round(float(scores[k]), 4),
                        'flagged': bool(flagged.get(k, False)),
                        'reason': str(model_explanations.get(k, "Normal behavior detected."))
                    } for k in scores if scores.get(k, 0) > 0 or flagged.get(k, False)
                }
            }
            
>>>>>>> bd1274c (Added Chat and rafactored code)
        except Exception as e:
            return {
                'anomaly_score': None,
                'is_anomaly': None,
                'category': 'ERROR',
<<<<<<< HEAD
                'explanation': f"Error during evaluation: {str(e)}"
            }
=======
                'explanation': f"Error during ensemble evaluation: {str(e)}"
            }

    def _get_statistical_score_details(self, preprocessed):
        """Heuristic statistical score with detailed reasons."""
        score = 0.0
        reasons = []
        value = preprocessed.get('Estimated_Value_Numeric', 0) or 0
        word_count = preprocessed.get('Word_Count', 0)
        
        # High value/Short description mismatch
        if value > 1e7 and word_count < 100:
            score += 0.4
            reasons.append(f"Value-Description mismatch: €{value:,.0f} budget but very short description.")
        elif value > 1e6 and word_count < 50:
            score += 0.3
            reasons.append(f"Suspiciously brief description for €{value:,.0f} budget.")
            
        # Extreme values
        if value > 1e9:
            score += 0.5
            reasons.append(f"Unusually high budget: €{value:,.0f}")
        elif value > 1e8:
            score += 0.3
            reasons.append(f"Very high budget: €{value:,.0f}")
        
        return min(1.0, score), reasons

    def _generate_ensemble_explanation(self, preprocessed, scores, flagged, final_score):
        """Generate a summarized explanation based on ensemble results."""
        reasons = []
        
        # Specific patterns
        value = preprocessed.get('Estimated_Value_Numeric', 0) or 0
        word_count = preprocessed.get('Word_Count', 0)
        
        # Priority 1: High-level anomaly detection findings
        if final_score > 0.8:
            reasons.append("Extreme structural non-compliance and suspicious data patterns detected.")
        elif final_score > 0.55:
            reasons.append("Significant deviations from standard procurement data formats.")

        # Priority 2: Concrete red flags
        if value > 1e8:
            reasons.append(f"Unusually high valuation (approx. €{value:,.0f}) compared to normal database entries.")
        if word_count < 50:
            reasons.append(f"Abnormally brief description (only {word_count} words), lacking required professional detail.")
            
        if not reasons:
            if final_score > 0.55:
                reasons.append("System identified complex anomalous patterns in the tender structure.")
            else:
                reasons.append("No significant anomalies detected; tender follows standard patterns.")
                
        # Clean up and return the primary reason
        return " | ".join(reasons)
>>>>>>> bd1274c (Added Chat and rafactored code)
    
    def _evaluate_without_model(self, preprocessed):
        """Fallback evaluation using statistical heuristics when no model is loaded."""
        anomaly_score = 0.0
        reasons = []
        
        # Check for suspicious patterns
        value = preprocessed.get('Estimated_Value_Numeric', 0) or 0
        word_count = preprocessed.get('Word_Count', 0)
        title_length = preprocessed.get('Title_Length', 0)
        
        # Very high value
        if value > 1e9:  # Over 1 billion
            anomaly_score += 0.4
            reasons.append(f"Extremely high value: €{value:,.0f}")
        elif value > 1e8:  # Over 100 million
            anomaly_score += 0.2
            reasons.append(f"Very high value: €{value:,.0f}")
        
        # Very short description
        if word_count < 50:
            anomaly_score += 0.2
            reasons.append(f"Very short description ({word_count} words)")
        
        # Very long description (potential padding)
        if word_count > 10000:
            anomaly_score += 0.1
            reasons.append(f"Unusually long description ({word_count:,} words)")
        
        # Empty or very short title
        if title_length < 10:
            anomaly_score += 0.1
            reasons.append("Very short or empty title")
        
        # Invalid CPV
        if preprocessed.get('CPV_Level_2', 0) == 0:
            anomaly_score += 0.1
            reasons.append("Missing or invalid CPV code")
        
        # Normalize score
        anomaly_score = min(1.0, anomaly_score)
        category = self._categorize_score(anomaly_score)
        is_anomaly = anomaly_score >= 0.3
        
        explanation = " | ".join(reasons) if reasons else "No specific red flags detected"
        
        return {
            'anomaly_score': round(anomaly_score, 4),
            'is_anomaly': is_anomaly,
            'category': category,
            'explanation': f"(Heuristic evaluation - no trained model) {explanation}"
        }
    
    def _categorize_score(self, score):
        """Categorize anomaly score."""
        if score >= self.THRESHOLDS['EXTREME']:
            return 'EXTREME'
        elif score >= self.THRESHOLDS['HIGH']:
            return 'HIGH'
        elif score >= self.THRESHOLDS['MEDIUM']:
            return 'MEDIUM'
        elif score >= self.THRESHOLDS['LOW']:
            return 'LOW'
        else:
            return 'NORMAL'
    
    def _generate_explanation(self, data, score, is_anomaly):
        """Generate human-readable explanation for the anomaly score."""
        explanations = []
        
        if is_anomaly:
            explanations.append("Flagged by Isolation Forest model")
        
        # Value anomalies
        value = data.get('Estimated_Value_Numeric', 0) or 0
        if value > 1e7:  # > 10 million
            explanations.append(f"High value: €{value:,.0f}")
        elif value < 1000 and value > 0:
            explanations.append(f"Low value: €{value:,.0f}")
        
        # Text anomalies
        word_count = data.get('Word_Count', 0)
        if word_count < 50:
            explanations.append(f"Short description ({word_count} words)")
        elif word_count > 10000:
            explanations.append(f"Very long description ({word_count:,} words)")
        
        # CPV check
        if data.get('CPV_Level_2', 0) == 0:
            explanations.append("Missing CPV code")
        
        return " | ".join(explanations) if explanations else "No specific red flags"
    
    # ==========================================
    # MODEL TRAINING (Optional)
    # ==========================================
    
    def train_model(self, training_data, contamination=0.05):
        """
        Train the Isolation Forest model on historical data.
        
        Args:
            training_data: list of dicts or DataFrame with tender data
            contamination: expected proportion of anomalies
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        
        print("Training Isolation Forest model...")
        
        # Process all training data
        feature_list = []
        for tender in training_data:
            if isinstance(tender, dict):
                preprocessed = self.preprocess(tender)
                features = self.engineer_features(preprocessed)
                feature_list.append(features)
        
        X = np.array(feature_list)
        print(f"Training on {len(X)} samples with {X.shape[1]} features")
        
        # Fit scaler on numerical features (last 13 features)
        self.scaler = StandardScaler()
        self.scaler.fit(X[:, -13:])
        
        # Scale numerical features
        X[:, -13:] = self.scaler.transform(X[:, -13:])
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X)
        
        print("✓ Model training complete")
        
        # Save models
        self._save_models()
    
    def _save_models(self):
        """Save trained models to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        if self.isolation_forest:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.isolation_forest, f)
            print(f"✓ Saved Isolation Forest to {self.model_path}")
        
        if self.scaler:
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✓ Saved scaler to {self.scaler_path}")
