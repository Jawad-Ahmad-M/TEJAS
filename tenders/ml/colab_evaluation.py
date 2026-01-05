# TENDER ANOMALY DETECTION EVALUATION (GOOGLE COLAB)
# =================================================

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Try to import torch for Autoencoder
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==========================================
# 1. TENDER ANOMALY EVALUATOR CLASS
# ==========================================

if TORCH_AVAILABLE:
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, input_dim)
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

class TenderAnomalyEvaluator:
    REQUIRED_COLUMNS = ['Title', 'Authority', 'Object_Description', 'CPV', 'Estimated_Value', 'Award_Criteria', 'Conditions']
    THRESHOLDS = {'EXTREME': 0.9, 'HIGH': 0.7, 'MEDIUM': 0.5, 'LOW': 0.3}
    
    def __init__(self, models_dir='/content/models'):
        self.models_dir = Path(models_dir)
        self.isolation_forest = None
        self.lof = None
        self.oc_svm = None
        self.autoencoder = None
        self.scaler = None
        self.sentence_model = None
        
        # Load models from specific Colab path
        self._load_models()
    
    def _load_models(self):
        paths = {
            'iso': self.models_dir / 'isolation_forest.pkl',
            'lof': self.models_dir / 'lof.pkl',
            'svm': self.models_dir / 'one_class_svm.pkl',
            'ae': self.models_dir / 'autoencoder.pth',
            'scaler': self.models_dir / 'scaler.pkl'
        }
        
        if paths['iso'].exists():
            with open(paths['iso'], 'rb') as f: self.isolation_forest = pickle.load(f)
        if paths['lof'].exists():
            with open(paths['lof'], 'rb') as f: self.lof = pickle.load(f)
        if paths['svm'].exists():
            with open(paths['svm'], 'rb') as f: self.oc_svm = pickle.load(f)
        if paths['scaler'].exists():
            with open(paths['scaler'], 'rb') as f: self.scaler = pickle.load(f)
        if TORCH_AVAILABLE and paths['ae'].exists():
            self.autoencoder = Autoencoder(input_dim=768)
            self.autoencoder.load_state_dict(torch.load(paths['ae'], map_location=torch.device('cpu')))
            self.autoencoder.eval()

    def _load_sentence_model(self):
        if self.sentence_model is None:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        return self.sentence_model

    def preprocess(self, data):
        processed = data.copy()
        processed['Estimated_Value_Numeric'] = self._parse_val(data.get('Estimated_Value', ''))
        processed.update(self._extract_cpv(data.get('CPV', '')))
        for col in ['Title', 'Object_Description', 'Award_Criteria', 'Conditions']:
            processed[col+'_Clean'] = self._clean(data.get(col, ''))
        processed['Combined_Text'] = " | ".join([f"{c}: {processed[c+'_Clean']}" for c in ['Title', 'Object_Description', 'Award_Criteria', 'Conditions'] if processed.get(c+'_Clean')])
        processed['Word_Count'] = len(processed['Combined_Text'].split())
        processed['Log_Estimated_Value'] = np.log1p(processed['Estimated_Value_Numeric']) if (processed.get('Estimated_Value_Numeric') and processed['Estimated_Value_Numeric']>0) else 0
        return processed

    def _parse_val(self, v):
        if not v or pd.isna(v): return 0
        v = re.sub(r'[^\d.]', '', str(v))
        try: return float(v)
        except: return 0

    def _extract_cpv(self, c):
        c = re.sub(r'[^0-9]', '', str(c)).ljust(8, '0')[:8]
        return {f'CPV_Level_{i}': int(c[:i]) if c[:i].isdigit() else 0 for i in [2,4,6]}

    def _clean(self, t): return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', str(t))).strip()

    def engineer_features(self, p):
        model = self._load_sentence_model()
        emb = model.encode([p.get('Combined_Text', '')])[0]
        num = np.array([p.get('Estimated_Value_Numeric',0), p.get('Log_Estimated_Value',0), 0, len(p.get('Title_Clean','')), len(p.get('Object_Description_Clean','')), p.get('Word_Count',0), len(p.get('Combined_Text','')), 1, 1, p.get('CPV_Level_2',0), p.get('CPV_Level_4',0), p.get('Estimated_Value_Numeric',0)/(p.get('Word_Count',0)+1), len(p.get('Object_Description_Clean',''))/(p.get('Word_Count',0)+1)])
        if self.scaler: num = self.scaler.transform(num.reshape(1, -1))[0]
        return np.concatenate([emb, num])

    def evaluate(self, data):
        p = self.preprocess(data)
        X = self.engineer_features(p).reshape(1, -1)
        scores, flagged = {}, {}
        
        if self.isolation_forest:
            scores['iso'] = max(0, min(1, 0.5 - self.isolation_forest.decision_function(X)[0]))
            flagged['iso'] = self.isolation_forest.predict(X)[0] == -1
        if self.lof:
            scores['lof'] = max(0, min(1, 0.5 - self.lof.decision_function(X)[0]))
            flagged['lof'] = self.lof.predict(X)[0] == -1
        if self.oc_svm:
            scores['svm'] = max(0, min(1, 0.5 - self.oc_svm.decision_function(X)[0]))
            flagged['svm'] = self.oc_svm.predict(X)[0] == -1
        if TORCH_AVAILABLE and self.autoencoder:
            with torch.no_grad():
                mse = torch.mean((torch.FloatTensor(X[:, :768]) - self.autoencoder(torch.FloatTensor(X[:, :768])))**2).item()
                scores['ae'] = max(0, min(1, mse * 500))
        
        # Simple weighted ensemble
        final_score = np.mean(list(scores.values())) if scores else 0
        return {
            'anomaly_score': final_score,
            'is_anomaly': final_score > 0.45 or sum(flagged.values()) >= 2,
            'model_scores': scores
        }

# ==========================================
# 2. DATA LOADING & TESTING
# ==========================================

def load_test_data(csv_path):
    """Load the high-quality pre-generated test dataset."""
    df = pd.read_csv(csv_path).fillna('')
    y_true = df['is_anomaly'].tolist()
    return df, y_true

def main():
    # SETUP: Update paths to match your Google Drive mount
    # 1. Mount Drive in Colab: from google.colab import drive; drive.mount('/content/drive')
    # 2. Upload your 'models' folder and 'high_quality_test_dataset.csv' to your Drive
    
    MODELS_PATH = '/content/drive/MyDrive/tender_models'
    DATASET_PATH = '/content/drive/MyDrive/high_quality_test_dataset.csv'
    
    if not os.path.exists(DATASET_PATH):
        print(f"FAILED: Test Dataset not found at {DATASET_PATH}")
        print("Please upload 'high_quality_test_dataset.csv' to your Google Drive.")
        return

    evaluator = TenderAnomalyEvaluator(models_dir=MODELS_PATH)
    test_df, y_true = load_test_data(DATASET_PATH)
    
    model_preds = {'iso': [], 'lof': [], 'svm': [], 'ae': [], 'ensemble': []}
    
    print("Running evaluations...")
    for _, row in test_df.iterrows():
        res = evaluator.evaluate(row.to_dict())
        m_scores = res['model_scores']
        for k in ['iso', 'lof', 'svm', 'ae']:
            if k in m_scores: model_preds[k].append(1 if m_scores[k] > 0.45 else 0)
            else: model_preds[k].append(0)
        model_preds['ensemble'].append(1 if res['is_anomaly'] else 0)

    # Metrics
    for m, y_pred in model_preds.items():
        if len(y_pred) == 0: continue
        print(f"\n--- {m.upper()} PERFORMANCE ---")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {m}")
        plt.show()

if __name__ == "__main__":
    main()
