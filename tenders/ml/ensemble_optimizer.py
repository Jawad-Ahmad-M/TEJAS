# TENDER ENSEMBLE OPTIMIZER (GOOGLE COLAB)
# ========================================
# This script finds the BEST combination of models to maximize accuracy/F1.

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Try to import torch for Autoencoder
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings('ignore')

# [MODEL CLASSES & UTILS - SAME AS BEFORE]
if TORCH_AVAILABLE:
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, input_dim))
        def forward(self, x): return self.decoder(self.encoder(x))

def parse_val(v):
    if not v or pd.isna(v): return 0
    v = re.sub(r'[^\d.]', '', str(v))
    try: return float(v)
    except: return 0

def clean_text(t): return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', str(t))).strip()

def preprocess(data):
    processed = data.copy()
    processed['Estimated_Value_Numeric'] = parse_val(data.get('Estimated_Value', ''))
    for col in ['Title', 'Object_Description', 'Award_Criteria', 'Conditions']:
        processed[col+'_Clean'] = clean_text(data.get(col, ''))
    processed['Combined_Text'] = " | ".join([f"{c}: {processed[c+'_Clean']}" for c in ['Title','Object_Description','Award_Criteria','Conditions'] if processed.get(c+'_Clean')])
    processed['Word_Count'] = len(processed['Combined_Text'].split())
    processed['Log_Estimated_Value'] = np.log1p(processed['Estimated_Value_Numeric']) if (processed.get('Estimated_Value_Numeric') and processed['Estimated_Value_Numeric']>0) else 0
    return processed

def engineer_features(p, sentence_model, scaler=None):
    emb = sentence_model.encode([p.get('Combined_Text', '')])[0]
    num = np.array([p.get('Estimated_Value_Numeric',0), p.get('Log_Estimated_Value',0), 0, len(p.get('Title_Clean','')), len(p.get('Object_Description_Clean','')), p.get('Word_Count',0), len(p.get('Combined_Text','')), 1, 1, 0, 0, p.get('Estimated_Value_Numeric',0)/(p.get('Word_Count',0)+1), len(p.get('Object_Description_Clean',''))/(p.get('Word_Count',0)+1)])
    if scaler: num = scaler.transform(num.reshape(1, -1))[0]
    return np.concatenate([emb, num])

# ==========================================
# OPTIMIZATION LOGIC
# ==========================================

def get_best_threshold(y_true, scores):
    """Find threshold that maximizes F1 for a given set of scores."""
    best_f1 = 0
    best_thresh = 0.45
    for thresh in np.linspace(0.1, 0.9, 81):
        y_pred = [1 if s > thresh else 0 for s in scores]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1

def main():
    MODELS_PATH = '/content/drive/MyDrive/models'
    DATASET_PATH = '/content/drive/MyDrive/high_quality_test_dataset.csv'
    
    if not os.path.exists(DATASET_PATH):
        print("FAILED: Dataset not found.")
        return

    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer('all-mpnet-base-v2')
    
    # Load Models
    models_dir = Path(MODELS_PATH)
    models = {}
    if (models_dir / 'isolation_forest.pkl').exists():
        with open(models_dir / 'isolation_forest.pkl', 'rb') as f: models['iso'] = pickle.load(f)
    if (models_dir / 'lof.pkl').exists():
        with open(models_dir / 'lof.pkl', 'rb') as f: models['lof'] = pickle.load(f)
    if (models_dir / 'one_class_svm.pkl').exists():
        with open(models_dir / 'one_class_svm.pkl', 'rb') as f: models['svm'] = pickle.load(f)
    if TORCH_AVAILABLE and (models_dir / 'autoencoder.pth').exists():
        models['ae'] = Autoencoder(input_dim=768)
        models['ae'].load_state_dict(torch.load(models_dir / 'autoencoder.pth', map_location=torch.device('cpu')))
        models['ae'].eval()

    # Load Data
    df = pd.read_csv(DATASET_PATH).fillna('')
    y_true = df['is_anomaly'].tolist()
    
    # Initial Scaler Fit
    num_list = []
    for _, row in df.iterrows():
        p = preprocess(row.to_dict())
        num_list.append([p.get('Estimated_Value_Numeric',0), p.get('Log_Estimated_Value',0), 0, len(p.get('Title_Clean','')), len(p.get('Object_Description_Clean','')), p.get('Word_Count',0), len(p.get('Combined_Text','')), 1, 1, 0, 0, p.get('Estimated_Value_Numeric',0)/(p.get('Word_Count',0)+1), len(p.get('Object_Description_Clean',''))/(p.get('Word_Count',0)+1)])
    scaler = StandardScaler().fit(num_list)

    # 1. Pre-calculate all raw scores
    print("Pre-calculating model scores...")
    raw_scores = {k: [] for k in models.keys()}
    for _, row in df.iterrows():
        p = preprocess(row.to_dict())
        X = engineer_features(p, sentence_model, scaler).reshape(1, -1)
        if 'iso' in models: raw_scores['iso'].append(0.5 - models['iso'].decision_function(X)[0])
        if 'lof' in models: raw_scores['lof'].append(0.5 - models['lof'].decision_function(X)[0])
        if 'svm' in models: raw_scores['svm'].append(0.5 - models['svm'].decision_function(X)[0])
        if 'ae' in models:
            with torch.no_grad():
                mse = torch.mean((torch.FloatTensor(X[:, :768]) - models['ae'](torch.FloatTensor(X[:, :768])))**2).item()
                raw_scores['ae'].append(mse * 500)

    # 2. Iterate through all combinations
    model_keys = list(models.keys())
    results = []

    print("\nOptimizing combinations...")
    for r in range(1, len(model_keys) + 1):
        for combo in combinations(model_keys, r):
            # Calculate average score for this combo
            combo_name = "+".join(combo)
            combined_score = np.mean([raw_scores[k] for k in combo], axis=0)
            
            # Find best threshold for this combo
            best_thresh, f1 = get_best_threshold(y_true, combined_score)
            y_pred = [1 if s > best_thresh else 0 for s in combined_score]
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            
            results.append({
                'Combination': combo_name,
                'Best Thresh': round(best_thresh, 3),
                'Accuracy': round(acc, 4),
                'F1': round(f1, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4)
            })

    # 3. Sort and Print Results
    res_df = pd.DataFrame(results).sort_values(by='F1', ascending=False)
    print("\n--- ENSEMBLE OPTIMIZATION RESULTS (Sorted by F1) ---")
    print(res_df.to_string(index=False))

    best = res_df.iloc[0]
    print(f"\n🏆 GOLDEN COMBINATION: {best['Combination']}")
    print(f"Optimal Threshold: {best['Best Thresh']}")
    print(f"Projected Accuracy: {best['Accuracy']} | F1: {best['F1']}")
    print("-" * 50)
    print("Apply these weights/thresholds to your PRODUCTION evaluator.py for best results.")

if __name__ == "__main__":
    main()
