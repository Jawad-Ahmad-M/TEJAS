import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import sys

# Ensure we can import from the tenders.ml package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from tenders.ml.evaluator import TenderAnomalyEvaluator
except ImportError:
    # Handle if run from inside ml directory
    from evaluator import TenderAnomalyEvaluator

def load_and_clean_data(file_path):
    """Load real tenders from Excel and clean them."""
    print(f"[1/6] Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    # Basic cleaning
    df = df.fillna({
        'Title': 'Untitled Tender',
        'Authority': 'Unknown Authority',
        'Object_Description': '',
        'CPV': '00000000',
        'Estimated_Value': '0',
        'Award_Criteria': '',
        'Conditions': ''
    })
    
    print(f"✓ Loaded {len(df)} real tenders.")
    return df

def generate_synthetic_anomalies(real_df, count=50):
    """Generate synthetic anomalous tenders based on real data."""
    print(f"[2/6] Generating {count} synthetic anomalies...")
    anomalies = []
    
    for _ in range(count):
        # Pick a random real tender as a base
        base = real_df.iloc[random.randint(0, len(real_df) - 1)].to_dict()
        
        # Introduce anomaly types
        type_choice = random.choice(['value', 'text', 'cpv', 'stat'])
        
        anomaly = base.copy()
        
        if type_choice == 'value':
            # Massive budget anomaly
            try:
                # Extract numeric value if possible
                val_str = str(base.get('Estimated_Value', '0'))
                nums = [float(s) for s in val_str.replace(',', '').split() if s.replace('.', '', 1).isdigit()]
                val = nums[0] if nums else 100000
                anomaly['Estimated_Value'] = f"{val * random.uniform(10, 100):.2f} EUR"
            except:
                anomaly['Estimated_Value'] = "999,999,999,999.99 EUR"
                
        elif type_choice == 'text':
            # Suspiciously short description or gibberish
            anomaly['Object_Description'] = "X" * random.randint(1, 10)
            anomaly['Title'] = "URGENT BUY"
            
        elif type_choice == 'cpv':
            # Mismatched or fake CPV
            anomaly['CPV'] = "99999999"
            anomaly['Object_Description'] = "This description matches nothing in the CPV category."
            
        elif type_choice == 'stat':
            # Budget vs word count mismatch
            anomaly['Estimated_Value'] = "500,000,000.00 EUR"
            anomaly['Object_Description'] = "Short desc."
            
        anomalies.append(anomaly)
        
    return pd.DataFrame(anomalies)

def run_evaluation(test_set, labels, threshold=0.45):
    """Evaluate all models individually and as an ensemble."""
    print("[3/6] Running evaluations...")
    evaluator = TenderAnomalyEvaluator()
    
    results = {
        'iso': [], 'lof': [], 'svm': [], 'ae': [], 'stat': [], 'ensemble': []
    }
    
    for _, row in test_set.iterrows():
        tender_data = {
            'Title': row.get('Title', ''),
            'Authority': row.get('Authority', ''),
            'Object_Description': row.get('Object_Description', ''),
            'CPV': row.get('CPV', ''),
            'Estimated_Value': row.get('Estimated_Value', ''),
            'Award_Criteria': row.get('Award_Criteria', ''),
            'Conditions': row.get('Conditions', '')
        }
        
        res = evaluator.evaluate(tender_data)
        
        # Individual model predictions (binarized by score)
        details = res.get('model_details', {})
        results['iso'].append(1 if details.get('iso', {}).get('score', 0) > threshold else 0)
        results['lof'].append(1 if details.get('lof', {}).get('score', 0) > threshold else 0)
        results['svm'].append(1 if details.get('svm', {}).get('score', 0) > threshold else 0)
        results['ae'].append(1 if details.get('ae', {}).get('score', 0) > threshold else 0)
        results['stat'].append(1 if details.get('stat', {}).get('score', 0) > threshold else 0)
        
        # Ensemble prediction
        results['ensemble'].append(1 if res.get('is_anomaly') else 0)
        
    return results

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(base_dir, 'ingested_tenders_test.xlsx')
    
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        return

    # 1 & 2. Load and Generate
    real_df = load_and_clean_data(test_file)
    # Take a subset if too large for quick testing
    real_sample = real_df.sample(min(100, len(real_df)))
    
    anomaly_df = generate_synthetic_anomalies(real_sample, count=len(real_sample))
    
    # Combine into labeled set
    # Real = 0, Anomaly = 1
    labels = [0] * len(real_sample) + [1] * len(anomaly_df)
    test_set = pd.concat([real_sample, anomaly_df], ignore_index=True)
    
    # 3. Evaluate
    results = run_evaluation(test_set, labels)
    
    # 4. Report
    print(f"\n[4/6] Report Generation")
    print("-" * 60)
    print(f"{'Model':25} | {'Acc':6} | {'Prec':6} | {'Recall':6} | {'F1':6}")
    print("-" * 60)
    
    metrics_summary = []
    
    for model_name, predictions in results.items():
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        display_name = {
            'iso': 'Isolation Forest',
            'lof': 'LOF',
            'svm': 'One-Class SVM',
            'ae': 'Autoencoder',
            'stat': 'Statistical',
            'ensemble': 'Ensemble (Final)'
        }.get(model_name, model_name)
        
        print(f"{display_name:25} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f}")
        metrics_summary.append({
            'model': display_name,
            'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1
        })
        
        # 5. Confusion Matrices
        plot_confusion_matrix(labels, predictions, display_name, os.path.join(base_dir, f"cm_{model_name}.png"))
        
    # 6. Combinations Analysis (Simulated)
    print("\n[5/6] Ensemble Combination Analysis")
    # We can calculate combinations of models by looking at their agreement
    # Here we show how "Agreement >= 2 models" would perform
    agreement_pred = []
    for i in range(len(test_set)):
        # Count how many models said 1 (excluding the 'ensemble' result itself)
        votes = sum([results[m][i] for m in ['iso', 'lof', 'svm', 'ae', 'stat']])
        agreement_pred.append(1 if votes >= 2 else 0)
        
    acc_comb = accuracy_score(labels, agreement_pred)
    print(f"Combination (Any 2+ Models)   | Accuracy: {acc_comb:.4f}")
    
    print(f"\n[6/6] Saved confusion matrix plots to {base_dir}")
    print("Done.")

if __name__ == "__main__":
    main()
