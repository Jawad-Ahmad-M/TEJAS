import os
import sys
import numpy as np

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tenders.ml.evaluator import TenderAnomalyEvaluator

def test_evaluation():
    print("Testing TenderAnomalyEvaluator locally...")
    
    # Sample data mimicking the expected input
    sample_tender = {
        'Title': 'Procurement of medical equipment for hospital',
        'Authority': 'Ministry of Health',
        'Object_Description': 'The hospital requires 50 new ventilators and heart rate monitors for the intensive care unit.',
        'CPV': '33100000', # Medical equipments
        'Estimated_Value': '500000 EUR',
        'Award_Criteria': 'Lowest price and technical specifications',
        'Conditions': 'Must have at least 5 years of experience in the field.'
    }
    
    try:
        evaluator = TenderAnomalyEvaluator()
        
        print("\n--- Running Evaluation ---")
        result = evaluator.evaluate(sample_tender)
        
        print(f"\nResults:")
        print(f"  Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"  Is Anomaly: {result['is_anomaly']}")
        print(f"  Category: {result['category']}")
        print(f"  Summary: {result['explanation']}")
        
        if 'model_details' in result:
            print(f"\n  Model Breakthrough:")
            for model_id, details in result['model_details'].items():
                flag_str = "[!] " if details['flagged'] else "    "
                name_map = {
                    'iso': 'Isolation Forest',
                    'lof': 'Local Outlier Factor',
                    'svm': 'One-Class SVM',
                    'ae': 'Autoencoder (Deep Learning)',
                    'stat': 'Statistical Heuristics'
                }
                display_name = name_map.get(model_id, model_id.upper())
                print(f"    {flag_str}{display_name:25} | Score: {details['score']:.4f}")
                print(f"        Reason: {details['reason']}")
        
        if result['is_anomaly']:
            print("\n[!] WARNING: Tender flagged as anomalous.")
        else:
            print("\n[V] SUCCESS: Tender passed anomaly detection.")
            
    except Exception as e:
        print(f"\n[X] ERROR: Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluation()
