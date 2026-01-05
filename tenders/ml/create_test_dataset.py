import os
import pandas as pd
import numpy as np
import random
import re

def parse_numeric(val_str):
    """Attempt to extract a numeric value from string."""
    try:
        val_str = str(val_str).replace(',', '')
        nums = re.findall(r'[\d.]+', val_str)
        return float(nums[0]) if nums else 0
    except:
        return 0

def create_dataset():
    # Paths
    base_dir = r"c:\Users\User\Desktop\tender model\tenders\ml"
    excel_path = os.path.join(base_dir, 'ingested_tenders_test.xlsx')
    output_path = os.path.join(base_dir, 'high_quality_test_dataset.csv')
    
    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} not found.")
        return

    # 1. Load and Clean Real Tenders
    print("Loading real tenders...")
    df = pd.read_excel(excel_path)
    df = df.fillna('')
    
    # Select 150 real samples (use oversampling if data is scarce)
    real_samples = df.sample(150, replace=True if len(df) < 150 else False, random_state=42).copy()
    
    real_samples['is_anomaly'] = 0
    print(f"✓ Selected {len(real_samples)} real tenders.")

    # 2. Generate 150 Realistic Anomalies
    print("Generating 150 realistic anomalies...")
    anomalies = []
    
    # Use real tenders as templates for the anomalies
    templates = df.sample(150, replace=True, random_state=7).to_dict('records')
    
    anomaly_types = ['extreme_value', 'suspiciously_short', 'cpv_mismatch', 'gibberish_injection', 'complexity_mismatch']
    
    for i, base in enumerate(templates):
        a = base.copy()
        type_choice = anomaly_types[i % len(anomaly_types)]
        
        if type_choice == 'extreme_value':
            # Budget is 50x higher than normal for this type of tender
            orig_val = parse_numeric(base.get('Estimated_Value', 0))
            if orig_val == 0: orig_val = 500000
            a['Estimated_Value'] = f"{orig_val * random.uniform(50, 100):.2f} EUR"
            a['Anomaly_Reason'] = "Extreme Value Inflation"
            
        elif type_choice == 'suspiciously_short':
            # High value but almost no description
            a['Estimated_Value'] = "10,000,000.00 EUR"
            a['Object_Description'] = "Procurement of various items. Contact us for details ASAP."
            a['Award_Criteria'] = "Lowest price"
            a['Anomaly_Reason'] = "Suspiciously Brief Description"
            
        elif type_choice == 'cpv_mismatch':
            # Medical CPV for a Construction title
            a['CPV'] = "33600000" # Pharmaceutical products
            a['Title'] = "Construction of a new highway bridge crossing"
            a['Object_Description'] = "Civil engineering and bridge building services."
            a['Anomaly_Reason'] = "CPV and Theme Mismatch"
            
        elif type_choice == 'gibberish_injection':
            # Repeated text or placeholder strings
            a['Object_Description'] = (base.get('Object_Description', '')[:100] + 
                                       " [REDACTED] " * 50 + 
                                       " Lorem ipsum dolor sit amet " * 10)
            a['Conditions'] = "X" * 500
            a['Anomaly_Reason'] = "Nonsense/Gibberish Injection"
            
        elif type_choice == 'complexity_mismatch':
            # Very low value but 5000+ words of legal jargon
            a['Estimated_Value'] = "5.00 EUR"
            a['Object_Description'] = "This is a highly complex tender with multiple sub-clauses... " * 500
            a['Anomaly_Reason'] = "Value vs Complexity Mismatch"
            
        a['is_anomaly'] = 1
        anomalies.append(a)
        
    anomaly_df = pd.DataFrame(anomalies)
    print(f"✓ Generated 150 realistic anomalies.")

    # 3. Combine and Save
    final_df = pd.concat([real_samples, anomaly_df], ignore_index=True)
    
    # Shuffle the dataset
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    
    # Ensure columns match what evaluator expects
    cols = ['Title', 'Authority', 'Object_Description', 'CPV', 'Estimated_Value', 'Award_Criteria', 'Conditions', 'is_anomaly']
    # If any columns are missing in some rows, fill them
    for col in cols:
        if col not in final_df.columns:
            final_df[col] = ''
            
    final_df[cols].to_csv(output_path, index=False)
    print(f"✓ Dataset saved to {output_path}")
    print(f"Final Counts: {len(final_df)} rows. Anomalies: {final_df['is_anomaly'].sum()}.")

if __name__ == "__main__":
    create_dataset()
