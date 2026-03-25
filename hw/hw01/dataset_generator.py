import pandas as pd
import numpy as np
import random

MALE_NAMES = [
    "John", "David", "Michael", "James", "Robert", 
    "William", "Richard", "Thomas", "Mark", "Daniel"
]

FEMALE_NAMES = [
    "Mary", "Lily", "Jennifer", "Linda", "Elizabeth",
    "Emily", "Susan", "Jessica", "Sarah", "Grace"
]

def create_frequency_dataset(df_verbs, seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    
    data = []
    pair_id = 1  
    
    for _, verb_row in df_verbs.iterrows():
        verb = verb_row['verb']
        freq_bin = verb_row['freq_bin']
        
        male_name1, male_name2 = random.sample(MALE_NAMES, 2)
        female_name = random.choice(FEMALE_NAMES)
        
        
        # Expected: NP1 continuation (he/she refers to NP1)
        expected_sentence = f"{male_name1} {verb} {male_name2} because he"
        
        # Unexpected: NP2 continuation (he/she refers to NP2) 
        unexpected_sentence = f"{female_name} {verb} {male_name2} because he"
        
        roi = len(expected_sentence.split()) - 1

        # Add the minimal pair
        data.append({
            'sentid': pair_id * 2 - 1,  # 1, 3, 5, 7, ...
            'pairid': pair_id,
            'comparison': 'expected',
            'sentence': expected_sentence,
            'ROI': str(roi), 
            'verb': verb,
            'freq_bin': freq_bin,
            'frequency': verb_row['frequency']
        })
        
        # Unexpected (NP2 continuation)
        data.append({
            'sentid': pair_id * 2,  # 2, 4, 6, 8, ...
            'pairid': pair_id,
            'comparison': 'unexpected', 
            'sentence': unexpected_sentence,
            'ROI': str(roi),
            'verb': verb,
            'freq_bin': freq_bin,
            'frequency': verb_row['frequency'],
        })
        
        pair_id += 1
    
    dataset_df = pd.DataFrame(data)
    
    return dataset_df

def create_semantic_dataset(df_verbs, seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    
    data = []
    pair_id = 1  
    
    for _, verb_row in df_verbs.iterrows():
        verb = verb_row['verb']
        # ic = verb_row['ic']
        semantic_class_detailed = verb_row['semantic_class_detailed']
        
        male_name1, male_name2 = random.sample(MALE_NAMES, 2)
        female_name = random.choice(FEMALE_NAMES)
        
        
        # Expected: 
        expected_sentence = f"{male_name1} {verb} {male_name2} because he"
        
        # Unexpected: 
        if semantic_class_detailed == 'AgP' or semantic_class_detailed == 'StimExp':
            unexpected_sentence = f"{female_name} {verb} {male_name2} because he"
        else:  
            unexpected_sentence = f"{male_name1} {verb} {female_name} because he"
        
        roi = len(expected_sentence.split()) - 1

        # Add the minimal pair
        data.append({
            'sentid': pair_id * 2 - 1,  # 1, 3, 5, 7, ...
            'pairid': pair_id,
            'comparison': 'expected',
            'sentence': expected_sentence,
            'ROI': str(roi), 
            'verb': verb,
            'semantic': verb_row['semantic_class'],
            'semantic_class_detailed': verb_row['semantic_class_detailed']
            # 'ic': ic,
            # 'bias': verb_row['bias']
        })
        
        # Unexpected 
        data.append({
            'sentid': pair_id * 2,  # 2, 4, 6, 8, ...
            'pairid': pair_id,
            'comparison': 'unexpected', 
            'sentence': unexpected_sentence,
            'ROI': str(roi),
            'verb': verb,
            'semantic': verb_row['semantic_class'],
            'semantic_class_detailed': verb_row['semantic_class_detailed']
            # 'ic': ic,
            # 'bias': verb_row['bias']
        })
        
        pair_id += 1
    
    dataset_df = pd.DataFrame(data)
    
    return dataset_df

def save_dataset(dataset_df, filename):
    
    # Select only the required columns for MinimalPair
    minimalPair_columns = ['sentid', 'pairid', 'comparison', 'sentence', 'ROI']
    additional_columns = [col for col in dataset_df.columns if col not in minimalPair_columns]
    all_columns = minimalPair_columns + additional_columns
    
    output_df = dataset_df[all_columns].copy()
    output_df.to_csv(filename, sep='\t', index=False)
    
    print(f"Saved dataset to: {filename}") 
    return filename