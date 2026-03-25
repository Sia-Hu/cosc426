import pandas as pd
from itertools import product

MALE_NAMES = [
    "John", "David"
]

FEMALE_NAMES = [
    "Mary", "Sarah"
]

def create_frequency_dataset(df_verbs):
    
    data = []
    pair_id = 1
    
    # Create all possible name combinations
    all_names = MALE_NAMES + FEMALE_NAMES
    
    for _, verb_row in df_verbs.iterrows():
        verb = verb_row['verb']
        freq_bin = verb_row['freq_bin']
        
        # Generate all possible combinations of NP1 and NP2
        for np1, np2 in product(all_names, repeat=2):
            if np1 != np2:  # Avoid same name combinations
                
                # Determine expected pronoun based on NP1's gender
                if np1 in MALE_NAMES:
                    expected_pronoun = "he"
                    # For unexpected: use a female name with "he" (mismatch)
                    unexpected_names = FEMALE_NAMES
                else:
                    expected_pronoun = "she"
                    # For unexpected: use a male name with "she" (mismatch)
                    unexpected_names = MALE_NAMES
                
                # Create pairs with each possible unexpected NP1
                for unexpected_np1 in unexpected_names:
                    
                    if unexpected_np1 == np2:
                        continue
                
                    expected_sentence = f"{np1} {verb} {np2} because {expected_pronoun}"
                    unexpected_sentence = f"{unexpected_np1} {verb} {np2} because {expected_pronoun}"
                    
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
                        'frequency': verb_row['frequency'],
                        'NP1': np1,
                        'NP2': np2,
                        'pronoun': expected_pronoun
                    })
                    
                    # Unexpected
                    data.append({
                        'sentid': pair_id * 2,  # 2, 4, 6, 8, ...
                        'pairid': pair_id,
                        'comparison': 'unexpected',
                        'sentence': unexpected_sentence,
                        'ROI': str(roi),
                        'verb': verb,
                        'freq_bin': freq_bin,
                        'frequency': verb_row['frequency'],
                        'NP1': unexpected_np1,
                        'NP2': np2,
                        'pronoun': expected_pronoun
                    })
                    
                    pair_id += 1
    
    dataset_df = pd.DataFrame(data)
    
    return dataset_df

def create_semantic_dataset(df_verbs):
    
    data = []
    pair_id = 1
    
    # Create all possible name combinations
    all_names = MALE_NAMES + FEMALE_NAMES
    
    data = []
    pair_id = 1  
    
    for _, verb_row in df_verbs.iterrows():
        verb = verb_row['verb']
        semantic_class_detailed = verb_row['semantic_class_detailed']
        for np1, np2 in product(all_names, repeat=2):
            if np1 != np2:  # Avoid same name combinations
                if semantic_class_detailed == 'AgP' or semantic_class_detailed == 'StimExp': 
                    # Determine expected pronoun
                    if np1 in MALE_NAMES:
                        expected_pronoun = "he"
                        # For unexpected: use a female name with "he" (mismatch)
                        unexpected_names = FEMALE_NAMES
                    else:
                        expected_pronoun = "she"
                        # For unexpected: use a male name with "she" (mismatch)
                        unexpected_names = MALE_NAMES
                    
                    for unexpected_name in unexpected_names:
                    
                        if unexpected_name == np2:
                            continue
                        
                        expected_sentence = f"{np1} {verb} {np2} because {expected_pronoun}"
                        unexpected_sentence = f"{unexpected_name} {verb} {np2} because {expected_pronoun}"
                        
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
                        })
                
                        pair_id += 1

                else:
                    # Determine expected pronoun
                    if np2 in MALE_NAMES:
                        expected_pronoun = "he"
                        # For unexpected: use a female name with "he" (mismatch)
                        unexpected_names = FEMALE_NAMES
                    else:
                        expected_pronoun = "she"
                        # For unexpected: use a male name with "she" (mismatch)
                        unexpected_names = MALE_NAMES
                    for unexpected_name in unexpected_names:
                    
                        if unexpected_name == np1:
                            continue
                        
                        expected_sentence = f"{np1} {verb} {np2} because {expected_pronoun}"
                        unexpected_sentence = f"{np1} {verb} {unexpected_name} because {expected_pronoun}"
                        
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
                        })
                
                        pair_id += 1
    
    dataset_df = pd.DataFrame(data)
    
    return dataset_df

def save_dataset(dataset_df, filename):
    minimalPair_columns = ['sentid', 'pairid', 'comparison', 'sentence', 'ROI']
    additional_columns = [col for col in dataset_df.columns if col not in minimalPair_columns]
    all_columns = minimalPair_columns + additional_columns
    
    output_df = dataset_df[all_columns].copy()
    output_df.to_csv(filename, sep='\t', index=False)
    
    print(f"Saved dataset to: {filename}")
    
    return filename