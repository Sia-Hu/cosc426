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
    
    for _, verb_row in df_verbs.iterrows():
        verb = verb_row['verb']
        freq_bin = verb_row['freq_bin']
        
        # Create same-gender pairs (expected)
        # Male-Male combinations
        for np1, np2 in product(MALE_NAMES, repeat=2):
            if np1 != np2:  # Avoid same name combinations
                expected_pronoun = "he"
                
                # Find a different-gender pair for unexpected
                for female_np in FEMALE_NAMES:
                    expected_sentence = f"{np1} {verb} {np2} because {expected_pronoun}"
                    unexpected_sentence = f"{female_np} {verb} {np2} because {expected_pronoun}"
                    
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
                    
                    # Unexpected (different gender)
                    data.append({
                        'sentid': pair_id * 2,  # 2, 4, 6, 8, ...
                        'pairid': pair_id,
                        'comparison': 'unexpected',
                        'sentence': unexpected_sentence,
                        'ROI': str(roi),
                        'verb': verb,
                        'freq_bin': freq_bin,
                        'frequency': verb_row['frequency'],
                        'NP1': np1,
                        'NP2': female_np,
                        'pronoun': expected_pronoun
                    })
                    
                    pair_id += 1
        
        # Female-Female combinations
        for np1, np2 in product(FEMALE_NAMES, repeat=2):
            if np1 != np2:  # Avoid same name combinations
                expected_pronoun = "she"
                
                # Find a different-gender pair for unexpected
                for male_np in MALE_NAMES:
                    expected_sentence = f"{np1} {verb} {np2} because {expected_pronoun}"
                    unexpected_sentence = f"{male_np} {verb} {np2} because {expected_pronoun}"
                    
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
                    
                    # Unexpected (different gender)
                    data.append({
                        'sentid': pair_id * 2,  # 2, 4, 6, 8, ...
                        'pairid': pair_id,
                        'comparison': 'unexpected',
                        'sentence': unexpected_sentence,
                        'ROI': str(roi),
                        'verb': verb,
                        'freq_bin': freq_bin,
                        'frequency': verb_row['frequency'],
                        'NP1': np1,
                        'NP2': male_np,
                        'pronoun': expected_pronoun
                    })
                    
                    pair_id += 1
    
    dataset_df = pd.DataFrame(data)
    
    return dataset_df

def create_semantic_dataset(df_verbs):
    
    data = []
    pair_id = 1
    
    for _, verb_row in df_verbs.iterrows():
        verb = verb_row['verb']
        semantic_class_detailed = verb_row['semantic_class_detailed']
        
        if semantic_class_detailed == 'AgP' or semantic_class_detailed == 'StimExp':
            # For AgP and StimExp: pronoun refers to NP1
            
            # Male NP1 cases (pronoun = "he")
            for np1 in MALE_NAMES:
                # Expected: same gender (Male NP2)
                for np2 in MALE_NAMES:
                    if np1 != np2:
                        expected_pronoun = "he"
                        
                        # Unexpected: different gender (Female NP1)
                        for female_np in FEMALE_NAMES:
                            expected_sentence = f"{np1} {verb} {np2} because {expected_pronoun}"
                            unexpected_sentence = f"{female_np} {verb} {np2} because {expected_pronoun}"
                            
                            roi = len(expected_sentence.split()) - 1

                            # Add the minimal pair
                            data.append({
                                'sentid': pair_id * 2 - 1,
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
                                'sentid': pair_id * 2,
                                'pairid': pair_id,
                                'comparison': 'unexpected', 
                                'sentence': unexpected_sentence,
                                'ROI': str(roi),
                                'verb': verb,
                                'semantic': verb_row['semantic_class'],
                                'semantic_class_detailed': verb_row['semantic_class_detailed']
                            })
                    
                            pair_id += 1
            
            # Female NP1 cases (pronoun = "she")
            for np1 in FEMALE_NAMES:
                # Expected: same gender (Female NP2)
                for np2 in FEMALE_NAMES:
                    if np1 != np2:
                        expected_pronoun = "she"
                        
                        # Unexpected: different gender (Male NP1)
                        for male_np in MALE_NAMES:
                            expected_sentence = f"{np1} {verb} {np2} because {expected_pronoun}"
                            unexpected_sentence = f"{male_np} {verb} {np2} because {expected_pronoun}"
                            
                            roi = len(expected_sentence.split()) - 1

                            # Add the minimal pair
                            data.append({
                                'sentid': pair_id * 2 - 1,
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
                                'sentid': pair_id * 2,
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
            # For other semantic classes: pronoun refers to NP2
            
            # Male NP2 cases (pronoun = "he")
            for np2 in MALE_NAMES:
                # Expected: same gender (Male NP1)
                for np1 in MALE_NAMES:
                    if np1 != np2:
                        expected_pronoun = "he"
                        
                        # Unexpected: different gender (Female NP2)
                        for female_np in FEMALE_NAMES:
                            expected_sentence = f"{np1} {verb} {np2} because {expected_pronoun}"
                            unexpected_sentence = f"{np1} {verb} {female_np} because {expected_pronoun}"
                            
                            roi = len(expected_sentence.split()) - 1

                            # Add the minimal pair
                            data.append({
                                'sentid': pair_id * 2 - 1,
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
                                'sentid': pair_id * 2,
                                'pairid': pair_id,
                                'comparison': 'unexpected', 
                                'sentence': unexpected_sentence,
                                'ROI': str(roi),
                                'verb': verb,
                                'semantic': verb_row['semantic_class'],
                                'semantic_class_detailed': verb_row['semantic_class_detailed']
                            })
                    
                            pair_id += 1
            
            # Female NP2 cases (pronoun = "she")
            for np2 in FEMALE_NAMES:
                # Expected: same gender (Female NP1)
                for np1 in FEMALE_NAMES:
                    if np1 != np2:
                        expected_pronoun = "she"
                        
                        # Unexpected: different gender (Male NP2)
                        for male_np in MALE_NAMES:
                            expected_sentence = f"{np1} {verb} {np2} because {expected_pronoun}"
                            unexpected_sentence = f"{np1} {verb} {male_np} because {expected_pronoun}"
                            
                            roi = len(expected_sentence.split()) - 1

                            # Add the minimal pair
                            data.append({
                                'sentid': pair_id * 2 - 1,
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
                                'sentid': pair_id * 2,
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