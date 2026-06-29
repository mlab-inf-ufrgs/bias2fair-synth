import pandas as pd
import numpy as np

datasets = ['compas', 'adult', 'diabetes']
generators = ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM']
scenarios = ['S3_2.0', 'S4', 'S5', 'S6']

for ds in datasets:
    df = pd.read_csv(f'outputs/{ds}_results.csv')
    df['mitigator'] = df['mitigator'].fillna('None')
    df['generator'] = df['generator'].fillna('None')
    
    print(f"\n==================== DATASET: {ds.upper()} ====================")
    for scen in scenarios:
        print(f"\n--- Scenario: {scen} ---")
        print(f"{'Generator':<18} | {'F1 Mean ± Std':<18} | {'DI Mean ± Std':<18}")
        print("-" * 60)
        scen_df = df[df['scenario'] == scen]
        for gen in generators:
            gen_df = scen_df[scen_df['generator'] == gen]
            if gen_df.empty:
                continue
            f1_mean, f1_std = gen_df['f1'].mean(), gen_df['f1'].std()
            di_mean, di_std = gen_df['disparate_impact'].mean(), gen_df['disparate_impact'].std()
            print(f"{gen:<18} | {f1_mean:.4f} ± {f1_std:.4f}   | {di_mean:.4f} ± {di_std:.4f}")
