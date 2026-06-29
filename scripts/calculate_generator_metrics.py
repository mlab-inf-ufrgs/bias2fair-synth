import pandas as pd

datasets = ['compas', 'adult', 'diabetes']
generators = ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM']

print("=========================================================================")
print("GENERATOR METRICS SUMMARY PER DATASET (SYNTHETIC SCENARIOS S3-S6)")
print("=========================================================================")

for ds in datasets:
    df = pd.read_csv(f'outputs/{ds}_results.csv')
    
    # Filter only synthetic scenarios (S3_1.5, S3_2.0, S3_3.0, S4, S5, S6)
    synth_df = df[df['scenario'].isin(['S3_1.5', 'S3_2.0', 'S3_3.0', 'S4', 'S5', 'S6'])]
    
    # Baseline S1 for comparison
    s1_df = df[df['scenario'] == 'S1']
    s1_f1_mean = s1_df['f1'].mean()
    s1_di_mean = s1_df['disparate_impact'].mean()
    
    print(f"\n=== DATASET: {ds.upper()} (S1 Baseline: F1={s1_f1_mean:.3f}, DI={s1_di_mean:.3f}) ===")
    print(f"{'Generator':<18} | {'F1 Mean ± Std':<18} | {'DI Mean ± Std':<18}")
    print("-" * 60)
    
    for gen in generators:
        gen_df = synth_df[synth_df['generator'] == gen]
        if gen_df.empty:
            continue
            
        f1_mean, f1_std = gen_df['f1'].mean(), gen_df['f1'].std()
        di_mean, di_std = gen_df['disparate_impact'].mean(), gen_df['disparate_impact'].std()
        
        print(f"{gen:<18} | {f1_mean:.3f} ± {f1_std:.3f}     | {di_mean:.3f} ± {di_std:.3f}")
