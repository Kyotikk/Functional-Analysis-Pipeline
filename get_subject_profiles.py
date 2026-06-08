import pandas as pd

df = pd.read_csv('output/correlation/run2/combined_analysis.csv')

# Best pair: sub_0303 vs sub_ei (both high capacity stage but very different effort)
for subj in ['sub_0303', 'sub_ei']:
    print(f'\n{subj}:')
    subj_df = df[df['subject_id'] == subj].dropna(subset=['effort', 'r4_stage', 'capacity_stage'])
    print(subj_df[['domain', 'capacity_stage', 'effort', 'r4_stage']].to_string(index=False))
    cap_mean = subj_df['capacity_stage'].mean()
    eff_mean = subj_df['effort'].mean()
    print(f'Mean capacity: {cap_mean:.1f}')
    print(f'Mean effort: {eff_mean:.1f}')
