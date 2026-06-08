import pandas as pd
import numpy as np

df = pd.read_csv('output/correlation/run2/combined_analysis.csv')

print('=== Subjects with identical capacity but different effort (per domain) ===\n')

pairs = []
for domain in df['domain'].unique():
    domain_df = df[df['domain'] == domain].dropna(subset=['capacity_stage', 'effort'])
    
    for stage in sorted(domain_df['capacity_stage'].unique()):
        stage_df = domain_df[domain_df['capacity_stage'] == stage]
        
        if len(stage_df) >= 2:
            efforts = stage_df[['subject_id', 'effort']].sort_values('effort')
            diff = efforts['effort'].max() - efforts['effort'].min()
            if diff > 25:
                subj_low = efforts.iloc[0]['subject_id']
                subj_high = efforts.iloc[-1]['subject_id']
                eff_low = efforts.iloc[0]['effort']
                eff_high = efforts.iloc[-1]['effort']
                print(f'{domain} (Capacity Stage {int(stage)}):')
                print(f'  {subj_low}: effort = {eff_low:.1f}')
                print(f'  {subj_high}: effort = {eff_high:.1f}')
                print(f'  Difference: {diff:.1f}\n')
                pairs.append((subj_low, subj_high, domain, stage, eff_low, eff_high, diff))

# Find subject pair that appears most consistently
if pairs:
    print('\n=== Best candidate pairs (appear in multiple domains) ===\n')
    from collections import defaultdict
    pair_counts = defaultdict(list)
    for subj_low, subj_high, domain, stage, eff_low, eff_high, diff in pairs:
        key = tuple(sorted([subj_low, subj_high]))
        pair_counts[key].append((domain, stage, eff_low, eff_high, diff))
    
    for (s1, s2), domains in sorted(pair_counts.items(), key=lambda x: -len(x[1]))[:5]:
        print(f'{s1} vs {s2}: appears in {len(domains)} domain(s)')
        for domain, stage, eff_low, eff_high, diff in domains:
            print(f'  {domain}: effort diff = {diff:.1f}')
        print()
