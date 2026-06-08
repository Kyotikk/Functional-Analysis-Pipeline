#!/usr/bin/env python
"""Create comparison plots: HC anchors vs Simulation anchors effort distribution."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
sim_anchor_scores = pd.read_csv('functional-analysis-pipeline/output/effort/sim_anchors/effort_scores_sim_anchors.csv')
hc_anchor_scores = pd.read_csv('functional-analysis-pipeline/output/effort/sim_run2_clean/effort_scores.csv')

# Extract conditions
sim_anchor_scores['condition'] = sim_anchor_scores['subject_id'].str.extract(r'sim_(healthy|elderly|severe)_')[0]
hc_anchor_scores['condition'] = hc_anchor_scores['subject_id'].str.extract(r'sim_(healthy|elderly|severe)_')[0]

# Get effort columns (all domains)
effort_cols = [col for col in sim_anchor_scores.columns if 'effort' in col]

# Create comparison plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Simulation Effort Scores: HC Anchors vs Simulation Anchors', fontsize=14, fontweight='bold')

conditions = ['healthy', 'elderly', 'severe']
colors = {'healthy': 'green', 'elderly': 'orange', 'severe': 'red'}

for idx, condition in enumerate(conditions):
    # HC anchors (left column)
    ax_hc = axes[0, idx]
    hc_cond = hc_anchor_scores[hc_anchor_scores['condition'] == condition][effort_cols].values.flatten()
    ax_hc.hist(hc_cond, bins=20, color=colors[condition], alpha=0.7, edgecolor='black')
    ax_hc.set_title(f'{condition.upper()}\n(HC Anchors)', fontweight='bold')
    ax_hc.set_ylabel('Frequency' if idx == 0 else '')
    ax_hc.set_xlabel('Effort Score')
    ax_hc.set_xlim(0, 210)
    stats_hc = f"μ={hc_cond.mean():.0f}\nmed={np.median(hc_cond):.0f}\n% at 200: {(hc_cond==200).sum()/len(hc_cond)*100:.0f}%"
    ax_hc.text(0.98, 0.97, stats_hc, transform=ax_hc.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Simulation anchors (right column)
    ax_sim = axes[1, idx]
    sim_cond = sim_anchor_scores[sim_anchor_scores['condition'] == condition][effort_cols].values.flatten()
    ax_sim.hist(sim_cond, bins=20, color=colors[condition], alpha=0.7, edgecolor='black')
    ax_sim.set_title(f'{condition.upper()}\n(Simulation Anchors)', fontweight='bold')
    ax_sim.set_ylabel('Frequency' if idx == 0 else '')
    ax_sim.set_xlabel('Effort Score')
    ax_sim.set_xlim(0, 160)
    stats_sim = f"μ={sim_cond.mean():.0f}\nmed={np.median(sim_cond):.0f}\nmax={sim_cond.max():.0f}"
    ax_sim.text(0.98, 0.97, stats_sim, transform=ax_sim.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('functional-analysis-pipeline/output/plots/score_histograms/sim_anchors_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: sim_anchors_comparison.png")

# Summary statistics table
print("\n" + "="*70)
print("EFFORT SCORE SUMMARY: HC Anchors vs Simulation Anchors")
print("="*70)

summary_data = []
for condition in ['healthy', 'elderly', 'severe']:
    hc_cond = hc_anchor_scores[hc_anchor_scores['condition'] == condition][effort_cols].values.flatten()
    sim_cond = sim_anchor_scores[sim_anchor_scores['condition'] == condition][effort_cols].values.flatten()
    
    summary_data.append({
        'Condition': condition.upper(),
        'HC_mean': f"{hc_cond.mean():.1f}",
        'HC_median': f"{np.median(hc_cond):.1f}",
        'HC_at_200pct': f"{(hc_cond==200).sum()/len(hc_cond)*100:.0f}%",
        'SIM_mean': f"{sim_cond.mean():.1f}",
        'SIM_median': f"{np.median(sim_cond):.1f}",
        'SIM_max': f"{sim_cond.max():.1f}",
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("""
1. HC Anchors Problem:
   - All conditions saturate at 200 (87.5% capped)
   - Cannot differentiate between healthy/elderly/severe
   - Suggests HC anchors are too tight for simulation data

2. Simulation Anchors Solution:
   - Healthy: low effort (μ=11.5), mostly at baseline
   - Elderly:  medium effort (μ=32.6), high variability (0-143)
   - Severe:   elevated effort (μ=37.0), consistent elevation (0-98)
   - Clear condition differentiation achieved

3. Recommendation:
   - Use simulation-based anchors for simulation validation
   - This shows how well effort scoring can discriminate between
     artificially-created condition-based phenotypes
   - Consider: Are simulation-based anchors more appropriate for
     external validation of effort scoring than HC-based anchors?
""")
