#!/usr/bin/env python
"""
Score simulation effort using simulation-based anchors instead of HC anchors.
This allows us to see how simulation conditions distribute within their own feature range.
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from effort.reference import build_reference, load_effort_config
from effort.scorer import score_subject

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def build_simulation_anchors(batch_dir, subject_glob, config, references):
    """Build anchor reference from simulation batch itself by collecting raw scores."""
    logger.info(f"Building simulation anchors from {batch_dir}")
    
    # Collect raw scores from all simulation subjects for each activity
    activity_scores = {}  # activity -> list of raw scores
    
    batch_path = Path(batch_dir)
    subject_dirs = sorted(batch_path.glob(subject_glob))
    logger.info(f"Found {len(subject_dirs)} simulation subjects")
    
    for subj_dir in subject_dirs:
        logger.info(f"  Scoring {subj_dir.name}")
        
        # Score this subject against HC references
        result = score_subject(subj_dir, config, references, subject_id=subj_dir.name)
        
        # Collect raw scores per activity
        for domain_result in result.domain_results.values():
            for activity_name, activity_result in domain_result.activity_results.items():
                if activity_name not in activity_scores:
                    activity_scores[activity_name] = []
                
                if activity_result.raw_score is not None:
                    activity_scores[activity_name].append(activity_result.raw_score)
    
    # Compute anchors from raw score distributions
    sim_anchors = {}
    for activity, scores in activity_scores.items():
        if len(scores) > 0:
            scores_arr = np.array(scores)
            anchor_minus_100 = np.percentile(scores_arr, 5)
            anchor_0 = np.median(scores_arr)
            anchor_100 = np.percentile(scores_arr, 95)
            sim_anchors[activity] = {
                'anchor_minus_100': anchor_minus_100,
                'anchor_0': anchor_0,
                'anchor_100': anchor_100,
                'n_scores': len(scores),
                'min': scores_arr.min(),
                'max': scores_arr.max(),
                'mean': scores_arr.mean(),
            }
            logger.info(
                f"  {activity}: anchor_-100={anchor_minus_100:.3f}, "
                f"anchor_0={anchor_0:.3f}, anchor_100={anchor_100:.3f}, n={len(scores)}"
            )
    
    return sim_anchors


def score_with_sim_anchors(batch_dir, subject_glob, sim_anchors, config, references, output_dir):
    """Score simulation subjects using simulation-based anchors."""
    logger.info(f"Scoring simulation subjects with sim anchors")
    
    batch_path = Path(batch_dir)
    subject_dirs = sorted(batch_path.glob(subject_glob))
    
    results = []
    details = []
    
    for subj_dir in subject_dirs:
        logger.info(f"Rescore {subj_dir.name} with sim anchors")
        subject_id = subj_dir.name
        row = {'subject_id': subject_id}
        
        # Score subject using HC references to get raw scores
        hc_result = score_subject(subj_dir, config, references, subject_id=subject_id)
        
        # Rescore with simulation anchors
        for domain_result in hc_result.domain_results.values():
            domain_name = domain_result.r4_label
            domain_scores = []
            
            for activity_name, activity_result in domain_result.activity_results.items():
                if activity_result.raw_score is not None and activity_name in sim_anchors:
                    raw_score = activity_result.raw_score
                    anchor_minus_100 = sim_anchors[activity_name]['anchor_minus_100']
                    anchor_0 = sim_anchors[activity_name]['anchor_0']
                    anchor_100 = sim_anchors[activity_name]['anchor_100']
                    
                    # Centered normalization to [-100, +100] using simulation anchors.
                    if raw_score >= anchor_0 and anchor_100 > anchor_0:
                        effort = ((raw_score - anchor_0) / (anchor_100 - anchor_0)) * 100
                    elif raw_score < anchor_0 and anchor_0 > anchor_minus_100:
                        effort = -((anchor_0 - raw_score) / (anchor_0 - anchor_minus_100)) * 100
                    else:
                        effort = 0
                    
                    # Clip to [-100, 100]
                    effort = np.clip(effort, -100, 100)
                    domain_scores.append(effort)
                    
                    details.append({
                        'subject_id': subject_id,
                        'domain': domain_name,
                        'activity': activity_name,
                        'raw_score': raw_score,
                        'anchor_minus_100': anchor_minus_100,
                        'anchor_0': anchor_0,
                        'anchor_100': anchor_100,
                        'sim_min': sim_anchors[activity_name]['min'],
                        'sim_max': sim_anchors[activity_name]['max'],
                        'effort_score': effort,
                    })
            
            if domain_scores:
                row[f'{domain_name}_effort'] = np.median(domain_scores)
            else:
                row[f'{domain_name}_effort'] = np.nan
        
        results.append(row)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / 'effort_scores_sim_anchors.csv', index=False)
    logger.info(f"Saved effort scores to {output_path / 'effort_scores_sim_anchors.csv'}")
    
    details_df = pd.DataFrame(details)
    details_df.to_csv(output_path / 'effort_details_sim_anchors.csv', index=False)
    logger.info(f"Saved effort details to {output_path / 'effort_details_sim_anchors.csv'}")
    
    # Print summary
    print("\n=== Effort Scores with Simulation Anchors ===\n")
    print(results_df.to_string(index=False))
    
    return results_df, details_df


def main():
    parser = argparse.ArgumentParser(description='Score simulation effort using simulation-based anchors')
    parser.add_argument('--patient-batch-dir', required=True, help='Path to simulation batch directory')
    parser.add_argument('--hc-batch-dir', required=True, help='Path to HC batch directory')
    parser.add_argument('--patient-subject-glob', default='sim_*_[2-5]', help='Glob pattern for simulation subjects')
    parser.add_argument('--hc-subject-glob', default='sub_*', help='Glob pattern for HC subjects')
    parser.add_argument('--config', default='config/effort_config.yaml', help='Path to effort config')
    parser.add_argument('--output-dir', default='output/effort/sim_anchors', help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    config = load_effort_config(args.config)
    
    # Build HC reference
    logger.info(f"Building HC reference from {args.hc_batch_dir}")
    references = build_reference(
        hc_batch_dir=Path(args.hc_batch_dir),
        config=config,
        subject_glob=args.hc_subject_glob
    )
    
    # Build simulation anchors
    sim_anchors = build_simulation_anchors(
        args.patient_batch_dir,
        args.patient_subject_glob,
        config,
        references
    )
    
    # Score with simulation anchors
    results_df, details_df = score_with_sim_anchors(
        args.patient_batch_dir,
        args.patient_subject_glob,
        sim_anchors,
        config,
        references,
        args.output_dir
    )


if __name__ == '__main__':
    main()
