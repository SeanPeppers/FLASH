import pandas as pd
import glob
import os

results_dir = 'fl_results_hfl'
strategies = ['flash', 'flare', 'fedavg']

report = []

for strat in strategies:
    eval_file = os.path.join(results_dir, f'{strat}_HFL_eval_metrics.csv')
    fit_file = os.path.join(results_dir, f'{strat}_HFL_fit_metrics.csv')
    hw_file = os.path.join(results_dir, f'{strat}_server_hw.csv')
    
    if not os.path.exists(eval_file):
        continue
        
    df_eval = pd.read_csv(eval_file)
    df_fit = pd.read_csv(fit_file)
    df_hw = pd.read_csv(hw_file)
    
    rounds = df_eval['round'].max()
    final_acc = df_eval['accuracy'].iloc[-1]
    
    leaf_energy = df_fit['leaf_energy_joules'].sum()
    agg_energy = df_fit['agg_energy_joules'].sum()
    server_energy = df_hw['server_energy_joules'].sum()
    total_energy = leaf_energy + agg_energy + server_energy
    
    avg_comp = df_fit['leaf_compression_ratio_applied'].mean() if 'leaf_compression_ratio_applied' in df_fit.columns else 1.0
    
    # Check for NaNs/Infs/Zeros
    sane = True
    if df_eval['accuracy'].isna().any() or (df_eval['accuracy'] < 0).any(): sane = False
    if df_fit['leaf_energy_joules'].isna().any() or (df_fit['leaf_energy_joules'] < 0).any(): sane = False
    if server_energy == 0: sane = "Suspect (Zero Server Energy)"
    
    report.append({
        'strategy': strat,
        'rounds': rounds,
        'final_accuracy': final_acc,
        'total_energy': total_energy,
        'leaf_energy': leaf_energy,
        'agg_energy': agg_energy,
        'server_energy': server_energy,
        'avg_compression': avg_comp,
        'sane': sane
    })

for r in report:
    print(r)
