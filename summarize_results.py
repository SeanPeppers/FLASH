import pandas as pd
import os

strategies = ['flash', 'fixedcompress', 'fedavg', 'adamc']
results = []

for strategy in strategies:
    stats = {'Strategy': strategy.upper()}
    
    # 1. Final eval loss (last round)
    loss_file = f'fl_results_hfl/{strategy}_HFL_eval_loss.csv'
    if os.path.exists(loss_file):
        df_loss = pd.read_csv(loss_file)
        # Get the last non-NaN loss if possible, or just the last one
        last_loss = df_loss['eval_loss'].iloc[-1]
        stats['Final Eval Loss'] = last_loss
    
    # 2. Best eval accuracy
    metrics_file = f'fl_results_hfl/{strategy}_HFL_eval_metrics.csv'
    if os.path.exists(metrics_file):
        df_metrics = pd.read_csv(metrics_file)
        stats['Best Eval Accuracy'] = df_metrics['accuracy'].max()
    
    # 3. Total server-side energy
    hw_file = f'fl_results_hfl/{strategy}_server_hw.csv'
    if os.path.exists(hw_file):
        df_hw = pd.read_csv(hw_file)
        # Explicitly look for 'server_energy_joules'
        if 'server_energy_joules' in df_hw.columns:
            stats['Total Server Energy (J)'] = df_hw['server_energy_joules'].sum()
        else:
            # Fallback to any 'energy' column that has 'server' in it
            energy_cols = [col for col in df_hw.columns if 'energy' in col.lower() and 'server' in col.lower()]
            if energy_cols:
                stats['Total Server Energy (J)'] = df_hw[energy_cols[0]].sum()
            else:
                stats['Total Server Energy (J)'] = 'N/A'

    # 4. Mean compression ratio
    fit_file = f'fl_results_hfl/{strategy}_HFL_fit_metrics.csv'
    if os.path.exists(fit_file):
        df_fit = pd.read_csv(fit_file)
        comp_col = [col for col in df_fit.columns if 'compression_ratio' in col.lower()]
        if comp_col:
            stats['Mean Compression Ratio'] = df_fit[comp_col[0]].mean()
        else:
            stats['Mean Compression Ratio'] = 'N/A'
            
    results.append(stats)

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
