import pandas as pd
import glob
import os

strategies = ['flash', 'fixedcompress', 'fedavg', 'adamc']
results = []

for strat in strategies:
    data = {'Strategy': strat.upper() if strat != 'adamc' else 'adaMC'}
    
    # 1. Final eval loss at round 60
    loss_file = f'fl_results_hfl/{strat}_HFL_eval_loss.csv'
    if os.path.exists(loss_file):
        df_loss = pd.read_csv(loss_file)
        if 60 in df_loss['round'].values:
            data['Final Loss (R60)'] = df_loss[df_loss['round'] == 60]['eval_loss'].values[0]
        else:
            # If round 60 is not found, take the last available round
            data['Final Loss (R60)'] = df_loss['eval_loss'].iloc[-1]
            data['Final Loss (R60)'] = f"{data['Final Loss (R60)']} (R{int(df_loss['round'].iloc[-1])})"
    else:
        data['Final Loss (R60)'] = 'N/A'

    # 2. Best eval accuracy across all rounds
    metrics_file = f'fl_results_hfl/{strat}_HFL_eval_metrics.csv'
    if os.path.exists(metrics_file):
        df_metrics = pd.read_csv(metrics_file)
        if 'accuracy' in df_metrics.columns:
            data['Best Accuracy'] = df_metrics['accuracy'].max()
        else:
            data['Best Accuracy'] = 'N/A'
    else:
        data['Best Accuracy'] = 'N/A'

    # 3. Total server energy in joules
    hw_file = f'fl_results_hfl/{strat}_server_hw.csv'
    if os.path.exists(hw_file):
        df_hw = pd.read_csv(hw_file)
        if 'server_energy_joules' in df_hw.columns:
            data['Total Server Energy (J)'] = df_hw['server_energy_joules'].sum()
        else:
            data['Total Server Energy (J)'] = 'N/A'
    else:
        data['Total Server Energy (J)'] = 'N/A'

    # 4. Mean compression ratio from fit_metrics
    fit_file = f'fl_results_hfl/{strat}_HFL_fit_metrics.csv'
    if os.path.exists(fit_file):
        df_fit = pd.read_csv(fit_file)
        col = 'leaf_compression_ratio_applied'
        if col in df_fit.columns:
            data['Mean Comp. Ratio'] = df_fit[col].mean()
        else:
            # Fallback if column name is different
            found = False
            for c in df_fit.columns:
                if 'compression_ratio_applied' in c:
                    data['Mean Comp. Ratio'] = df_fit[c].mean()
                    found = True
                    break
            if not found:
                data['Mean Comp. Ratio'] = 'N/A'
    else:
        data['Mean Comp. Ratio'] = 'N/A'

    results.append(data)

df_results = pd.DataFrame(results)
df_results['Total Server Energy (J)'] = df_results['Total Server Energy (J)'].map(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
df_results['Mean Comp. Ratio'] = df_results['Mean Comp. Ratio'].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
print(df_results.to_string(index=False))
