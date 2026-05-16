import pandas as pd
import os

results_dir = r'C:\Users\gagep\code\FLASH\fl_results_hfl'
strategies = ['flash', 'fixedcompress', 'fedavg', 'adamc']

data = []

for strategy in strategies:
    # 1. Final eval loss at round 60 and round 10
    eval_loss_file = os.path.join(results_dir, f'{strategy}_HFL_eval_loss.csv')
    df_eval_loss = pd.read_csv(eval_loss_file)
    loss_60 = df_eval_loss[df_eval_loss['round'] == 60]['eval_loss'].values[0] if 60 in df_eval_loss['round'].values else df_eval_loss.iloc[-1]['eval_loss']
    loss_10 = df_eval_loss[df_eval_loss['round'] == 10]['eval_loss'].values[0] if 10 in df_eval_loss['round'].values else None
    
    # 2. Best eval accuracy and round
    eval_metrics_file = os.path.join(results_dir, f'{strategy}_HFL_eval_metrics.csv')
    df_eval_metrics = pd.read_csv(eval_metrics_file)
    best_acc = df_eval_metrics['accuracy'].max()
    best_acc_round = df_eval_metrics[df_eval_metrics['accuracy'] == best_acc]['round'].values[0]
    
    # 3. Total server energy
    server_hw_file = os.path.join(results_dir, f'{strategy}_server_hw.csv')
    df_server_hw = pd.read_csv(server_hw_file)
    total_server_energy = df_server_hw['server_energy_joules'].sum()
    
    # 4. Mean compression ratio
    fit_metrics_file = os.path.join(results_dir, f'{strategy}_HFL_fit_metrics.csv')
    df_fit_metrics = pd.read_csv(fit_metrics_file)
    # The column name might be 'compression_ratio_applied' or 'leaf_compression_ratio_applied'
    col_name = 'compression_ratio_applied' if 'compression_ratio_applied' in df_fit_metrics.columns else 'leaf_compression_ratio_applied'
    mean_comp_ratio = df_fit_metrics[col_name].mean()
    
    data.append({
        'Strategy': strategy.upper(),
        'Loss (R10)': loss_10,
        'Loss (R60)': loss_60,
        'Best Accuracy': best_acc,
        'Best Acc Round': best_acc_round,
        'Total Server Energy (J)': total_server_energy,
        'Mean Comp Ratio': mean_comp_ratio
    })

df_results = pd.DataFrame(data)
for index, row in df_results.iterrows():
    print(f"{row['Strategy']}|{row['Loss (R10)']:.4f}|{row['Loss (R60)']:.4f}|{row['Best Accuracy']:.4f}|{int(row['Best Acc Round'])}|{row['Total Server Energy (J)']:.2f}|{row['Mean Comp Ratio']:.4f}")
