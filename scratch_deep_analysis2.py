import pandas as pd
import numpy as np

df = pd.read_csv('processed_agentic_traces.csv')

# Additional analysis for more charts

# 1. Error rate evolution across turns per model
print('=== Error Rate by Turn per Model ===')
turn_model_err = df[df['turn_number'] <= 30].groupby(['turn_number', 'model'])['has_error'].mean().reset_index()
print(turn_model_err.head(20).to_string())
print()

# 2. Cost accumulation per session (cumulative)
print('=== Cost accumulation pattern ===')
df_sorted = df.sort_values(['session_id', 'turn_number'])
df_sorted['cum_cost'] = df_sorted.groupby('session_id')['turn_cost'].cumsum()
cum_by_turn_model = df_sorted.groupby(['turn_number', 'model'])['cum_cost'].mean().reset_index()
print(cum_by_turn_model.head(20).to_string())
print()

# 3. Output length distribution by model
print('=== Output Length by Model ===')
print(df.groupby('model')['output_length'].describe())
print()

# 4. Success rate by task size per model
print('=== Success Rate by Task Size ===')
df['success'] = 1 - df['has_error']
df['task_size'] = pd.cut(df['output_length'], bins=[-1, 100, 500, np.inf], labels=['Light', 'Medium', 'Heavy'])
print(df.groupby(['model', 'task_size'], observed=False)['success'].mean().unstack())
print()

# 5. Context size vs cost
print('=== Cost by Context Size per Model ===')
df['context_size'] = pd.cut(df['input_tokens'], bins=[-1, 10000, 30000, np.inf], labels=['Low', 'Medium', 'High'])
print(df.groupby(['model', 'context_size'], observed=False)['turn_cost'].mean().unstack())
print()

# 6. Session duration analysis
session_stats = df.groupby('session_id').agg(
    model=('model', 'first'),
    total_turns=('turn_number', 'max'),
    total_cost=('turn_cost', 'sum'),
    error_rate=('has_error', 'mean'),
    avg_latency=('pre_gap', 'mean'),
    max_latency=('pre_gap', 'max'),
).reset_index()
print('=== Session Duration Distribution ===')
print(session_stats.groupby('model')['total_turns'].describe())
print()

# 7. Error cascading analysis - consecutive errors
print('=== Consecutive Error Analysis ===')
df_sorted2 = df.sort_values(['session_id', 'turn_number'])
df_sorted2['prev_error'] = df_sorted2.groupby('session_id')['has_error'].shift(1).fillna(0)
df_sorted2['error_streak'] = df_sorted2.groupby('session_id')['has_error'].transform(
    lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
) * df_sorted2['has_error']
print('Max error streak per model:')
print(df_sorted2.groupby('model')['error_streak'].max())
print()
print('Mean error streak (when erroring):')
erroring = df_sorted2[df_sorted2['has_error'] == 1]
print(erroring.groupby('model')['error_streak'].mean())
print()

# 8. Latency distribution by model (for violin plot)
print('=== Latency by Model (non-outlier range) ===')
lat_q99 = df['pre_gap'].quantile(0.99)
df_lat_clip = df[df['pre_gap'] <= lat_q99]
print(df_lat_clip.groupby('model')['pre_gap'].describe())
print()

# 9. Cost vs Error correlation by model
print('=== Cost vs Error by Turn Number ===')
turn_analysis = df.groupby(['turn_number', 'model']).agg(
    avg_cost=('turn_cost', 'mean'),
    error_rate=('has_error', 'mean'),
    count=('turn_cost', 'count')
).reset_index()
print(turn_analysis[turn_analysis['turn_number'] <= 10].to_string())
print()

# 10. Efficiency score: (1-error_rate) / cost
print('=== Efficiency Score (Success/Cost) ===')
for model in df['model'].unique():
    mdf = df[df['model'] == model]
    success_rate = 1 - mdf['has_error'].mean()
    avg_cost = mdf['turn_cost'].mean()
    if avg_cost > 0:
        efficiency = success_rate / avg_cost
        print(f"{model}: {efficiency:.2f} (success_rate={success_rate:.3f}, avg_cost=${avg_cost:.4f})")
print()

# 11. Sunk cost breakdown
print('=== Sunk Cost Analysis ===')
df['sunk_cost'] = df['has_error'] * df['turn_cost']
df['success_cost'] = (1 - df['has_error']) * df['turn_cost']
sunk_analysis = df.groupby('model').agg(
    sunk_total=('sunk_cost', 'sum'),
    useful_total=('success_cost', 'sum'),
    total_cost=('turn_cost', 'sum'),
).reset_index()
sunk_analysis['sunk_pct'] = (sunk_analysis['sunk_total'] / sunk_analysis['total_cost'] * 100).round(1)
print(sunk_analysis.to_string())
print()

# 12. Throughput analysis
print('=== Throughput (output_length / latency) ===')
df_nonzero_lat = df[df['pre_gap'] > 0]
df_nonzero_lat = df_nonzero_lat.copy()
df_nonzero_lat['throughput'] = df_nonzero_lat['output_length'] / df_nonzero_lat['pre_gap']
print(df_nonzero_lat.groupby('model')['throughput'].describe())
