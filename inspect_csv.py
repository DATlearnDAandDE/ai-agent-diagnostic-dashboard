import pandas as pd
import numpy as np

df = pd.read_csv("processed_agentic_traces.csv")

# Parse session_id
parts = df['session_id'].str.split('__')
df['benchmark'] = parts.str[0]
df['project'] = parts.str[1]
df['issue'] = parts.str[2]
df['model_run'] = parts.str[3]
df['is_run2'] = df['session_id'].str.endswith('__run2') | df['session_id'].str.contains('run2')

print("PROJECTS:", df['project'].unique().tolist())
print("BENCHMARKS:", df['benchmark'].unique().tolist())
print("MODEL_RUN samples:", df['model_run'].unique()[:10].tolist())
print("IS_RUN2 samples:", df[df['is_run2']]['session_id'].unique()[:5].tolist())
print("TOTAL UNIQUE SESSIONS:", df['session_id'].nunique())

# Per-task aggregation
per_task = df.groupby('session_id').agg(
    model=('model','first'),
    project=('project','first'),
    max_turn=('turn_number','max'),
    total_cost=('turn_cost','sum'),
    max_input_tokens=('input_tokens','max'),
    total_output_length=('output_length','sum'),
    avg_pre_gap=('pre_gap','mean'),
    max_pre_gap=('pre_gap','max'),
    error_rate=('has_error','mean'),
    first_no_error_turn=('turn_number','min')
).reset_index()

print("\nPER TASK DESCRIBE:\n", per_task['total_cost'].describe())
print("HIGH COST TASKS:", per_task[per_task['total_cost'] > 1]['session_id'].tolist()[:10])
print("MAX COST:", per_task['total_cost'].max())
print("MIN COST:", per_task['total_cost'].min())
print("MEAN MAX TURN:", per_task['max_turn'].mean())
print("MAX TURN:", per_task['max_turn'].max())
print("\nPER TASK COUNT BY MODEL:\n", per_task.groupby('model').size())

# Check for resolved-like signals
print("\nHAS_ERROR distribution:\n", df['has_error'].value_counts())
print("IS_SYSTEM_PROMPT distribution:\n", df['is_system_prompt_present'].value_counts())

# Phase definition
df['phase'] = pd.cut(df['turn_number'], bins=[0,10,20,30,40,50], labels=[1,2,3,4,5])
print("\nPHASE distribution:\n", df['phase'].value_counts().sort_index())

# Spike detection
print("\nMAX TURN_COST:", df['turn_cost'].max())
df['is_spike'] = df['turn_cost'] > 0.1
print("SPIKE COUNT:", df['is_spike'].sum())
print("SPIKE SAMPLES:\n", df[df['is_spike']][['session_id','turn_number','turn_cost','input_tokens']].head(10))
