import pandas as pd
import numpy as np

df = pd.read_csv('processed_agentic_traces.csv')
print(f"Turns: {len(df)}")
print(f"Sessions: {df['session_id'].nunique()}")
print(f"Total cost: ${df['turn_cost'].sum():.2f}")

# parse session_id
parts = df['session_id'].str.split('__')
df['benchmark'] = parts.str[0]
df['project'] = parts.str[1]
df['issue'] = parts.str[2]

# last turn resolved
last_turns = df.groupby('session_id').last()
last_resolved = (last_turns['has_error'] == 0).astype(int).to_dict()
df['resolved'] = df['session_id'].map(last_resolved)

print("\nModel Summary:")
for m, g in df.groupby('model'):
    n_sess = g['session_id'].nunique()
    n_turns = len(g)
    tot_cost = g['turn_cost'].sum()
    avg_cost = tot_cost / n_sess
    err_rate = g['has_error'].mean()
    print(f"  {m}: {n_sess} sess, {n_turns} turns, ${avg_cost:.2f}/sess, ${tot_cost:.2f} tot, error {err_rate*100:.1f}%")

print("\nSonnet by is_system_prompt_present:")
sonnet = df[df['model'] == 'claude-sonnet-4-6']
for sys_p, g in sonnet.groupby('is_system_prompt_present'):
    n_sess = g['session_id'].nunique()
    avg_tok = g['input_tokens'].mean()
    err_rate = g['has_error'].mean()
    turns_per_sess = len(g) / n_sess
    cost_per_sess = g['turn_cost'].sum() / n_sess
    
    # fail-100% sessions: sessions where all turns have error
    sess_err_means = g.groupby('session_id')['has_error'].mean()
    fail_100_pct = (sess_err_means == 1.0).mean()
    
    print(f"  sys_prompt={sys_p}: n_sess={n_sess}, avg_tok={avg_tok:.1f}, err_rate={err_rate*100:.1f}%, fail100={fail_100_pct*100:.1f}%, turns/sess={turns_per_sess:.1f}, cost/sess=${cost_per_sess:.2f}")
