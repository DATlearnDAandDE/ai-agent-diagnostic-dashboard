import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('processed_agentic_traces.csv')

# 1. ETL
parts = df['session_id'].str.split('__')
df['benchmark'] = parts.str[0]
df['project'] = parts.str[1]
df['issue'] = parts.str[2]

df = df.sort_values(['session_id', 'turn_number']).reset_index(drop=True)
df['cum_cost'] = df.groupby('session_id')['turn_cost'].cumsum()

# Last turn resolved
last_turns = df.groupby('session_id').last()
last_resolved_map = (last_turns['has_error'] == 0).astype(int).to_dict()
df['resolved'] = df['session_id'].map(last_resolved_map)

sess_agg = df.groupby(['session_id', 'model', 'benchmark']).agg(
    total_cost=('turn_cost', 'sum'),
    n_turns=('turn_number', 'count'),
    resolved=('resolved', 'first'),
    error_rate=('has_error', 'mean'),
    is_system_prompt=('is_system_prompt_present', 'first')
).reset_index()

print("=== 4.1 Survival Analysis (Unresolved Rate at turn t) ===")
turns_range = range(1, 40)
survival_data = []

for m in df['model'].unique():
    m_df = df[df['model'] == m]
    m_sess = sess_agg[sess_agg['model'] == m]
    for t in range(1, 40):
        active_sessions = m_df[m_df['turn_number'] == t]['session_id'].unique()
        if len(active_sessions) == 0:
            continue
        unresolved_count = m_sess[(m_sess['session_id'].isin(active_sessions)) & (m_sess['resolved'] == 0)]['session_id'].nunique()
        resolved_at_or_after_t = m_sess[(m_sess['session_id'].isin(active_sessions)) & (m_sess['resolved'] == 1)]['session_id'].nunique()
        prob_unresolved = unresolved_count / len(active_sessions)
        survival_data.append({
            'model': m, 'turn': t, 'active': len(active_sessions),
            'unresolved': unresolved_count, 'resolved_remaining': resolved_at_or_after_t,
            'prob_unresolved': prob_unresolved
        })

surv_df = pd.DataFrame(survival_data)

print("\n=== 4.2 Cost Forecasting at k=5 and k=10 ===")
def forecast_cost(k=5):
    sess_k = df[df['turn_number'] == k][['session_id', 'cum_cost']].rename(columns={'cum_cost': f'cost_at_{k}'})
    merged = pd.merge(sess_agg, sess_k, on='session_id')
    
    X = merged[[f'cost_at_{k}']].values
    y = merged['total_cost'].values
    
    model_lin = LinearRegression()
    model_lin.fit(X, y)
    preds = model_lin.predict(X)
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    
    X_log = np.log1p(X)
    y_log = np.log1p(y)
    model_log = LinearRegression()
    model_log.fit(X_log, y_log)
    preds_log = np.expm1(model_log.predict(X_log))
    r2_log = r2_score(y, preds_log)
    mae_log = mean_absolute_error(y, preds_log)
    
    print(f"k={k} (n={len(merged)} sessions):")
    print(f"  Linear model: R2={r2:.4f}, MAE=${mae:.4f}")
    print(f"  Log-transformed model: R2={r2_log:.4f}, MAE=${mae_log:.4f}")

forecast_cost(5)
forecast_cost(10)

print("\n=== 4.3 100% Fail Risk Classifier from first 5 turns ===")
first_5 = df[df['turn_number'] <= 5]
f5_feat = first_5.groupby('session_id').agg(
    error_streak=('has_error', 'sum'),
    cost_at_5=('cum_cost', 'max'),
    tokens_growth=('input_tokens', lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0),
    avg_tokens_5=('input_tokens', 'mean')
).reset_index()

f5_merged = pd.merge(sess_agg, f5_feat, on='session_id')
f5_merged['fail_100'] = (f5_merged['error_rate'] == 1.0).astype(int)

print(f"Total sessions with >=5 turns: {len(f5_merged)}")
print("Fail-100 base rates by model:")
print(f5_merged.groupby('model')['fail_100'].agg(['count', 'mean']))

# Evaluated on Sonnet subset
sonnet_f5 = f5_merged[f5_merged['model'] == 'claude-sonnet-4-6']
X_sonnet = sonnet_f5[['error_streak', 'cost_at_5', 'tokens_growth', 'avg_tokens_5']]
y_sonnet = sonnet_f5['fail_100']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_sonnet, y_sonnet)
probs = clf.predict_proba(X_sonnet)[:, 1]
auc_sonnet = roc_auc_score(y_sonnet, probs)
print(f"Sonnet subset (n={len(sonnet_f5)}) AUC: {auc_sonnet:.4f}")

print("\n=== 5.1 Circuit Breaker Simulation ===")
def simulate_circuit_breaker(cutoff_turn=20):
    total_saved = 0.0
    resolved_lost = 0
    total_resolved = sess_agg['resolved'].sum()
    
    for idx, row in sess_agg.iterrows():
        sid = row['session_id']
        s_turns = df[df['session_id'] == sid]
        n_t = row['n_turns']
        res = row['resolved']
        
        if n_t > cutoff_turn:
            cost_at_cutoff = s_turns[s_turns['turn_number'] == cutoff_turn]['cum_cost'].values[0]
            total_cost_sess = row['total_cost']
            saved = total_cost_sess - cost_at_cutoff
            total_saved += saved
            
            # Check if it was resolved AFTER cutoff_turn
            # In our contract: resolved = 1 if turn LAST turn is 0.
            # If resolution happened at turn <= cutoff_turn, resolved_lost is 0.
            # Let's find first turn where has_error == 0.
            no_err_turns = s_turns[s_turns['has_error'] == 0]['turn_number'].values
            if res == 1 and len(no_err_turns) > 0 and no_err_turns[0] > cutoff_turn:
                resolved_lost += 1
                
    print(f"Cutoff at Turn {cutoff_turn}: Saved ${total_saved:.2f}, Resolved Lost: {resolved_lost}/{total_resolved} ({(resolved_lost/total_resolved)*100:.1f}%)")

for t in [15, 20, 25]:
    simulate_circuit_breaker(t)

print("\n=== 5.2 Prompt Pruning Savings ===")
sonnet_sys0 = sess_agg[(sess_agg['model'] == 'claude-sonnet-4-6') & (sess_agg['is_system_prompt'] == 0)]
sonnet_sys1 = sess_agg[(sess_agg['model'] == 'claude-sonnet-4-6') & (sess_agg['is_system_prompt'] == 1)]
cost_sys0 = sonnet_sys0['total_cost'].mean()
cost_sys1 = sonnet_sys1['total_cost'].mean()
diff = cost_sys1 - cost_sys0
n_sys1 = len(sonnet_sys1)
prompt_savings = diff * n_sys1
print(f"Cost/sess sys=0: ${cost_sys0:.2f}, sys=1: ${cost_sys1:.2f}, Diff: ${diff:.2f}")
print(f"Estimated upper-bound savings: ${prompt_savings:.2f} across {n_sys1} sessions")

print("\n=== 5.3 Cost-Efficiency Index & Model Routing ===")
for m in sess_agg['model'].unique():
    sub = sess_agg[sess_agg['model'] == m]
    res_rate = sub['resolved'].mean()
    avg_c = sub['total_cost'].mean()
    cei = res_rate / avg_c if avg_c > 0 else 0
    print(f"  {m}: Resolved Rate = {res_rate*100:.1f}%, Avg Cost = ${avg_c:.3f}, CEI = {cei:.3f} resolved/$$")

