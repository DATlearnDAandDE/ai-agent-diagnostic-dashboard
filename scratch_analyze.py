import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('processed_agentic_traces.csv')
for col in ['output_length', 'pre_gap', 'has_error', 'turn_cost', 'turn_number', 'input_tokens']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print("=== TỔNG QUAN ===")
print("Tổng số dòng:", len(df))
print("Các models:", df['model'].unique())

print("\n=== 1. DESCRIPTIVE (MÔ TẢ - TÌM OUTLIERS) ===")
print("\n--- OUTLIER CHI PHÍ (Cost) ---")
Q1_cost = df['turn_cost'].quantile(0.25)
Q3_cost = df['turn_cost'].quantile(0.75)
IQR_cost = Q3_cost - Q1_cost
outliers_cost = df[df['turn_cost'] > (Q3_cost + 1.5 * IQR_cost)]
print(f"Ngưỡng outlier cost: > {Q3_cost + 1.5 * IQR_cost:.4f} USD")
print(f"Số lượng outlier: {len(outliers_cost)} ({len(outliers_cost)/len(df)*100:.2f}%)")
print("Phân bố outlier chi phí theo model:")
if len(outliers_cost) > 0:
    print(outliers_cost.groupby('model')['turn_cost'].agg(['count', 'mean', 'max']))

print("\n--- OUTLIER ĐỘ TRỄ (Latency) ---")
Q1_lat = df['pre_gap'].quantile(0.25)
Q3_lat = df['pre_gap'].quantile(0.75)
IQR_lat = Q3_lat - Q1_lat
outliers_lat = df[df['pre_gap'] > (Q3_lat + 1.5 * IQR_lat)]
print(f"Ngưỡng outlier latency: > {Q3_lat + 1.5 * IQR_lat:.2f} s")
print(f"Số lượng outlier: {len(outliers_lat)} ({len(outliers_lat)/len(df)*100:.2f}%)")
print("Phân bố outlier độ trễ theo model:")
if len(outliers_lat) > 0:
    print(outliers_lat.groupby('model')['pre_gap'].agg(['count', 'mean', 'max']))

print("\n--- OUTLIER HỘI THOẠI DÀI (Turn Number) ---")
Q1_turn = df['turn_number'].quantile(0.25)
Q3_turn = df['turn_number'].quantile(0.75)
IQR_turn = Q3_turn - Q1_turn
outliers_turn = df[df['turn_number'] > (Q3_turn + 1.5 * IQR_turn)]
print(f"Ngưỡng outlier turn: > {Q3_turn + 1.5 * IQR_turn:.0f} turns")
print(f"Tỷ lệ lỗi ở nhóm hội thoại siêu dài này: {outliers_turn['has_error'].mean()*100:.2f}% (so với {df['has_error'].mean()*100:.2f}% trung bình)")

print("\n=== 2. DIAGNOSTIC (CHẨN ĐOÁN) ===")
print("\n--- Ảnh hưởng của System Prompt ---")
print(df.groupby(['model', 'is_system_prompt_present'])['has_error'].mean())

print("\n--- Sunk Cost (Rò rỉ ngân sách do lỗi) ---")
df['sunk_cost'] = df['has_error'] * df['turn_cost']
print("Chi phí vô ích (Sunk Cost) theo model:")
sunk = df.groupby('model')['sunk_cost'].sum()
print(sunk)
print(f"TỔNG CHI PHÍ: {df['turn_cost'].sum():.2f} USD")
print(f"TỔNG CHI PHÍ VÔ ÍCH: {sunk.sum():.2f} USD ({sunk.sum()/df['turn_cost'].sum()*100:.2f}%)")
