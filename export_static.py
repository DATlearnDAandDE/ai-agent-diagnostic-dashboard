import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

print("Đang khởi tạo trình xuất file tĩnh (Static Site Generator)...")

# 1. Tạo thư mục dist
if not os.path.exists('dist'):
    os.makedirs('dist')

# 2. Xử lý dữ liệu
print("Đang xử lý dữ liệu...")
df = pd.read_csv('processed_agentic_traces.csv')
num_cols = ['output_length', 'pre_gap', 'has_error', 'turn_cost', 'turn_number', 'input_tokens', 'is_system_prompt_present']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

df['latency'] = df['pre_gap']
df['success'] = 1 - df['has_error']
df['sunk_cost'] = df['has_error'] * df['turn_cost']
df['success_cost'] = df['success'] * df['turn_cost']
df['throughput'] = np.where(df['latency'] > 0, df['output_length'] / df['latency'], 0)

df = df.sort_values(['session_id', 'turn_number'])
df['cum_cost'] = df.groupby('session_id')['turn_cost'].cumsum()
df['error_streak'] = df.groupby('session_id')['has_error'].transform(
    lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
) * df['has_error']

df['task_size'] = pd.cut(df['output_length'], bins=[-1, 150, 600, np.inf], labels=['Nhẹ (<150)', 'Vừa (150-600)', 'Nặng (>600)'])
df['context_size'] = pd.cut(df['input_tokens'], bins=[-1, 15000, 35000, np.inf], labels=['Thấp (<15K)', 'Trung bình (15K-35K)', 'Cao (>35K)'])

COLORS = {
    'claude-opus-4-6': '#10b981',
    'claude-sonnet-4-6': '#3b82f6',
    'deepseek-v3.1': '#f59e0b',
    'minimax-m2.5': '#f43f5e'
}

layout_config = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Plus Jakarta Sans', color='#ffffff', size=14, weight='bold'),
    title=dict(font=dict(size=18, color='#ffffff', weight='bold')),
    legend=dict(bgcolor='rgba(30,41,59,0.9)', bordercolor='rgba(255,255,255,0.4)', borderwidth=1, font=dict(color='#ffffff', size=13)),
    margin=dict(t=60, l=50, r=30, b=50),
    hovermode='x unified'
)

def style_fig(fig):
    fig.update_layout(**layout_config)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.15)', zeroline=False, title_font=dict(color='#bae6fd', size=14, weight='bold'), tickfont=dict(color='#f8fafc', size=12))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.15)', zeroline=False, title_font=dict(color='#bae6fd', size=14, weight='bold'), tickfont=dict(color='#f8fafc', size=12))
    return fig

html_figs = {}

def get_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)

print("Đang sinh các biểu đồ Plotly...")

# Q1
agg_q1 = df.groupby('model').agg(turns=('turn_cost','count'), cost=('turn_cost','sum')).reset_index()
fig1 = px.pie(agg_q1, values='turns', names='model', hole=0.5, color='model', color_discrete_map=COLORS, title="[Q1] Phân bổ Khối lượng (Total Turns)")
fig1.update_traces(textinfo='percent+label', textfont_size=14)
html_figs['fig1'] = get_html(style_fig(fig1))

fig2 = px.pie(agg_q1, values='cost', names='model', hole=0.5, color='model', color_discrete_map=COLORS, title="[Q1] Phân bổ Tiêu hao Ngân sách (Total Cost)")
fig2.update_traces(textinfo='percent+label', textfont_size=14)
html_figs['fig2'] = get_html(style_fig(fig2))

# Q2
sess_q2 = df.groupby(['session_id', 'model'])['turn_number'].max().reset_index()
fig3 = px.box(sess_q2, x='model', y='turn_number', color='model', color_discrete_map=COLORS, title="[Q2] Hình thái Độ dài Phiên (Max Turns/Session)")
fig3.update_layout(yaxis_title="Số Lượt (Turns)")
html_figs['fig3'] = get_html(style_fig(fig3))

fig4 = px.violin(df, x='model', y='input_tokens', color='model', color_discrete_map=COLORS, box=True, title="[Q2] Hình thái Khối lượng Ngữ cảnh (Input Tokens)")
fig4.update_layout(yaxis_title="Input Tokens")
html_figs['fig4'] = get_html(style_fig(fig4))

# Q3
df_sonnet = df[df['model'] == 'claude-sonnet-4-6'].groupby('task_size', observed=False).agg(sunk=('sunk_cost','sum'), roi=('success_cost','sum')).reset_index()
fig5 = go.Figure()
fig5.add_trace(go.Bar(y=df_sonnet['task_size'], x=-df_sonnet['sunk'], name='Lỗ (Sunk Cost)', orientation='h', marker_color='#f43f5e', text=df_sonnet['sunk'].apply(lambda x: f"${x:,.0f}")))
fig5.add_trace(go.Bar(y=df_sonnet['task_size'], x=df_sonnet['roi'], name='Lãi (ROI Cost)', orientation='h', marker_color='#10b981', text=df_sonnet['roi'].apply(lambda x: f"${x:,.0f}")))
fig5.update_layout(barmode='relative', title="[Q3] Sunk Cost vs ROI của Sonnet phân bổ theo Kích thước Tác vụ", xaxis_title="Ngân sách ($)")
html_figs['fig5'] = get_html(style_fig(fig5))

# Q4
err_trend = df[df['turn_number']<=35].groupby('turn_number')['has_error'].mean().reset_index()
fig6 = px.line(err_trend, x='turn_number', y='has_error', markers=True, line_shape="spline", title="[Q4] Hội chứng Suy giảm Ngữ cảnh (Error vs Turn)")
fig6.update_traces(line_color="#f59e0b", line_width=4, marker=dict(size=8))
fig6.update_layout(yaxis_title="Tỷ lệ Lỗi (Error Rate)", xaxis_title="Lượt chat (Turn Number)", yaxis_tickformat='.0%')
fig6.add_vline(x=15, line_dash="dash", line_color="#f43f5e", annotation_text="Điểm gãy ngữ cảnh")
html_figs['fig6'] = get_html(style_fig(fig6))

# Q5
pivot_sys = df.groupby(['model', 'is_system_prompt_present'])['has_error'].mean().unstack().fillna(0) * 100
pivot_sys.columns = ['Không Có System Prompt', 'Có System Prompt (Ràng buộc)']
fig7 = go.Figure(data=go.Heatmap(z=pivot_sys.values, x=pivot_sys.columns, y=pivot_sys.index, colorscale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#f43f5e']], text=np.round(pivot_sys.values, 1), texttemplate="<b>%{text}%</b>", textfont=dict(size=15, color="white")))
fig7.update_layout(title="[Q5] Mức độ Ngộ độc System Prompt lên Tỷ lệ Lỗi", yaxis_title="")
html_figs['fig7'] = get_html(style_fig(fig7))

# Q6
df_pred = df.copy()
df_pred['Turn_Bins'] = pd.cut(df_pred['turn_number'], bins=[0, 5, 15, 30, 999], labels=['1-5 Lượt', '6-15 Lượt', '16-30 Lượt', '>30 Lượt'])
df_pred['Token_Bins'] = pd.cut(df_pred['input_tokens'], bins=[0, 10000, 20000, 30000, 999999], labels=['<10k Tokens', '10k-20k Tokens', '20k-30k Tokens', '>30k Tokens'])
risk_matrix = df_pred.groupby(['Token_Bins', 'Turn_Bins'], observed=False)['has_error'].mean().unstack() * 100
fig8 = go.Figure(data=go.Heatmap(z=risk_matrix.values, x=risk_matrix.columns, y=risk_matrix.index, colorscale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#f43f5e']], text=np.round(risk_matrix.values, 1), texttemplate="<b>%{text}%</b>", textfont=dict(size=14, color="white")))
fig8.update_layout(title="[Q6] Ma trận Tiên lượng Xác suất Thất bại (%)", xaxis_title="Thời lượng Phiên (Turns)", yaxis_title="Kích thước Đầu vào (Tokens)")
html_figs['fig8'] = get_html(style_fig(fig8))

# Q7
heavy_tasks = df[(df['output_length'] > 500) & (df['model'].isin(['claude-opus-4-6', 'claude-sonnet-4-6']))]
heavy_tasks['Trạng thái'] = heavy_tasks['has_error'].map({1: 'Lỗi', 0: 'Thành công'})
fig9 = px.scatter(heavy_tasks, x='latency', y='turn_cost', color='model', symbol='Trạng thái', color_discrete_map=COLORS, size='output_length', title="[Q7] Đánh đổi tại Tác vụ Nặng (>500 out-tokens): Opus vs Sonnet", size_max=20)
fig9.update_layout(xaxis_title="Độ trễ xử lý (Latency - giây)", yaxis_title="Chi phí (USD)")
html_figs['fig9'] = get_html(style_fig(fig9))

# Q8
route_mat = df.groupby(['context_size', 'task_size'], observed=False)['success'].mean().unstack() * 100
fig10 = go.Figure(data=go.Heatmap(z=route_mat.values, x=route_mat.columns, y=route_mat.index, colorscale=[[0, '#f43f5e'], [0.5, '#f59e0b'], [1, '#10b981']], text=np.round(route_mat.values, 1), texttemplate="<b>%{text}%</b>", textfont=dict(color='white', size=15)))
fig10.update_layout(title="[Q8] Tiêu chí Định tuyến: Ma trận Tỷ lệ Thành công (%)", xaxis_title="Khối lượng Đầu ra (Task Size)", yaxis_title="Kích thước Đầu vào (Context Size)")
html_figs['fig10'] = get_html(style_fig(fig10))

# Q9
cutoffs = list(range(1, 21))
savings = []
tot_sunk = df['sunk_cost'].sum()
for c in cutoffs:
    saved = df[df['turn_number'] > c]['sunk_cost'].sum()
    savings.append((saved / tot_sunk) * 100 if tot_sunk else 0)
fig11 = px.line(x=cutoffs, y=savings, markers=True, title="[Q9] Ngưỡng Ngắt mạch (Circuit Breaker): Đường cong Cứu vãn Ngân sách")
fig11.update_traces(line_color="#10b981", line_width=4, marker=dict(size=10))
fig11.update_layout(xaxis_title="Cắt đứt chuỗi tại Lượt chat số (Turn Cutoff)", yaxis_title="Ngân sách Chìm Được Bảo toàn (%)", yaxis_tickformat='.0f')
html_figs['fig11'] = get_html(style_fig(fig11))

# Q10
cheap = df[(df['model'].isin(['minimax-m2.5', 'deepseek-v3.1'])) & (df['throughput']>0)]
cheap_tp = cheap.groupby(['model', 'context_size'], observed=False)['throughput'].mean().reset_index()
fig12 = px.bar(cheap_tp, x='model', y='throughput', color='context_size', barmode='group', title="[Q10] Thông lượng lao dốc khi Context to")
fig12.update_layout(yaxis_title="Throughput (Tokens/s)")
html_figs['fig12'] = get_html(style_fig(fig12))

# 3. Lắp ráp HTML Template
print("Đang lắp ráp HTML Template...")
html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Agent Diagnostic Intelligence</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #0f172a;
            color: #ffffff;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .gradient-text {{
            background: linear-gradient(135deg, #7dd3fc, #818cf8, #d8b4fe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        .metrics-container {{
            display: flex;
            justify-content: space-between;
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            flex: 1;
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.4);
            text-align: center;
        }}
        .metric-label {{ color: #ffffff; font-weight: 600; font-size: 1.1rem; margin-bottom: 10px; }}
        .metric-value {{ font-weight: 800; font-size: 1.8rem; color: #ffffff; }}
        .metric-delta {{ color: #f43f5e; font-size: 1rem; }}
        
        .tabs {{
            display: flex;
            gap: 15px;
            border-bottom: 2px solid rgba(56, 189, 248, 0.2);
            margin-bottom: 20px;
        }}
        .tab-btn {{
            background-color: #1e293b;
            border: 2px solid rgba(148, 163, 184, 0.2);
            border-bottom: none;
            color: #94a3b8;
            padding: 14px 28px;
            font-size: 1.1rem;
            font-weight: 700;
            border-radius: 10px 10px 0 0;
            cursor: pointer;
            font-family: inherit;
        }}
        .tab-btn.active {{
            background: linear-gradient(180deg, rgba(56, 189, 248, 0.25) 0%, rgba(14, 165, 233, 0.05) 100%);
            color: #38bdf8;
            border-color: #38bdf8;
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        
        .insight-box {{
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(56, 189, 248, 0.4);
            border-left: 5px solid #38bdf8;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 24px;
            line-height: 1.7;
            color: #f1f5f9;
        }}
        .insight-box h4 {{ color: #ffffff; margin-top: 0; text-transform: uppercase; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
        .chart-box {{ width: 100%; overflow: hidden; }}
        pre {{ background: #1e293b; padding: 15px; border-radius: 8px; overflow-x: auto; }}
        code {{ color: #7dd3fc; }}
    </style>
</head>
<body>

    <div class="header">
        <div class="gradient-text">🧠 AI Agent Research Analytics (10 Questions Edition)</div>
        <p>*Báo cáo phân tích chuyên sâu nhằm trực tiếp giải quyết 10 câu hỏi nghiên cứu lõi về Tối ưu hoá LLM Agents.*</p>
    </div>

    <div class="metrics-container">
        <div class="metric-card"><div class="metric-label">Tổng Lượt (Turns)</div><div class="metric-value">{len(df):,}</div></div>
        <div class="metric-card"><div class="metric-label">Số Phiên (Sessions)</div><div class="metric-value">{df['session_id'].nunique():,}</div></div>
        <div class="metric-card"><div class="metric-label">Tỷ lệ Lỗi Tổng</div><div class="metric-value">{df['has_error'].mean()*100:.1f}%</div></div>
        <div class="metric-card"><div class="metric-label">Tổng Ngân sách</div><div class="metric-value">${df['turn_cost'].sum():,.2f}</div></div>
        <div class="metric-card"><div class="metric-label">Chi phí Chìm (Sunk)</div><div class="metric-value">${df['sunk_cost'].sum():,.2f}</div><div class="metric-delta">{(df['sunk_cost'].sum()/df['turn_cost'].sum())*100:.1f}% lãng phí</div></div>
    </div>

    <div class="tabs">
        <button class="tab-btn active" onclick="openTab(event, 'tab1')">📊 1. Descriptive (Q1-Q2)</button>
        <button class="tab-btn" onclick="openTab(event, 'tab2')">🔍 2. Diagnostic (Q3-Q5)</button>
        <button class="tab-btn" onclick="openTab(event, 'tab3')">🔮 3. Predictive (Q6-Q7)</button>
        <button class="tab-btn" onclick="openTab(event, 'tab4')">💡 4. Prescriptive (Q8-Q10)</button>
    </div>

    <!-- TAB 1 -->
    <div id="tab1" class="tab-content active">
        <div class="insight-box">
            <h4>📊 Cấp 1: Phân tích Mô tả (Descriptive)</h4>
            <b>Q1: Phân bổ Khối lượng công việc (Turns) và Ngân sách (Cost) đang diễn ra như thế nào?</b><br/>
            => <i>Nghịch lý:</i> Sonnet chiếm phần lớn tổng ngân sách mặc dù số lượt gọi không phải là tuyệt đối. Minimax và Deepseek gánh khối lượng (Turns) rất cao nhưng chi phí không đáng kể.<br/><br/>
            <b>Q2: Hình thái của các phiên làm việc (Độ dài Session, Lượng Token trung bình) có đặc điểm gì?</b><br/>
            => <i>Đặc điểm:</i> Các phiên sử dụng Minimax và Deepseek thường bị kéo dài bất thường (vượt mốc 20-30 lượt) chứng tỏ tình trạng mắc kẹt. Trong khi đó Opus giải quyết vấn đề cực kỳ gãy gọn (hầu hết dưới 5 lượt) dù Context nạp vào rất lớn.
        </div>
        <div class="grid-2">
            <div class="chart-box">{html_figs['fig1']}</div>
            <div class="chart-box">{html_figs['fig2']}</div>
        </div>
        <div class="grid-2">
            <div class="chart-box">{html_figs['fig3']}</div>
            <div class="chart-box">{html_figs['fig4']}</div>
        </div>
    </div>

    <!-- TAB 2 -->
    <div id="tab2" class="tab-content">
        <div class="insight-box" style="border-left-color: #f43f5e;">
            <h4>🔍 Cấp 2: Phân tích Chẩn đoán (Diagnostic)</h4>
            <b>Q3: Tại sao Sonnet chiếm ngân sách khổng lồ nhưng tỷ lệ lãng phí lại quá cao?</b><br/>
            => Sonnet liên tục tạo ra "Sunk Cost" khổng lồ ở các Tác vụ Nặng (>600 tokens). Việc cố đấm ăn xôi ép Sonnet làm task quá sức chính là nguyên nhân đốt tiền.<br/><br/>
            <b>Q4: Nguyên nhân dẫn đến "Hội chứng Suy giảm Ngữ cảnh" (Vượt mốc 15-20 turns)?</b><br/>
            => Khi Turn > 15, Context bị phình to đột biến khiến tỷ lệ lỗi (Error Rate) của các mô hình vọt lên thẳng đứng. AI gần như mất hoàn toàn trí nhớ ngắn hạn.<br/><br/>
            <b>Q5: System Prompt đang đóng vai trò hỗ trợ hay làm hại?</b><br/>
            => Nó đang làm hại nghiêm trọng. Khi ép thêm System Prompt (ràng buộc khắt khe), tỷ lệ lỗi của các model bị kích nổ rất mạnh.
        </div>
        <div class="chart-box" style="margin-bottom: 20px;">{html_figs['fig5']}</div>
        <div class="grid-2">
            <div class="chart-box">{html_figs['fig6']}</div>
            <div class="chart-box">{html_figs['fig7']}</div>
        </div>
    </div>

    <!-- TAB 3 -->
    <div id="tab3" class="tab-content">
        <div class="insight-box" style="border-left-color: #f59e0b;">
            <h4>🔮 Cấp 3: Phân tích Dự đoán (Predictive)</h4>
            <b>Q6: Xác suất thất bại nếu Phiên > 15 lượt và > 30.000 tokens?</b><br/>
            => Ma trận Rủi ro (3.1) kết luận đanh thép: Khi chạm vào vùng cấm địa (Turn > 15 & Token > 30k), xác suất thất bại chạm ngưỡng cực độ. Ở điểm này, model bị ảo giác hoàn toàn.<br/><br/>
            <b>Q7: Task > 500 tokens: Nên chọn Opus hay Sonnet?</b><br/>
            => Giao cho Opus sẽ mất phí cao hơn, nhưng bù lại Độ trễ (Latency) ổn định và "Một phát ăn ngay". Sonnet ở mức Task này thường xuyên chập chờn và sinh ra lỗi.
        </div>
        <div class="grid-2">
            <div class="chart-box">{html_figs['fig8']}</div>
            <div class="chart-box">{html_figs['fig9']}</div>
        </div>
    </div>

    <!-- TAB 4 -->
    <div id="tab4" class="tab-content">
        <div class="insight-box" style="border-left-color: #10b981;">
            <h4>💡 Cấp 4: Phân tích Đề xuất (Prescriptive)</h4>
            <b>Q8: Tiêu chí Định tuyến (Smart Routing) tối ưu nhất?</b><br/>
            => Context < 15k & Task Nhẹ -> Dùng Sonnet. Context > 35k hoặc Task Cực Nặng -> Bắt buộc dùng Opus (Sonnet tịt ngòi 100% ở vùng này).<br/><br/>
            <b>Q9: Ngưỡng ngắt mạch (Circuit Breaker) cứu vãn ngân sách?</b><br/>
            => Nếu Cắt tự động ở Turn số 5, ta sẽ bảo toàn được đại đa số ngân sách chìm. Nếu chờ tới Turn 10 mới ngắt, rất nhiều tiền đã bốc hơi vô nghĩa.<br/><br/>
            <b>Q10: Tái cấu trúc Luồng kiểm thử (Testing Pipeline)?</b><br/>
            => Không giao task "Viết lại cả file code" cho Deepseek/Minimax. Phải tạo Micro-Tasking (Chia nhỏ hàm, context < 2k tokens) để vắt kiệt chi phí siêu rẻ của chúng mà không làm sập trí nhớ ngắn hạn.
        </div>
        <div class="grid-2">
            <div class="chart-box">{html_figs['fig10']}</div>
            <div class="chart-box">{html_figs['fig11']}</div>
        </div>
        <div class="grid-2">
            <div>
<pre><code># [TÁI CẤU TRÚC PIPELINE CHỐNG TRÀN NGỮ CẢNH]
# Giải quyết điểm yếu của Minimax và Deepseek
def micro_task_pipeline(file_content, target_bug):
    # 1. Trích xuất cục bộ (Extract AST)
    local_context = extract_function(file_content, target_bug) # < 2k tokens
    
    # 2. Xóa bỏ System Prompt không cần thiết 
    prompt = build_lightweight_prompt(local_context)
    
    # 3. Giao task cho AI giá rẻ
    cheap_model = route_to_cheap_model(prompt)
    patch = cheap_model.generate(prompt)
    
    # 4. Xác thực bằng Opus nếu cần
    return apply_patch(file_content, patch)</code></pre>
            </div>
            <div class="chart-box">{html_figs['fig12']}</div>
        </div>
    </div>

    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].className = tabcontent[i].className.replace(" active", "");
            }}
            tablinks = document.getElementsByClassName("tab-btn");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).className += " active";
            evt.currentTarget.className += " active";
            
            // Xử lý resize Plotly khi chuyển tab
            window.dispatchEvent(new Event('resize'));
        }}
    </script>
</body>
</html>
"""

# 4. Ghi file HTML
print("Đang ghi file index.html...")
with open('dist/index.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

print("✅ THÀNH CÔNG! Thư mục 'dist/' đã sẵn sàng để kéo thả (deploy) lên Cloudflare Pages.")
