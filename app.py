import os
import re
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import linregress
from openai import OpenAI

# ========================
# CONFIG
# ========================
st.set_page_config(
    page_title="Gia sư Vật lí AI – Hỗ trợ học tập và thí nghiệm Vật lí thông minh",
    page_icon="🔬",
    layout="wide"
)

# ========================
# STYLE - DASHBOARD MODERN
# ========================
st.markdown("""
<style>
:root {
    --primary: #2563eb;
    --primary-2: #1d4ed8;
    --green: #16a34a;
    --bg-soft: #f8fbff;
    --card: rgba(255,255,255,0.88);
    --text: #0f172a;
    --muted: #64748b;
    --border: rgba(148,163,184,0.25);
    --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 30%),
        radial-gradient(circle at top right, rgba(22, 163, 74, 0.08), transparent 30%),
        linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
    color: var(--text);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.main-title-wrap {
    background: linear-gradient(135deg, rgba(37,99,235,0.95), rgba(22,163,74,0.9));
    border-radius: 24px;
    padding: 26px 30px;
    color: white;
    box-shadow: var(--shadow);
    margin-bottom: 18px;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 8px;
}

.main-subtitle {
    font-size: 16px;
    opacity: 0.95;
}

.stat-card {
    background: var(--card);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: var(--shadow);
    height: 100%;
}

.stat-label {
    font-size: 14px;
    color: var(--muted);
    margin-bottom: 8px;
}

.stat-value {
    font-size: 28px;
    font-weight: 800;
    color: var(--text);
}

.section-card {
    background: var(--card);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 20px;
    box-shadow: var(--shadow);
    margin-bottom: 14px;
}

.report-card {
    background: white;
    border: 1px solid #dbeafe;
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.08);
    margin-top: 10px;
}

.report-title {
    font-size: 24px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 6px;
}

.report-subtitle {
    color: #475569;
    margin-bottom: 18px;
}

.small-chip {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: #eff6ff;
    color: #1d4ed8;
    font-size: 13px;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
}

.metric-box {
    padding: 14px 16px;
    border-radius: 16px;
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #dbeafe;
    text-align: center;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.05);
}

.metric-label {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 22px;
    font-weight: 800;
    color: #0f172a;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #16a34a) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.62rem 1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.18) !important;
}

.stDownloadButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
}

div[data-testid="stTabs"] button {
    border-radius: 12px !important;
    padding: 10px 14px !important;
    font-weight: 700 !important;
}

div[data-testid="stExpander"] details {
    border-radius: 16px !important;
    border: 1px solid #dbeafe !important;
    background: rgba(255,255,255,0.78);
}

[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    border-radius: 12px !important;
}

hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ========================
# API KEY
# ========================
api_key = os.getenv("OPENAI_API_KEY")

client = None
if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.warning("⚠️ Chưa có API key → các tính năng AI sẽ không hoạt động.")

# ========================
# HEADER
# ========================
st.markdown("""
<div class="main-title-wrap">
    <center>
    <div class="main-title">🔬 Gia sư Vật lí AI – Hỗ trợ học tập và thí nghiệm Vật lí thông minh</div>
    <div class="main-subtitle">
        Hệ thống học tập Vật lí THPT tích hợp hỏi đáp AI, giải bài, trắc nghiệm, phòng thí nghiệm thông minh,
        mô phỏng, xử lí số liệu và viết báo cáo thí nghiệm tự động.
    </div>
    </center>
</div>
""", unsafe_allow_html=True)

# ========================
# SESSION STATE
# ========================
if "history" not in st.session_state:
    st.session_state.history = []

if "quiz_text" not in st.session_state:
    st.session_state.quiz_text = ""

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "last_lab_result" not in st.session_state:
    st.session_state.last_lab_result = None

# ========================
# AI FUNCTION
# ========================
def ask_ai(messages):
    if client is None:
        return "⚠️ Chưa cấu hình API key."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Lỗi API: {str(e)}"

# ========================
# HELPERS
# ========================
def parse_number_series(text):
    text = text.strip().replace(",", " ").replace(";", " ")
    if not text:
        return np.array([])
    return np.array([float(x) for x in text.split()], dtype=float)

def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else 0.0

def add_history(question, answer):
    st.session_state.history.append({"question": question, "answer": answer})

def render_metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

def create_basic_plot(x, y, xlabel, ylabel, title, mode="scatter", return_fig=False):
    fig, ax = plt.subplots(figsize=(7, 4))

    if mode == "line":
        ax.plot(x, y, linewidth=2.5)
    elif mode == "line_with_marker":
        ax.plot(x, y, marker="o", markersize=4, linewidth=2)
    else:
        ax.scatter(x, y, s=35)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if return_fig:
        return fig

    st.pyplot(fig)
    plt.close(fig)

def create_regression_plot(x, y, xlabel, ylabel, title, return_fig=False):
    slope, intercept, r, _, _ = linregress(x, y)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, s=35, label="Dữ liệu đo")
    ax.plot(x, slope * x + intercept, linewidth=2.5, label="Đường hồi quy")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    if return_fig:
        return slope, intercept, r, fig

    st.pyplot(fig)
    plt.close(fig)
    return slope, intercept, r

def fig_to_download_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")

def build_lab_dataframe(lab_data):
    x = lab_data.get("x_data")
    y = lab_data.get("y_data")
    x_name = lab_data.get("x_name", "X")
    y_name = lab_data.get("y_name", "Y")

    if x is None or y is None:
        return None

    return pd.DataFrame({
        x_name: x,
        y_name: y
    })

def make_report_plot_figure(lab_data):
    if not lab_data:
        return None

    x = lab_data.get("x_data")
    y = lab_data.get("y_data")
    exp_type = lab_data.get("type")
    exp_name = lab_data.get("experiment")
    x_name = lab_data.get("x_name", "X")
    y_name = lab_data.get("y_name", "Y")

    if x is None or y is None or len(x) < 2 or len(y) < 2:
        return None

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.6))

    if exp_type == "ohm":
        slope, intercept, _, _, _ = linregress(x, y)
        ax.scatter(x, y, s=35, label="Dữ liệu đo")
        ax.plot(x, slope * x + intercept, linewidth=2.5, label="Đường hồi quy")
        ax.set_title("Đồ thị báo cáo: I - U")
        ax.legend()

    elif exp_type == "freefall":
        ax.scatter(x, y, s=35, label="Dữ liệu đo")
        ax.set_title("Đồ thị báo cáo: s - t")
        ax.legend()

    elif exp_type == "pendulum":
        T2 = y ** 2
        slope, intercept, _, _, _ = linregress(x, T2)
        ax.scatter(x, T2, s=35, label="Dữ liệu đo")
        ax.plot(x, slope * x + intercept, linewidth=2.5, label="Đường hồi quy")
        ax.set_ylabel("T² (s²)")
        ax.set_title("Đồ thị báo cáo: T² - l")
        ax.legend()

    elif exp_type in ["speed", "force", "magnetic_B"]:
        slope, intercept, _, _, _ = linregress(x, y)
        ax.scatter(x, y, s=35, label="Dữ liệu đo")
        ax.plot(x, slope * x + intercept, linewidth=2.5, label="Đường hồi quy")
        ax.set_title(f"Đồ thị báo cáo: {exp_name}")
        ax.legend()

    elif exp_type in ["sound_freq", "sound_speed", "measurement_error"]:
        ax.plot(x, y, marker="o", markersize=4, linewidth=2)
        ax.set_title(f"Đồ thị báo cáo: {exp_name}")

    elif exp_type == "boyle":
        ax.scatter(x, y, s=35, label="Dữ liệu đo")
        ax.set_title("Đồ thị báo cáo: p - V")
        ax.legend()

    else:
        ax.scatter(x, y, s=35)
        ax.set_title(f"Đồ thị báo cáo: {exp_name}")

    ax.set_xlabel(x_name if exp_type != "pendulum" else "l (m)")
    if exp_type != "pendulum":
        ax.set_ylabel(y_name)
    ax.grid(alpha=0.3)
    return fig

def ai_lab_analysis(exp_name, exp_info, raw_text, extra_result=""):
    prompt = f"""
Bạn là giáo viên Vật lí THPT và trợ lí thí nghiệm AI.

Tên thí nghiệm: {exp_name}
Mục tiêu: {exp_info['goal']}
Công thức lí thuyết: {exp_info['theory_text']}
Dụng cụ gợi ý: {', '.join(exp_info['tools'])}
Điểm cần tập trung: {exp_info['ai_note']}

Dữ liệu hoặc mô tả:
{raw_text}

Kết quả tính toán:
{extra_result}

Hãy trả lời theo cấu trúc:
1. Nhận xét dữ liệu
2. So sánh với lí thuyết
3. Sai số có thể gặp
4. Cách cải thiện thí nghiệm
5. Kết luận ngắn gọn cho học sinh THPT
"""
    return ask_ai([
        {
            "role": "system",
            "content": """
Bạn là gia sư Vật lí AI.
Trình bày rõ ràng, dễ hiểu, phù hợp học sinh THPT.
Nếu có công thức, dùng $...$ hoặc $$...$$.
Không dùng \\( \\) hoặc \\[ \\].
"""
        },
        {"role": "user", "content": prompt}
    ])

def ai_write_report(lab_data):
    prompt = f"""
Hãy viết một báo cáo thí nghiệm Vật lí ngắn gọn, rõ ràng, phù hợp học sinh THPT.

Tên thí nghiệm: {lab_data['experiment']}
Dữ liệu: {lab_data['raw_text']}
Kết quả xử lí: {lab_data['extra_result']}
Nhận xét AI: {lab_data['ai_answer']}

Yêu cầu bố cục:
1. Mục tiêu
2. Dụng cụ
3. Cơ sở lí thuyết
4. Tiến hành
5. Bảng số liệu
6. Đồ thị và kết quả xử lí
7. Nhận xét
8. Kết luận
"""
    return ask_ai([
        {
            "role": "system",
            "content": """
Bạn là giáo viên Vật lí THPT.
Viết báo cáo súc tích, mạch lạc, có tính sư phạm.
Nếu có công thức, dùng $...$ hoặc $$...$$.
"""
        },
        {"role": "user", "content": prompt}
    ])

# ========================
# EXPERIMENT CONFIG
# ========================
EXPERIMENTS = {
    "Định luật Ohm": {
        "grade": "Phù hợp THPT phần điện học nền tảng",
        "type": "ohm",
        "x_name": "U (V)",
        "y_name": "I (A)",
        "x_symbol": "U",
        "y_symbol": "I",
        "theory": r"I = \frac{U}{R}",
        "theory_text": "I = U / R",
        "goal": "Khảo sát mối quan hệ giữa điện áp và cường độ dòng điện qua điện trở.",
        "tools": ["Nguồn điện", "Điện trở", "Ampe kế", "Vôn kế", "Dây nối"],
        "ai_note": "Nhận xét độ tuyến tính của đồ thị I theo U, ước lượng R và nguyên nhân sai số.",
        "sample_x": "1 2 3 4 5",
        "sample_y": "0.10 0.20 0.31 0.39 0.50"
    },
    "Rơi tự do": {
        "grade": "Lớp 10",
        "type": "freefall",
        "x_name": "t (s)",
        "y_name": "s (m)",
        "x_symbol": "t",
        "y_symbol": "s",
        "theory": r"s = \frac{1}{2}gt^2",
        "theory_text": "s = 1/2 g t^2",
        "goal": "Xác định gia tốc rơi tự do từ dữ liệu quãng đường và thời gian.",
        "tools": ["Vật rơi", "Thước đo", "Đồng hồ thời gian / cổng quang điện"],
        "ai_note": "Đánh giá mức độ phù hợp với công thức rơi tự do và giải thích chênh lệch so với 9.8 m/s².",
        "sample_x": "0.1 0.2 0.3 0.4 0.5",
        "sample_y": "0.05 0.20 0.44 0.79 1.23"
    },
    "Con lắc đơn": {
        "grade": "Lớp 10-11",
        "type": "pendulum",
        "x_name": "l (m)",
        "y_name": "T (s)",
        "x_symbol": "l",
        "y_symbol": "T",
        "theory": r"T = 2\pi\sqrt{\frac{l}{g}}",
        "theory_text": "T = 2π√(l/g)",
        "goal": "Khảo sát sự phụ thuộc của chu kỳ dao động vào chiều dài dây.",
        "tools": ["Quả nặng", "Dây treo", "Giá đỡ", "Đồng hồ bấm giây"],
        "ai_note": "Phân tích đồ thị T² theo l và ước lượng g.",
        "sample_x": "0.2 0.4 0.6 0.8 1.0",
        "sample_y": "0.90 1.27 1.55 1.79 2.00"
    },
    "Đo tốc độ của vật chuyển động": {
        "grade": "Lớp 10",
        "type": "speed",
        "x_name": "t (s)",
        "y_name": "s (m)",
        "x_symbol": "t",
        "y_symbol": "s",
        "theory": r"v = \frac{s}{t}",
        "theory_text": "v = s / t",
        "goal": "Xác định tốc độ của vật từ số liệu quãng đường và thời gian.",
        "tools": ["Xe lăn / vật chuyển động", "Thước đo", "Đồng hồ thời gian"],
        "ai_note": "Nhận xét vật chuyển động đều hay không đều, phân tích tốc độ trung bình.",
        "sample_x": "1 2 3 4 5",
        "sample_y": "0.8 1.7 2.5 3.4 4.2"
    },
    "Tổng hợp lực": {
        "grade": "Lớp 10",
        "type": "force",
        "x_name": "F lý thuyết (N)",
        "y_name": "F thực nghiệm (N)",
        "x_symbol": "F_lt",
        "y_symbol": "F_tn",
        "theory": r"F^2 = F_1^2 + F_2^2 + 2F_1F_2\cos\alpha",
        "theory_text": "F^2 = F1^2 + F2^2 + 2F1F2cos(alpha)",
        "goal": "So sánh hợp lực tính theo lí thuyết với kết quả thực nghiệm.",
        "tools": ["Bàn lực", "Lò xo", "Quả nặng", "Ròng rọc", "Dây"],
        "ai_note": "Đánh giá sai lệch giữa giá trị lí thuyết và thực nghiệm trong thí nghiệm tổng hợp lực.",
        "sample_x": "1.0 1.5 2.0 2.5 3.0",
        "sample_y": "0.95 1.45 1.92 2.42 2.88"
    },
    "Đo tần số sóng âm": {
        "grade": "Lớp 11",
        "type": "sound_freq",
        "x_name": "Lần đo",
        "y_name": "f (Hz)",
        "x_symbol": "n",
        "y_symbol": "f",
        "theory": r"f = \frac{1}{T}",
        "theory_text": "f = 1 / T",
        "goal": "Đo tần số của sóng âm từ các lần đo thực nghiệm.",
        "tools": ["Âm thoa", "Micro", "Dao động kí / phần mềm đo âm"],
        "ai_note": "Nhận xét tính ổn định của tần số đo được và sai số phép đo.",
        "sample_x": "1 2 3 4 5",
        "sample_y": "438 440 441 439 440"
    },
    "Đo tốc độ truyền âm": {
        "grade": "Lớp 11",
        "type": "sound_speed",
        "x_name": "Lần đo",
        "y_name": "v (m/s)",
        "x_symbol": "n",
        "y_symbol": "v",
        "theory": r"v = \lambda f",
        "theory_text": "v = lambda * f",
        "goal": "Xác định tốc độ truyền âm trong không khí từ nhiều lần đo.",
        "tools": ["Ống cộng hưởng", "Âm thoa / nguồn âm", "Thước đo"],
        "ai_note": "So sánh giá trị thực nghiệm với tốc độ âm chuẩn trong không khí.",
        "sample_x": "1 2 3 4 5",
        "sample_y": "338 341 343 340 342"
    },
    "Định luật Boyle": {
        "grade": "Lớp 12",
        "type": "boyle",
        "x_name": "V",
        "y_name": "p",
        "x_symbol": "V",
        "y_symbol": "p",
        "theory": r"pV = \text{hằng số}",
        "theory_text": "pV = constant",
        "goal": "Khảo sát mối quan hệ giữa áp suất và thể tích của một lượng khí xác định ở nhiệt độ không đổi.",
        "tools": ["Xi lanh khí", "Pít-tông", "Cảm biến áp suất / thước chia"],
        "ai_note": "Kiểm tra xem tích pV có gần như không đổi hay không.",
        "sample_x": "1 2 3 4 5",
        "sample_y": "10 5 3.4 2.5 2.0"
    },
    "Đo độ lớn cảm ứng từ": {
        "grade": "Lớp 12",
        "type": "magnetic_B",
        "x_name": "I (A)",
        "y_name": "F (N)",
        "x_symbol": "I",
        "y_symbol": "F",
        "theory": r"F = BIL",
        "theory_text": "F = B I L",
        "goal": "Xác định độ lớn cảm ứng từ từ lực từ tác dụng lên dây dẫn mang dòng điện.",
        "tools": ["Nam châm", "Dây dẫn", "Ampe kế", "Nguồn điện", "Lực kế"],
        "ai_note": "Nhận xét sự phụ thuộc của lực từ vào cường độ dòng điện và ước lượng B.",
        "sample_x": "1 2 3 4 5",
        "sample_y": "0.02 0.04 0.06 0.08 0.10"
    },
    "Tính sai số phép đo": {
        "grade": "Lớp 10",
        "type": "measurement_error",
        "x_name": "Lần đo",
        "y_name": "Giá trị đo",
        "x_symbol": "n",
        "y_symbol": "x",
        "theory": r"\bar{x} = \frac{x_1+x_2+\cdots+x_n}{n}",
        "theory_text": "Gia tri trung binh, sai so tuyet doi, sai so tuong doi",
        "goal": "Rèn kĩ năng xử lí dữ liệu đo và đánh giá sai số thực nghiệm.",
        "tools": ["Dụng cụ đo phù hợp với đại lượng cần đo"],
        "ai_note": "Tính giá trị trung bình, sai số tuyệt đối trung bình và nhận xét độ tin cậy phép đo.",
        "sample_x": "1 2 3 4 5",
        "sample_y": "10.1 10.0 9.9 10.2 10.0"
    }
}

# ========================
# TOP DASHBOARD STATS
# ========================
colA, colB, colC, colD = st.columns(4)
with colA:
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">Số thí nghiệm hỗ trợ</div><div class="stat-value">{len(EXPERIMENTS)}</div></div>',
        unsafe_allow_html=True
    )
with colB:
    st.markdown('<div class="stat-card"><div class="stat-label">Chế độ thí nghiệm</div><div class="stat-value">4</div></div>', unsafe_allow_html=True)
with colC:
    st.markdown(f'<div class="stat-card"><div class="stat-label">Lịch sử học tập</div><div class="stat-value">{len(st.session_state.history)}</div></div>', unsafe_allow_html=True)
with colD:
    st.markdown('<div class="stat-card"><div class="stat-label">Tính năng AI</div><div class="stat-value">Đầy đủ</div></div>', unsafe_allow_html=True)

st.write("")

# ========================
# TABS
# ========================
tabs = st.tabs([
    "🤖 Hỏi đáp",
    "🧠 Giải bài",
    "📝 Trắc nghiệm",
    "🔬 Phòng thí nghiệm AI",
    "📝 Chấm bài",
    "📚 Công thức",
    "📜 Lịch sử"
])

# ========================
# TAB 1: HỎI ĐÁP
# ========================
with tabs[0]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🤖 Hỏi đáp Vật lí")
    question = st.text_area("Nhập câu hỏi", placeholder="Ví dụ: Giải thích định luật bảo toàn cơ năng là gì?")

    if st.button("AI trả lời", key="ask_btn"):
        if question.strip():
            answer = ask_ai([
                {
                    "role": "system",
                    "content": """
Bạn là gia sư vật lí.
Nếu có công thức:
- Viết dạng $...$
- Không dùng \\( \\) hoặc \\[ \\]
- Giải thích rõ ràng, ngắn gọn, dễ hiểu
"""
                },
                {"role": "user", "content": question}
            ])

            add_history(question, answer)
            st.markdown("**AI trả lời:**")
            st.markdown(answer)
        else:
            st.warning("Vui lòng nhập câu hỏi.")
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# TAB 2: GIẢI BÀI
# ========================
with tabs[1]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🧠 Giải bài tập Vật lí")
    problem = st.text_area("Nhập bài tập", key="input_problem_tab2", placeholder="Ví dụ: Một vật rơi tự do từ độ cao 20m...")

    col1, col2, col3 = st.columns(3)
    prompt_ai = None

    if col1.button("💡 Gợi ý", key="hint_btn"):
        prompt_ai = f"Gợi ý cách làm bài vật lí sau: {problem}"
    if col2.button("🧩 Bước 1", key="step1_btn"):
        prompt_ai = f"Hãy làm bước đầu tiên để giải bài vật lí sau: {problem}"
    if col3.button("✅ Giải đầy đủ", key="full_btn"):
        prompt_ai = f"Hãy giải chi tiết bài vật lí sau, trình bày từng bước rõ ràng: {problem}"

    if prompt_ai and problem.strip():
        answer = ask_ai([
            {
                "role": "system",
                "content": """
Bạn là gia sư vật lí chuyên nghiệp.
Quy tắc trình bày công thức:
- Dùng $...$ cho công thức cùng dòng
- Dùng $$...$$ cho công thức xuống dòng
- Không dùng \\( \\) hoặc \\[ \\]
- Trình bày từng bước rõ ràng
"""
            },
            {"role": "user", "content": prompt_ai}
        ])

        clean_answer = (
            answer.replace(r"\[", "$$")
                  .replace(r"\]", "$$")
                  .replace(r"\(", "$")
                  .replace(r"\)", "$")
        )

        add_history(problem, clean_answer)
        st.markdown(clean_answer)
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# TAB 3: TRẮC NGHIỆM
# ========================
with tabs[2]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📝 Tạo câu hỏi trắc nghiệm")

    topic = st.text_input(
        "Chủ đề",
        placeholder="Ví dụ: Tụ điện, định luật Ohm, dao động điều hòa..."
    )
    number = st.slider("Số câu", 1, 10, 5)

    col1, col2 = st.columns([1, 3])
    with col1:
        create_quiz = st.button("Tạo đề", key="quiz_btn", use_container_width=True)

    if create_quiz:
        if not topic.strip():
            st.warning("Vui lòng nhập chủ đề.")
        else:
            prompt = f"""
Tạo {number} câu trắc nghiệm vật lí về chủ đề: {topic}.

Bắt buộc đúng định dạng sau:
Câu 1: Nội dung câu hỏi
A. Nội dung đáp án A
B. Nội dung đáp án B
C. Nội dung đáp án C
D. Nội dung đáp án D
Đáp án: A
Giải thích: Nội dung giải thích

Câu 2: Nội dung câu hỏi
A. Nội dung đáp án A
B. Nội dung đáp án B
C. Nội dung đáp án C
D. Nội dung đáp án D
Đáp án: B
Giải thích: Nội dung giải thích

Không viết lời mở đầu.
Không viết lời kết.
"""

            result = ask_ai([
                {
                    "role": "system",
                    "content": "Bạn là giáo viên vật lí. Hãy tạo đề trắc nghiệm đúng định dạng yêu cầu."
                },
                {"role": "user", "content": prompt}
            ])

            st.session_state.quiz_text = result
            st.session_state.quiz_submitted = False

            for k in list(st.session_state.keys()):
                if k.startswith("quiz_answer_"):
                    del st.session_state[k]

    def parse_quiz(text):
        blocks = re.split(r"(?=Câu\s*\d+\s*:)", text.strip())
        blocks = [b.strip() for b in blocks if b.strip()]
        parsed = []

        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) < 6:
                continue

            question_line = lines[0]
            question_text = re.sub(r"^Câu\s*\d+\s*:\s*", "", question_line).strip()

            options = {"A": "", "B": "", "C": "", "D": ""}
            correct = ""
            explain = ""

            for line in lines[1:]:
                if re.match(r"^A\.\s*", line):
                    options["A"] = re.sub(r"^A\.\s*", "", line).strip()
                elif re.match(r"^B\.\s*", line):
                    options["B"] = re.sub(r"^B\.\s*", "", line).strip()
                elif re.match(r"^C\.\s*", line):
                    options["C"] = re.sub(r"^C\.\s*", "", line).strip()
                elif re.match(r"^D\.\s*", line):
                    options["D"] = re.sub(r"^D\.\s*", "", line).strip()
                elif re.match(r"^Đáp án\s*:\s*", line, re.IGNORECASE):
                    correct = re.sub(r"^Đáp án\s*:\s*", "", line, flags=re.IGNORECASE).strip().upper()
                elif re.match(r"^Giải thích\s*:\s*", line, re.IGNORECASE):
                    explain = re.sub(r"^Giải thích\s*:\s*", "", line, flags=re.IGNORECASE).strip()

            if question_text and all(options.values()) and correct in ["A", "B", "C", "D"]:
                parsed.append({
                    "question": question_text,
                    "options": options,
                    "correct": correct,
                    "explain": explain
                })

        return parsed

    if st.session_state.quiz_text:
        parsed_questions = parse_quiz(st.session_state.quiz_text)

        if not parsed_questions:
            st.error("Không đọc được đề theo đúng định dạng.")
            with st.expander("Xem nội dung AI trả về"):
                st.code(st.session_state.quiz_text)
        else:
            st.success(f"Đã tạo {len(parsed_questions)} câu hỏi.")

            for i, q in enumerate(parsed_questions):
                with st.container(border=True):
                    st.markdown(f"### Câu {i+1}: {q['question']}")
                    options_display = [
                        f"A. {q['options']['A']}",
                        f"B. {q['options']['B']}",
                        f"C. {q['options']['C']}",
                        f"D. {q['options']['D']}",
                    ]
                    st.radio(
                        "Chọn đáp án:",
                        options_display,
                        index=None,
                        key=f"quiz_answer_{i}"
                    )

            if st.button("Nộp bài", key="submit_quiz", use_container_width=True):
                st.session_state.quiz_submitted = True

            if st.session_state.quiz_submitted:
                score = 0
                for i, q in enumerate(parsed_questions):
                    selected = st.session_state.get(f"quiz_answer_{i}")
                    if selected:
                        selected_letter = selected.split(".")[0].strip().upper()
                        if selected_letter == q["correct"]:
                            score += 1

                st.success(f"🎯 Điểm của bạn: {score}/{len(parsed_questions)}")
                st.markdown("### Đáp án và giải thích")

                for i, q in enumerate(parsed_questions):
                    selected = st.session_state.get(f"quiz_answer_{i}")
                    selected_letter = selected.split(".")[0].strip().upper() if selected else "Chưa chọn"

                    with st.container(border=True):
                        st.markdown(f"**Câu {i+1}: {q['question']}**")
                        st.write(f"Bạn chọn: {selected_letter}")
                        st.write(f"Đáp án đúng: {q['correct']}")
                        st.write(f"Giải thích: {q['explain']}")
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# TAB 4: PHÒNG THÍ NGHIỆM AI
# ========================
with tabs[3]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🔬 Phòng thí nghiệm Vật lí AI thông minh")
    st.write("Kết hợp thí nghiệm thực, mô phỏng ảo, phân tích dữ liệu và viết báo cáo theo mẫu đẹp.")

    mode = st.radio(
        "Chọn chế độ",
        ["Thí nghiệm thực", "Thí nghiệm ảo", "AI phân tích", "AI viết báo cáo"],
        horizontal=True
    )

    exp_name = st.selectbox("Chọn thí nghiệm", list(EXPERIMENTS.keys()))
    exp = EXPERIMENTS[exp_name]

    with st.expander("📘 Thông tin thí nghiệm", expanded=True):
        st.markdown(f'<span class="small-chip">{exp["grade"]}</span>', unsafe_allow_html=True)
        st.write(f"**Mục tiêu:** {exp['goal']}")
        st.write("**Công thức lí thuyết:**")
        st.latex(exp["theory"])
        st.write(f"**Dụng cụ gợi ý:** {', '.join(exp['tools'])}")

    if mode == "Thí nghiệm thực":
        st.markdown("### 🧪 Nhập dữ liệu đo thực tế")
        col1, col2 = st.columns(2)

        with col1:
            x_input = st.text_area(
                f"Dữ liệu {exp['x_name']}",
                value=exp["sample_x"],
                height=90
            )
        with col2:
            y_input = st.text_area(
                f"Dữ liệu {exp['y_name']}",
                value=exp["sample_y"],
                height=90
            )

        st.caption("Bạn có thể nhập dữ liệu cách nhau bằng khoảng trắng, dấu phẩy hoặc dấu chấm phẩy.")

        if st.button("Phân tích dữ liệu", key="analyze_lab_btn", use_container_width=True):
            try:
                x = parse_number_series(x_input)
                y = parse_number_series(y_input)

                if len(x) != len(y) or len(x) < 2:
                    st.error("Hai dãy dữ liệu phải cùng số phần tử và có ít nhất 2 giá trị.")
                else:
                    raw_text = f"{exp['x_symbol']}: {x.tolist()}\n{exp['y_symbol']}: {y.tolist()}"
                    extra_result = ""

                    if exp["type"] == "ohm":
                        slope, intercept, r = create_regression_plot(
                            x, y, exp["x_name"], exp["y_name"], "Đồ thị I - U"
                        )
                        R_est = 1 / slope if slope != 0 else None
                        render_metric_row([
                            ("Hệ số góc", f"{slope:.5f}"),
                            ("Hệ số chặn", f"{intercept:.5f}"),
                            ("Hệ số tương quan r", f"{r:.5f}")
                        ])
                        if R_est is not None:
                            st.write(f"**Điện trở ước lượng:** $R \\approx {R_est:.5f}\\ \\Omega$")
                            extra_result += f"Điện trở ước lượng R ≈ {R_est:.5f} Ω\n"

                    elif exp["type"] == "freefall":
                        create_basic_plot(x, y, exp["x_name"], exp["y_name"], "Đồ thị s - t", mode="scatter")
                        g_list = [2 * y[i] / (x[i] ** 2) for i in range(len(x)) if x[i] != 0]
                        g_mean = safe_mean(g_list)
                        st.write(f"**Gia tốc rơi tự do trung bình:** $g \\approx {g_mean:.5f}\\ m/s^2$")
                        extra_result += f"Gia tốc rơi tự do trung bình g ≈ {g_mean:.5f} m/s²\n"

                    elif exp["type"] == "pendulum":
                        T2 = y ** 2
                        slope, intercept, r = create_regression_plot(
                            x, T2, "l (m)", "T² (s²)", "Đồ thị T² - l"
                        )
                        g_est = (4 * np.pi ** 2) / slope if slope != 0 else None
                        render_metric_row([
                            ("Hệ số góc", f"{slope:.5f}"),
                            ("Hệ số chặn", f"{intercept:.5f}"),
                            ("Hệ số tương quan r", f"{r:.5f}")
                        ])
                        if g_est is not None:
                            st.write(f"**Gia tốc trọng trường ước lượng:** $g \\approx {g_est:.5f}\\ m/s^2$")
                            extra_result += f"Gia tốc trọng trường ước lượng g ≈ {g_est:.5f} m/s²\n"

                    elif exp["type"] == "speed":
                        create_regression_plot(x, y, exp["x_name"], exp["y_name"], "Đồ thị s - t")
                        speeds = [y[i] / x[i] for i in range(len(x)) if x[i] != 0]
                        v_avg = safe_mean(speeds)
                        st.write(f"**Tốc độ trung bình:** $v \\approx {v_avg:.5f}\\ m/s$")
                        extra_result += f"Tốc độ trung bình v ≈ {v_avg:.5f} m/s\n"

                    elif exp["type"] == "force":
                        slope, intercept, r = create_regression_plot(
                            x, y, exp["x_name"], exp["y_name"], "So sánh F lí thuyết và F thực nghiệm"
                        )
                        diff = np.abs(y - x)
                        mean_diff = safe_mean(diff)
                        render_metric_row([
                            ("Sai lệch TB", f"{mean_diff:.5f} N"),
                            ("Hệ số góc", f"{slope:.5f}"),
                            ("r", f"{r:.5f}")
                        ])
                        extra_result += f"Sai lệch trung bình giữa giá trị lí thuyết và thực nghiệm ≈ {mean_diff:.5f} N\n"

                    elif exp["type"] == "sound_freq":
                        create_basic_plot(x, y, "Lần đo", "f (Hz)", "Kết quả đo tần số sóng âm", mode="line_with_marker")
                        f_avg = safe_mean(y)
                        st.write(f"**Tần số trung bình:** $f \\approx {f_avg:.5f}\\ Hz$")
                        extra_result += f"Tần số trung bình f ≈ {f_avg:.5f} Hz\n"

                    elif exp["type"] == "sound_speed":
                        create_basic_plot(x, y, "Lần đo", "v (m/s)", "Kết quả đo tốc độ truyền âm", mode="line_with_marker")
                        v_avg = safe_mean(y)
                        st.write(f"**Tốc độ truyền âm trung bình:** $v \\approx {v_avg:.5f}\\ m/s$")
                        extra_result += f"Tốc độ truyền âm trung bình v ≈ {v_avg:.5f} m/s\n"

                    elif exp["type"] == "boyle":
                        create_basic_plot(x, y, "V", "p", "Đồ thị p - V", mode="scatter")
                        pv = x * y
                        pv_avg = safe_mean(pv)
                        st.write(f"**Giá trị trung bình của tích pV:** ${pv_avg:.5f}$")
                        extra_result += f"Tích pV trung bình ≈ {pv_avg:.5f}\n"

                    elif exp["type"] == "magnetic_B":
                        L_assumed = st.number_input(
                            "Nhập chiều dài đoạn dây trong từ trường L (m) để ước lượng B",
                            min_value=0.01,
                            max_value=10.0,
                            value=1.0,
                            step=0.01,
                            key="wire_length_input"
                        )
                        slope, intercept, r = create_regression_plot(
                            x, y, "I (A)", "F (N)", "Đồ thị F - I"
                        )
                        B_est = slope / L_assumed if L_assumed != 0 else None
                        render_metric_row([
                            ("Hệ số góc", f"{slope:.5f}"),
                            ("Hệ số chặn", f"{intercept:.5f}"),
                            ("r", f"{r:.5f}")
                        ])
                        if B_est is not None:
                            st.write(f"**Cảm ứng từ ước lượng:** $B \\approx {B_est:.5f}\\ T$")
                            extra_result += f"Cảm ứng từ ước lượng B ≈ {B_est:.5f} T\n"

                    elif exp["type"] == "measurement_error":
                        create_basic_plot(x, y, "Lần đo", "Giá trị đo", "Kết quả các lần đo", mode="line_with_marker")
                        x_avg = safe_mean(y)
                        delta_abs = safe_mean(np.abs(y - x_avg))
                        delta_rel = (delta_abs / abs(x_avg) * 100) if x_avg != 0 else 0
                        render_metric_row([
                            ("Giá trị TB", f"{x_avg:.5f}"),
                            ("Sai số tuyệt đối TB", f"{delta_abs:.5f}"),
                            ("Sai số tương đối", f"{delta_rel:.2f}%")
                        ])
                        extra_result += f"Giá trị trung bình ≈ {x_avg:.5f}\n"
                        extra_result += f"Sai số tuyệt đối trung bình ≈ {delta_abs:.5f}\n"
                        extra_result += f"Sai số tương đối ≈ {delta_rel:.2f}%\n"

                    ai_answer = ai_lab_analysis(exp_name, exp, raw_text, extra_result)

                    st.markdown("### 🤖 Nhận xét của AI")
                    st.markdown(ai_answer)

                    st.session_state.last_lab_result = {
                        "experiment": exp_name,
                        "type": exp["type"],
                        "x_name": exp["x_name"],
                        "y_name": exp["y_name"],
                        "raw_text": raw_text,
                        "extra_result": extra_result,
                        "ai_answer": ai_answer,
                        "x_data": x.tolist(),
                        "y_data": y.tolist()
                    }

            except Exception as e:
                st.error(f"Dữ liệu không hợp lệ: {e}")

    elif mode == "Thí nghiệm ảo":
        st.markdown("### 🖥️ Mô phỏng thí nghiệm")

        if exp["type"] == "ohm":
            R = st.slider("Điện trở R (Ω)", 1.0, 100.0, 10.0)
            U_max = st.slider("Điện áp lớn nhất Umax (V)", 1.0, 24.0, 12.0)
            U = np.linspace(0, U_max, 50)
            I = U / R
            create_basic_plot(U, I, "U (V)", "I (A)", "Mô phỏng định luật Ohm", mode="line")
            st.write(f"Khi $R = {R:.2f}\\ \\Omega$ thì $I = U/R$.")

        elif exp["type"] == "freefall":
            g = st.slider("Gia tốc g (m/s²)", 1.0, 20.0, 9.8)
            t_max = st.slider("Thời gian quan sát (s)", 1.0, 10.0, 5.0)
            t = np.linspace(0, t_max, 60)
            s = 0.5 * g * t**2
            create_basic_plot(t, s, "t (s)", "s (m)", "Mô phỏng rơi tự do", mode="line")

        elif exp["type"] == "pendulum":
            l = st.slider("Chiều dài dây l (m)", 0.1, 2.0, 1.0)
            g = st.slider("Gia tốc trọng trường g (m/s²)", 1.0, 20.0, 9.8)
            T = 2 * np.pi * np.sqrt(l / g)
            t = np.linspace(0, 2 * T, 120)
            x_sim = np.cos(2 * np.pi * t / T)
            create_basic_plot(t, x_sim, "t (s)", "Li độ chuẩn hóa", "Mô phỏng con lắc đơn", mode="line")
            st.write(f"Chu kỳ dao động: $T = {T:.5f}\\ s$")

        elif exp["type"] == "speed":
            v = st.slider("Tốc độ v (m/s)", 0.1, 10.0, 1.5)
            t_max = st.slider("Thời gian tối đa (s)", 1.0, 20.0, 10.0)
            t = np.linspace(0, t_max, 50)
            s = v * t
            create_basic_plot(t, s, "t (s)", "s (m)", "Mô phỏng chuyển động đều", mode="line")

        elif exp["type"] == "boyle":
            k = st.slider("Hằng số k = pV", 1.0, 30.0, 10.0)
            V = np.linspace(0.5, 10, 60)
            p = k / V
            create_basic_plot(V, p, "V", "p", "Mô phỏng định luật Boyle", mode="line")

        elif exp["type"] == "magnetic_B":
            B = st.slider("Cảm ứng từ B (T)", 0.01, 2.0, 0.2)
            L = st.slider("Chiều dài dây L (m)", 0.05, 2.0, 0.5)
            I = np.linspace(0, 10, 60)
            F = B * I * L
            create_basic_plot(I, F, "I (A)", "F (N)", "Mô phỏng lực từ F = BIL", mode="line")

        elif exp["type"] == "sound_freq":
            f0 = st.slider("Tần số chuẩn (Hz)", 100, 1000, 440)
            n = np.arange(1, 11)
            simulated = f0 + np.random.normal(0, 1.5, size=len(n))
            create_basic_plot(n, simulated, "Lần đo", "f (Hz)", "Mô phỏng đo tần số sóng âm", mode="line_with_marker")

        elif exp["type"] == "sound_speed":
            v0 = st.slider("Tốc độ âm chuẩn (m/s)", 300, 360, 340)
            n = np.arange(1, 11)
            simulated = v0 + np.random.normal(0, 2.0, size=len(n))
            create_basic_plot(n, simulated, "Lần đo", "v (m/s)", "Mô phỏng đo tốc độ truyền âm", mode="line_with_marker")

        elif exp["type"] == "force":
            F_theory = np.linspace(0.5, 5, 20)
            noise = np.random.normal(0, 0.08, size=len(F_theory))
            F_exp = F_theory + noise
            slope, intercept, r, fig = create_regression_plot(
                F_theory, F_exp, "F lí thuyết (N)", "F thực nghiệm (N)", "Mô phỏng tổng hợp lực", return_fig=True
            )
            st.pyplot(fig)
            plt.close(fig)
            st.write(f"Hệ số tương quan mô phỏng: {r:.4f}")

        elif exp["type"] == "measurement_error":
            true_value = st.slider("Giá trị chuẩn giả định", 1.0, 100.0, 10.0)
            spread = st.slider("Mức dao động dữ liệu", 0.01, 5.0, 0.2)
            n = np.arange(1, 11)
            measured = true_value + np.random.normal(0, spread, size=len(n))
            create_basic_plot(n, measured, "Lần đo", "Giá trị đo", "Mô phỏng sai số phép đo", mode="line_with_marker")

        else:
            st.info("Mô phỏng cho thí nghiệm này sẽ tiếp tục được mở rộng.")

    elif mode == "AI phân tích":
        st.markdown("### 🤖 AI phân tích thí nghiệm")
        student_text = st.text_area(
            "Nhập mô tả, dữ liệu hoặc nhận xét của bạn",
            placeholder="Ví dụ: Tôi đo được U = 1 2 3 4 5 và I = 0.1 0.2 0.29 0.41 0.5..."
        )

        if st.button("Phân tích bằng AI", key="free_analysis_btn", use_container_width=True):
            if not student_text.strip():
                st.warning("Vui lòng nhập nội dung.")
            else:
                ai_answer = ai_lab_analysis(exp_name, exp, student_text, "")
                st.markdown(ai_answer)

                st.session_state.last_lab_result = {
                    "experiment": exp_name,
                    "type": exp["type"],
                    "x_name": exp["x_name"],
                    "y_name": exp["y_name"],
                    "raw_text": student_text,
                    "extra_result": "",
                    "ai_answer": ai_answer,
                    "x_data": None,
                    "y_data": None
                }

    elif mode == "AI viết báo cáo":
        st.markdown("### 📝 AI viết báo cáo thí nghiệm")

        if not st.session_state.last_lab_result:
            st.info("Bạn hãy phân tích một thí nghiệm trước để AI có dữ liệu viết báo cáo.")
        else:
            lab = st.session_state.last_lab_result
            st.write(f"**Thí nghiệm gần nhất:** {lab['experiment']}")

            if st.button("Tạo báo cáo thí nghiệm", key="write_report_btn", use_container_width=True):
                report = ai_write_report(lab)
                df = build_lab_dataframe(lab)
                report_fig = make_report_plot_figure(lab)

                st.markdown("""
                <div class="report-card">
                    <div class="report-title">📄 Báo cáo thí nghiệm</div>
                    <div class="report-subtitle">Mẫu trình bày tổng hợp gồm dữ liệu, đồ thị và kết luận để nộp giáo viên</div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<span class="small-chip">Tên thí nghiệm: {lab["experiment"]}</span>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<span class="small-chip">Trạng thái: Đã phân tích</span>', unsafe_allow_html=True)
                with c3:
                    has_data = "Có dữ liệu số" if lab.get("x_data") is not None else "Chỉ có mô tả"
                    st.markdown(f'<span class="small-chip">{has_data}</span>', unsafe_allow_html=True)

                st.markdown("#### 1. Bảng số liệu")
                if df is not None:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Không có bảng số liệu vì báo cáo này được tạo từ mô tả văn bản.")

                st.markdown("#### 2. Đồ thị thí nghiệm")
                if report_fig is not None:
                    st.pyplot(report_fig)

                    img_bytes = fig_to_download_bytes(report_fig)
                    st.download_button(
                        "Tải đồ thị PNG",
                        data=img_bytes,
                        file_name="do_thi_thi_nghiem.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    plt.close(report_fig)
                else:
                    st.info("Không đủ dữ liệu để vẽ đồ thị.")

                st.markdown("#### 3. Kết quả xử lí")
                if lab.get("extra_result", "").strip():
                    st.code(lab["extra_result"])
                else:
                    st.info("Chưa có kết quả xử lí số liệu chi tiết.")

                st.markdown("#### 4. Nhận xét và kết luận")
                st.markdown(report)

                if df is not None:
                    st.download_button(
                        "Tải bảng số liệu CSV",
                        data=dataframe_to_csv_bytes(df),
                        file_name="bang_so_lieu_thi_nghiem.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# TAB 5: CHẤM BÀI
# ========================
with tabs[4]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📝 Chấm bài bằng AI")
    student_answer = st.text_area("Bài làm của học sinh")
    correct_answer = st.text_area("Đáp án đúng")

    if st.button("Chấm bài", key="grade_btn"):
        if student_answer.strip() and correct_answer.strip():
            answer = ask_ai([
                {
                    "role": "system",
                    "content": """
Bạn là giáo viên vật lí.
Hãy chấm bài ngắn gọn theo cấu trúc:
1. Điểm mạnh
2. Chỗ sai
3. Gợi ý sửa
4. Đánh giá tổng quát
"""
                },
                {
                    "role": "user",
                    "content": f"So sánh bài làm sau:\n{student_answer}\n\nVới đáp án đúng:\n{correct_answer}"
                }
            ])
            st.markdown(answer)
        else:
            st.warning("Vui lòng nhập cả bài làm và đáp án đúng.")
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# TAB 6: CÔNG THỨC
# ========================
with tabs[5]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📚 Một số công thức Vật lí cơ bản")
    c1, c2 = st.columns(2)
    with c1:
        st.latex("v = v_0 + at")
        st.latex("s = v_0 t + \\frac{1}{2}at^2")
        st.latex("I = \\frac{U}{R}")
        st.latex("v = \\frac{s}{t}")
    with c2:
        st.latex("T = 2\\pi\\sqrt{\\frac{l}{g}}")
        st.latex("F = BIL")
        st.latex("pV = const")
        st.latex("\\bar{x} = \\frac{x_1+x_2+\\cdots+x_n}{n}")
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# TAB 7: LỊCH SỬ
# ========================
with tabs[6]:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📜 Lịch sử học tập")

    if st.session_state.history:
        for item in reversed(st.session_state.history):
            with st.expander(item["question"], expanded=False):
                st.markdown(item["answer"])
    else:
        st.info("Chưa có lịch sử hỏi đáp nào.")
    st.markdown('</div>', unsafe_allow_html=True)
