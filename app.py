import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from openai import OpenAI
import os
import json
import re

# ========================
# API KEY
# ========================
api_key = os.getenv("OPENAI_API_KEY")

client = None
if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.warning("⚠️ Chưa có API key → AI sẽ không hoạt động")

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="Gia sư Vật lí AI PRO", layout="wide")

# ========================
# STYLE
# ========================
st.markdown("""
<style>
.stButton button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ========================
# TITLE
# ========================
st.title("🔬 Gia sư Vật lí AI – Hỗ trợ học tập và thí nghiệm Vật lí thông minh")
st.write("Hệ thống học tập + phòng thí nghiệm Vật lí AI dành cho học sinh THPT")

# ========================
# MEMORY
# ========================
if "history" not in st.session_state:
    st.session_state.history = []

# ========================
# AI FUNCTION
# ========================
def ask_ai(messages):
    if client is None:
        return "⚠️ Chưa cấu hình API key"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Lỗi API: {str(e)}"

# ========================
# LATEX RENDER FIX
# ========================
def render_latex(text):
    import re

    # xử lý \(...\)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)

    # xử lý \[...\]
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text)

    parts = re.split(r'(\${1,2}.*?\${1,2})', text)

    for part in parts:
        if part.startswith("$$") and part.endswith("$$"):
            st.latex(part.strip("$$"))
        elif part.startswith("$") and part.endswith("$"):
            st.latex(part.strip("$"))
        else:
            st.write(part)

# ========================
# TABS
# ========================
tabs = st.tabs([
    "🤖 Hỏi đáp",
    "🧠 Giải bài",
    "📝 Trắc nghiệm",
    "🔬 Phòng thí nghiệm AI",
    "🧪 Mô phỏng",
    "📝 Chấm bài",
    "📚 Công thức",
    "📜 Lịch sử"
])

# ========================
# TAB 1: HỎI ĐÁP
# ========================
with tabs[0]:
    question = st.text_area("Nhập câu hỏi")

    if st.button("AI trả lời"):
        if question:
            st.session_state.history.append(question)

            answer = ask_ai([
                {"role": "system","content": "Bạn là gia sư vật lí THPT, giải thích dễ hiểu, có ví dụ, dùng LaTeX khi cần."},
                {"role": "user","content": question}
            ])

            render_latex(answer)

# ========================
# TAB 2: GIẢI BÀI
# ========================
with tabs[1]:
    problem = st.text_area("Nhập bài tập")

    col1, col2, col3 = st.columns(3)

    prompt = None

    if col1.button("💡 Gợi ý"):
        prompt = f"Gợi ý cách làm bài vật lí (không giải chi tiết): {problem}"

    if col2.button("🧩 Bước 1"):
        prompt = f"Giải bước đầu tiên của bài vật lí, dừng sau bước 1: {problem}"

    if col3.button("✅ Giải đầy đủ"):
        prompt = f"Giải bài vật lí từng bước chi tiết, có công thức LaTeX: {problem}"

    if prompt and problem:
        st.session_state.history.append(problem)

        answer = ask_ai([
            {"role":"system","content":"Bạn là gia sư vật lí, hướng dẫn học sinh tư duy rõ ràng."},
            {"role":"user","content":prompt}
        ])

        render_latex(answer)

# ========================
# TAB 3: TRẮC NGHIỆM (FULL)
# ========================
with tabs[2]:
    topic = st.text_input("Chủ đề")
    number = st.slider("Số câu",1,10,5)

    if st.button("Tạo đề"):
        if topic:
            prompt = f"""
            Tạo {number} câu trắc nghiệm vật lí về {topic}.
            Trả về JSON:
            [
              {{
                "question": "...",
                "A": "...",
                "B": "...",
                "C": "...",
                "D": "...",
                "answer": "A",
                "explain": "..."
              }}
            ]
            """

            result = ask_ai([{"role":"user","content":prompt}])

            try:
                data = json.loads(result)
                st.session_state.quiz = data
                st.session_state.user_answers = {}
            except:
                st.error("AI trả dữ liệu lỗi → thử lại!")

    if "quiz" in st.session_state:
        quiz = st.session_state.quiz

        for i, q in enumerate(quiz):
            st.write(f"### Câu {i+1}: {q['question']}")

            choice = st.radio(
                "Chọn đáp án:",
                ["A", "B", "C", "D"],
                key=f"q{i}"
            )

            st.session_state.user_answers[i] = choice

            st.write(f"A. {q['A']}")
            st.write(f"B. {q['B']}")
            st.write(f"C. {q['C']}")
            st.write(f"D. {q['D']}")

        if st.button("Nộp bài"):
            score = 0

            for i, q in enumerate(quiz):
                if st.session_state.user_answers.get(i) == q["answer"]:
                    score += 1

            st.success(f"🎯 Điểm: {score}/{len(quiz)}")

            for i, q in enumerate(quiz):
                st.write("---")
                st.write(f"**Câu {i+1}**")
                st.write(f"Đáp án đúng: {q['answer']}")
                st.write(f"Giải thích: {q['explain']}")

# ========================
# TAB 4: PHÒNG THÍ NGHIỆM AI
# ========================
with tabs[3]:
    st.subheader("🔬 Phòng thí nghiệm Vật lí AI")

    exp_type = st.selectbox("Chọn thí nghiệm", [
        "Định luật Ohm",
        "Rơi tự do",
        "Dao động điều hòa"
    ])

    x_input = st.text_input("Dữ liệu X")
    y_input = st.text_input("Dữ liệu Y")

    if st.button("Phân tích thông minh"):
        try:
            x = np.array(list(map(float,x_input.split())))
            y = np.array(list(map(float,y_input.split())))

            slope, intercept, r, _, _ = linregress(x,y)

            fig, ax = plt.subplots()
            ax.scatter(x,y)
            ax.plot(x, slope*x + intercept)
            st.pyplot(fig)

            st.write(f"Hệ số góc: {slope:.3f}")
            st.write(f"R: {r:.3f}")

            prompt = f"""
            Thí nghiệm {exp_type}
            slope = {slope}
            R = {r}

            Hãy:
            - Nhận xét
            - Kết luận vật lí
            - Sai số
            """

            answer = ask_ai([{"role":"user","content":prompt}])

            st.write("### 🤖 Nhận xét AI")
            render_latex(answer)

        except:
            st.error("Dữ liệu không hợp lệ!")

# ========================
# TAB 5: MÔ PHỎNG
# ========================
with tabs[4]:
    st.subheader("Con lắc đơn")

    L = st.slider("Chiều dài (m)", 0.1, 2.0, 1.0)
    g = 9.8

    T = 2 * np.pi * np.sqrt(L / g)

    st.write(f"Chu kỳ T = {T:.2f} s")

# ========================
# TAB 6: CHẤM BÀI
# ========================
with tabs[5]:
    student_answer = st.text_area("Bài làm học sinh")
    correct_answer = st.text_area("Đáp án đúng")

    if st.button("Chấm bài"):
        if student_answer and correct_answer:
            prompt = f"""
            So sánh bài làm và đáp án:

            Bài học sinh: {student_answer}
            Đáp án: {correct_answer}

            1. Chấm điểm /10
            2. Lỗi sai
            3. Gợi ý cải thiện
            """

            answer = ask_ai([{"role":"user","content":prompt}])

            render_latex(answer)

# ========================
# TAB 7: CÔNG THỨC
# ========================
with tabs[6]:
    st.latex("v=v_0+at")
    st.latex("s=v_0t+\\frac{1}{2}at^2")
    st.latex("I=\\frac{U}{R}")
    st.latex("T=2\\pi\\sqrt{\\frac{l}{g}}")

# ========================
# TAB 8: LỊCH SỬ
# ========================
with tabs[7]:
    st.subheader("Lịch sử học tập")

    for item in st.session_state.history:
        st.write("-", item)
