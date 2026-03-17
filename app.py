import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from openai import OpenAI
import os

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="Gia sư Vật lí AI PRO", layout="wide")

# ========================
# API KEY
# ========================
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key) if api_key else None
if client is None:
    st.warning("⚠️ Chưa có API key → AI sẽ không hoạt động")

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

# ========================
# SESSION STATE
# ========================
for key in ["answer", "solution", "quiz_text"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ========================
# FIX LATEX
# ========================
def fix_latex(text):
    import re
    text = text.replace("[", "$").replace("]", "$")
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    return text

# ========================
# AI FUNCTION
# ========================
def ask_ai(messages):
    if client is None:
        return "⚠️ Chưa có API key"

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"❌ Lỗi API: {str(e)}"

# ========================
# TABS
# ========================
tabs = st.tabs([
    "🤖 Hỏi đáp",
    "🧠 Giải bài",
    "📝 Trắc nghiệm",
    "🔬 Thí nghiệm",
    "🧪 Mô phỏng",
    "📝 Chấm bài",
    "📚 Công thức"
])

# ========================
# TAB 1: HỎI ĐÁP
# ========================
with tabs[0]:
    question = st.text_area("Nhập câu hỏi")

    if st.button("AI trả lời"):
        if question:
            st.session_state.answer = ask_ai([
                {"role": "system", "content": "Gia sư vật lí, dùng $...$ cho công thức"},
                {"role": "user", "content": question}
            ])

    if st.session_state.answer:
        st.markdown(fix_latex(st.session_state.answer))

# ========================
# TAB 2: GIẢI BÀI
# ========================
with tabs[1]:
    problem = st.text_area("Nhập bài tập")

    col1, col2, col3 = st.columns(3)

    if col1.button("💡 Gợi ý"):
        st.session_state.solution = ask_ai([
            {"role": "system", "content": "Gợi ý cách làm, dùng $...$"},
            {"role": "user", "content": problem}
        ])

    if col2.button("🧩 Bước 1"):
        st.session_state.solution = ask_ai([
            {"role": "system", "content": "Giải bước đầu tiên, dùng $...$"},
            {"role": "user", "content": problem}
        ])

    if col3.button("✅ Giải đầy đủ"):
        st.session_state.solution = ask_ai([
            {"role": "system", "content": "Giải chi tiết, dùng $...$"},
            {"role": "user", "content": problem}
        ])

    if st.session_state.solution:
        st.markdown(fix_latex(st.session_state.solution))

# ========================
# TAB 3: TRẮC NGHIỆM
# ========================
with tabs[2]:
    topic = st.text_input("Chủ đề")
    number = st.slider("Số câu",1,10,5)

    if st.button("Tạo đề"):
        if topic:
            st.session_state.quiz_text = ask_ai([
                {"role":"user","content":f"""
                Tạo {number} câu trắc nghiệm vật lí về {topic}

                Format:
                Câu 1: ...
                A. ...
                B. ...
                C. ...
                D. ...
                Đáp án: A
                Giải thích: ...
                """}
            ])

    if st.session_state.quiz_text:
        text = st.session_state.quiz_text
        questions = text.split("Câu ")[1:]

        user_answers = []
        correct_answers = []

        for i, q in enumerate(questions):
            lines = q.split("\n")

            if len(lines) < 7:
                continue

            question = lines[0]
            A = lines[1].replace("A. ","")
            B = lines[2].replace("B. ","")
            C = lines[3].replace("C. ","")
            D = lines[4].replace("D. ","")

            correct = lines[5].replace("Đáp án: ","").strip()
            explain = lines[6].replace("Giải thích: ","")

            st.write(f"### Câu {i+1}: {question}")

            choice = st.radio("Chọn:", ["A","B","C","D"], key=f"q{i}")

            user_answers.append(choice)
            correct_answers.append((correct, explain))

            st.write(f"A. {A}")
            st.write(f"B. {B}")
            st.write(f"C. {C}")
            st.write(f"D. {D}")

        if st.button("Nộp bài"):
            score = sum([user_answers[i]==correct_answers[i][0] for i in range(len(user_answers))])
            st.success(f"🎯 {score}/{len(user_answers)}")

            for i in range(len(user_answers)):
                st.write(f"--- Câu {i+1}")
                st.write(f"Đáp án: {correct_answers[i][0]}")
                st.write(f"Giải thích: {correct_answers[i][1]}")

# ========================
# TAB 4: THÍ NGHIỆM
# ========================
with tabs[3]:
    x_input = st.text_input("X")
    y_input = st.text_input("Y")

    if st.button("Phân tích"):
        try:
            x = np.array(list(map(float,x_input.split())))
            y = np.array(list(map(float,y_input.split())))

            slope, intercept, r, _, _ = linregress(x,y)

            fig, ax = plt.subplots()
            ax.scatter(x,y)
            ax.plot(x, slope*x + intercept)
            st.pyplot(fig)

            st.write(f"Slope: {slope:.2f}")
            st.write(f"R: {r:.2f}")

        except:
            st.error("Sai dữ liệu")

# ========================
# TAB 5: MÔ PHỎNG
# ========================
with tabs[4]:
    L = st.slider("Chiều dài", 0.1, 2.0, 1.0)
    T = 2 * np.pi * np.sqrt(L / 9.8)
    st.write(f"T = {T:.2f} s")

# ========================
# TAB 6: CHẤM BÀI
# ========================
with tabs[5]:
    a = st.text_area("Bài học sinh")
    b = st.text_area("Đáp án")

    if st.button("Chấm"):
        st.markdown(ask_ai([{"role":"user","content":f"So sánh:\n{a}\n{b}"}]))

# ========================
# TAB 7: CÔNG THỨC
# ========================
with tabs[6]:
    st.latex("v=v_0+at")
    st.latex("I=U/R")
