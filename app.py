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

# API KEY (Deploy dùng biến môi trường)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
st.title("🔬 Gia sư Vật lí AI PRO")
st.write("Hệ thống học tập Vật lí thông minh dành cho học sinh THPT")

# ========================
# MEMORY (LỊCH SỬ)
# ========================
if "history" not in st.session_state:
    st.session_state.history = []

# ========================
# TABS
# ========================
tabs = st.tabs([
"🤖 Hỏi đáp",
"🧠 Giải bài",
"📝 Trắc nghiệm",
"📊 Thí nghiệm",
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
        st.session_state.history.append(question)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system","content": "Bạn là gia sư vật lí THPT, giải thích dễ hiểu, có ví dụ."},
                {"role": "user","content": question}
            ]
        )

        st.write(response.choices[0].message.content)

# ========================
# TAB 2: GIẢI BÀI (THÔNG MINH)
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
        prompt = f"Giải bài vật lí từng bước chi tiết: {problem}"

    if prompt:
        st.session_state.history.append(problem)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"Bạn là gia sư vật lí, hướng dẫn học sinh tư duy."},
                {"role":"user","content":prompt}
            ]
        )

        st.write(response.choices[0].message.content)

# ========================
# TAB 3: TRẮC NGHIỆM
# ========================
with tabs[2]:
    topic = st.text_input("Chủ đề")
    number = st.slider("Số câu",1,10,5)

    if st.button("Tạo câu hỏi"):
        prompt = f"""
        Tạo {number} câu trắc nghiệm vật lí về {topic}.
        Có 4 đáp án A B C D.
        Ghi rõ đáp án đúng.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )

        st.write(response.choices[0].message.content)

# ========================
# TAB 4: PHÂN TÍCH THÍ NGHIỆM
# ========================
with tabs[3]:
    x_input = st.text_input("Nhập X (cách nhau bởi dấu cách)")
    y_input = st.text_input("Nhập Y")

    if st.button("Phân tích"):
        try:
            x = np.array(list(map(float,x_input.split())))
            y = np.array(list(map(float,y_input.split())))

            slope, intercept, r, _, _ = linregress(x,y)

            fig, ax = plt.subplots()
            ax.scatter(x,y)
            ax.plot(x, slope*x + intercept)
            st.pyplot(fig)

            st.write("Hệ số góc:", round(slope,3))
            st.write("R:", round(r,3))
            st.latex(f"y={slope:.2f}x+{intercept:.2f}")

            # AI giải thích
            if st.button("Giải thích kết quả"):
                prompt = f"""
                Phương trình: y = {slope:.2f}x + {intercept:.2f}
                R = {r:.2f}
                Hãy giải thích ý nghĩa vật lí.
                """

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}]
                )

                st.write(response.choices[0].message.content)

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
        prompt = f"""
        So sánh bài làm và đáp án:

        Bài học sinh: {student_answer}
        Đáp án: {correct_answer}

        1. Chấm điểm /10
        2. Chỉ ra lỗi sai
        3. Gợi ý cải thiện
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )

        st.write(response.choices[0].message.content)

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
