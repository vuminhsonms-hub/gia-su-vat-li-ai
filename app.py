import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from openai import OpenAI
import os

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
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Lỗi API: {str(e)}"

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

    if st.button("AI trả lời", key="ask_btn"):
        if question.strip():  # kiểm tra câu hỏi không rỗng
            answer = ask_ai([
                {"role": "system",
                 "content": """
                 Bạn là gia sư vật lí.
                 Nếu có công thức:
                 - Viết dạng $...$
                 - Không dùng \( \) hoặc [ ]
                 """},
                {"role": "user", "content": question}
            ])

            # Khởi tạo history nếu chưa có
            if "history" not in st.session_state:
                st.session_state.history = []

            # Lưu câu hỏi + đáp án
            st.session_state.history.append({"question": question, "answer": answer})

            st.markdown("**AI trả lời:**")
            st.markdown(answer)



# ========================
# TAB 2: GIẢI BÀI
# ========================
with tabs[1]:
    problem = st.text_area("Nhập bài tập")

    col1, col2, col3 = st.columns(3)

    prompt = None

    if col1.button("💡 Gợi ý", key="hint"):
        prompt = f"Gợi ý cách làm: {problem}"

    if col2.button("🧩 Bước 1", key="step1"):
        prompt = f"Giải bước đầu tiên: {problem}"

    if col3.button("✅ Giải đầy đủ", key="full"):
        prompt = f"Giải chi tiết có công thức: {problem}"

    if prompt and problem:
        st.session_state.history.append(problem)

        answer = ask_ai([
            {"role":"system","content":"Gia sư vật lí, giải thích dễ hiểu, dùng $...$ cho công thức"},
            {"role":"user","content":prompt}
        ])

        st.markdown(answer)

# ========================
# TAB 3: TRẮC NGHIỆM (ỔN ĐỊNH)
# ========================
with tabs[2]:
    topic = st.text_input("Chủ đề")
    number = st.slider("Số câu",1,10,5)

    if st.button("Tạo đề", key="quiz_btn"):
        if topic:
            prompt = f"""
            Tạo {number} câu trắc nghiệm vật lí về {topic}

            Format:
            Câu 1: ...
            A. ...
            B. ...
            C. ...
            D. ...
            Đáp án: A
            Giải thích: ...

            (lặp lại)
            """

            result = ask_ai([{"role":"user","content":prompt}])
            st.session_state.quiz_text = result

    if "quiz_text" in st.session_state:
        text = st.session_state.quiz_text

        questions = text.split("Câu ")[1:]

        user_answers = []
        correct_answers = []

        for i, q in enumerate(questions):
            lines = q.split("\n")

            question = lines[0]
            A = lines[1].replace("A. ","")
            B = lines[2].replace("B. ","")
            C = lines[3].replace("C. ","")
            D = lines[4].replace("D. ","")

            correct = lines[5].replace("Đáp án: ","").strip()
            explain = lines[6].replace("Giải thích: ","")

            st.write(f"### Câu {i+1}: {question}")

            choice = st.radio(
                "Chọn đáp án:",
                ["A", "B", "C", "D"],
                key=f"quiz_{i}"
            )

            user_answers.append(choice)
            correct_answers.append((correct, explain))

            st.write(f"A. {A}")
            st.write(f"B. {B}")
            st.write(f"C. {C}")
            st.write(f"D. {D}")

        if st.button("Nộp bài", key="submit_quiz"):
            score = 0

            for i in range(len(user_answers)):
                if user_answers[i] == correct_answers[i][0]:
                    score += 1

            st.success(f"🎯 Điểm: {score}/{len(user_answers)}")

            for i in range(len(user_answers)):
                st.write("---")
                st.write(f"Câu {i+1}")
                st.write(f"Đáp án đúng: {correct_answers[i][0]}")
                st.write(f"Giải thích: {correct_answers[i][1]}")

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

    if st.button("Phân tích", key="exp_btn"):
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

            answer = ask_ai([
                {"role":"user","content":f"Giải thích kết quả thí nghiệm {exp_type}"}
            ])

            st.markdown(answer)

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

    if st.button("Chấm bài", key="grade_btn"):
        if student_answer and correct_answer:
            answer = ask_ai([
                {"role":"user","content":f"So sánh:\n{student_answer}\nĐáp án:\n{correct_answer}"}
            ])

            st.markdown(answer)

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

    # Kiểm tra có history chưa
    if "history" in st.session_state and st.session_state.history:
        # Mỗi câu hỏi là một expander, click mở ra thấy đáp án
        for i, item in enumerate(st.session_state.history):
            with st.expander(item["question"], expanded=False):
                st.markdown(item["answer"])
    else:
        st.info("Chưa có lịch sử hỏi đáp nào.")
