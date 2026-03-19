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
    "🤖 Hỏi đáp 12",
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
# TAB 2: GIẢI BÀI (ĐÃ FIX LỖI HIỂN THỊ & LỊCH SỬ)
# ========================
with tabs[1]:
    problem = st.text_area("Nhập bài tập", key="input_problem_tab2")

    col1, col2, col3 = st.columns(3)
    # Khởi tạo prompt_ai để tránh trùng tên với biến hệ thống
    prompt_ai = None

    if col1.button("💡 Gợi ý", key="hint_btn"):
        prompt_ai = f"Gợi ý cách làm: {problem}"
    if col2.button("🧩 Bước 1", key="step1_btn"):
        prompt_ai = f"Giải bước đầu tiên: {problem}"
    if col3.button("✅ Giải đầy đủ", key="full_btn"):
        prompt_ai = f"Giải chi tiết có công thức: {problem}"

    if prompt_ai and problem.strip():
        # Áp dụng bộ quy tắc nghiêm ngặt giống hệt Tab Hỏi đáp
        answer = ask_ai([
            {"role": "system", "content": """
                 Bạn là gia sư vật lí chuyên nghiệp.
                 Quy tắc trình bày công thức:
                 - Dùng $...$ cho công thức nằm cùng dòng.
                 - Dùng $$...$$ cho công thức cần xuống dòng riêng biệt.
                 - TUYỆT ĐỐI KHÔNG dùng ký hiệu \[ \] hoặc \( \) hoặc dấu ngoặc vuông đơn lẻ [ ] để bao quanh công thức.
                 """},
            {"role": "user", "content": prompt_ai}
        ])

        # Bước bảo hiểm cuối cùng: Ép định dạng bằng code (nếu AI lỡ quên)
        # Thay thế các biến thể ngoặc vuông/ngoặc đơn thành dấu $
        clean_answer = answer.replace(r"\[", "$$").replace(r"\]", "$$").replace(r"\(", "$").replace(r"\)", "$")
        
        # Lưu vào history theo đúng định dạng dict để Tab 8 đọc được 
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({"question": problem, "answer": clean_answer})

        # Hiển thị kết quả sạch
        st.markdown(clean_answer)

# ========================
# TAB 3: TRẮC NGHIỆM
# ========================
with tabs[2]:
    import re

    st.markdown("""
    <style>
    .quiz-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .quiz-question {
        font-size: 22px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 10px;
    }
    .quiz-result {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

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

Bắt buộc đúng định dạng này:
Câu 1: Nội dung câu hỏi
A. Nội dung đáp án A
B. Nội dung đáp án B
C. Nội dung đáp án C
D. Nội dung đáp án D
Đáp án: A
Giải thích: Nội dung giải thích

Câu 2: Nội dung câu hỏi
A. ...
B. ...
C. ...
D. ...
Đáp án: B
Giải thích: ...

Không viết thêm lời mở đầu.
Không viết thêm lời kết.
"""

            result = ask_ai([
                {
                    "role": "system",
                    "content": "Bạn là giáo viên vật lí. Hãy tạo đề trắc nghiệm đúng định dạng yêu cầu."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ])

            st.session_state.quiz_text = result
            st.session_state.quiz_submitted = False

            # xóa lựa chọn cũ
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

    if "quiz_text" in st.session_state:
        parsed_questions = parse_quiz(st.session_state.quiz_text)

        if not parsed_questions:
            st.error("Không đọc được đề theo đúng định dạng.")
            with st.expander("Xem nội dung AI trả về"):
                st.code(st.session_state.quiz_text)
        else:
            st.info(f"Đã tạo {len(parsed_questions)} câu hỏi.")

            for i, q in enumerate(parsed_questions):
                st.markdown('<div class="quiz-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="quiz-question">Câu {i+1}: {q["question"]}</div>',
                    unsafe_allow_html=True
                )

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

                st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Nộp bài", key="submit_quiz", use_container_width=True):
                st.session_state.quiz_submitted = True

            if st.session_state.get("quiz_submitted", False):
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

                    if selected:
                        selected_letter = selected.split(".")[0].strip().upper()
                    else:
                        selected_letter = "Chưa chọn"

                    st.markdown('<div class="quiz-result">', unsafe_allow_html=True)
                    st.markdown(f"**Câu {i+1}: {q['question']}**")
                    st.write(f"Bạn chọn: {selected_letter}")
                    st.write(f"Đáp án đúng: {q['correct']}")
                    st.write(f"Giải thích: {q['explain']}")
                    st.markdown("</div>", unsafe_allow_html=True)


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
