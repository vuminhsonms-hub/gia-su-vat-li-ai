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
                {
                    "role": "user",
                    "content": prompt
                }
            ])

            st.session_state.quiz_text = result
            st.session_state.quiz_submitted = False

            # Xóa các lựa chọn cũ khi tạo đề mới
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
                    correct = re.sub(
                        r"^Đáp án\s*:\s*", "", line, flags=re.IGNORECASE
                    ).strip().upper()
                elif re.match(r"^Giải thích\s*:\s*", line, re.IGNORECASE):
                    explain = re.sub(
                        r"^Giải thích\s*:\s*", "", line, flags=re.IGNORECASE
                    ).strip()

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

            st.write("")

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
                    selected_letter = (
                        selected.split(".")[0].strip().upper() if selected else "Chưa chọn"
                    )

                    with st.container(border=True):
                        st.markdown(f"**Câu {i+1}: {q['question']}**")
                        st.write(f"Bạn chọn: {selected_letter}")
                        st.write(f"Đáp án đúng: {q['correct']}")
                        st.write(f"Giải thích: {q['explain']}")
# ========================
# TAB 4: PHÒNG THÍ NGHIỆM AI (NÂNG CẤP: THẬT + ẢO + AI)
# ========================
with tabs[3]:
    st.subheader("🔬 Phòng thí nghiệm Vật lí AI thông minh")
    st.write("Kết hợp thí nghiệm thực, mô phỏng ảo và phân tích AI.")

    lab_mode = st.radio(
        "Chọn chế độ",
        ["Thí nghiệm thực", "Thí nghiệm ảo", "AI phân tích thông minh"],
        horizontal=True
    )

    experiment = st.selectbox(
        "Chọn thí nghiệm",
        ["Định luật Ohm", "Rơi tự do", "Dao động điều hòa / Con lắc đơn"]
    )

    # ---------------------------------
    # CHẾ ĐỘ 1: THÍ NGHIỆM THỰC
    # ---------------------------------
    if lab_mode == "Thí nghiệm thực":
        st.markdown("### 🧪 Nhập dữ liệu thí nghiệm thực tế")

        if experiment == "Định luật Ohm":
            st.info("Nhập dữ liệu điện áp U và cường độ dòng điện I theo từng lần đo.")

            x_input = st.text_input("Dữ liệu U (V), cách nhau bằng khoảng trắng", placeholder="1 2 3 4 5")
            y_input = st.text_input("Dữ liệu I (A), cách nhau bằng khoảng trắng", placeholder="0.1 0.2 0.3 0.4 0.5")

            if st.button("Phân tích thí nghiệm Ohm", key="analyze_ohm_real"):
                try:
                    x = np.array(list(map(float, x_input.split())))
                    y = np.array(list(map(float, y_input.split())))

                    if len(x) != len(y) or len(x) < 2:
                        st.error("Cần nhập cùng số lượng giá trị cho U và I, tối thiểu 2 điểm.")
                    else:
                        slope, intercept, r, _, _ = linregress(x, y)

                        fig, ax = plt.subplots()
                        ax.scatter(x, y, label="Dữ liệu đo")
                        ax.plot(x, slope * x + intercept, label="Đường hồi quy")
                        ax.set_xlabel("U (V)")
                        ax.set_ylabel("I (A)")
                        ax.set_title("Đồ thị I - U")
                        ax.legend()
                        st.pyplot(fig)

                        R_est = 1 / slope if slope != 0 else None

                        st.write(f"**Hệ số góc a = {slope:.4f}**")
                        st.write(f"**Hệ số tương quan R = {r:.4f}**")
                        if R_est is not None:
                            st.write(f"**Điện trở ước lượng R ≈ {R_est:.4f} Ω**")
                        st.write(f"**Phương trình gần đúng:** I = {slope:.4f}·U + {intercept:.4f}")

                        ai_prompt = f"""
                        Đây là kết quả thí nghiệm Định luật Ohm.

                        Dữ liệu U: {x.tolist()}
                        Dữ liệu I: {y.tolist()}
                        Hệ số góc: {slope}
                        Hệ số chặn: {intercept}
                        Hệ số tương quan R: {r}

                        Hãy:
                        1. Nhận xét xem dữ liệu có phù hợp định luật Ohm không
                        2. Giải thích ý nghĩa vật lí của hệ số góc
                        3. Nêu các nguyên nhân gây sai số
                        4. Đưa ra kết luận ngắn gọn, dễ hiểu cho học sinh THPT
                        """

                        answer = ask_ai([
                            {"role": "system", "content": "Bạn là giáo viên vật lí, giải thích rõ ràng, dễ hiểu, ngắn gọn."},
                            {"role": "user", "content": ai_prompt}
                        ])

                        st.markdown("### 🤖 Nhận xét của AI")
                        st.markdown(answer)

                except Exception as e:
                    st.error(f"Dữ liệu không hợp lệ: {e}")

        elif experiment == "Rơi tự do":
            st.info("Nhập dữ liệu thời gian t và quãng đường s để kiểm tra quy luật rơi tự do.")

            t_input = st.text_input("Dữ liệu t (s)", placeholder="0.1 0.2 0.3 0.4 0.5")
            s_input = st.text_input("Dữ liệu s (m)", placeholder="0.05 0.2 0.44 0.8 1.2")

            if st.button("Phân tích thí nghiệm rơi tự do", key="analyze_freefall_real"):
                try:
                    t = np.array(list(map(float, t_input.split())))
                    s = np.array(list(map(float, s_input.split())))

                    if len(t) != len(s) or len(t) < 2:
                        st.error("Cần nhập cùng số lượng giá trị cho t và s, tối thiểu 2 điểm.")
                    else:
                        g_est_list = []
                        for i in range(len(t)):
                            if t[i] != 0:
                                g_est_list.append(2 * s[i] / (t[i] ** 2))

                        g_mean = np.mean(g_est_list) if g_est_list else 0

                        fig, ax = plt.subplots()
                        ax.scatter(t, s, label="Dữ liệu đo")
                        ax.set_xlabel("t (s)")
                        ax.set_ylabel("s (m)")
                        ax.set_title("Đồ thị s - t")
                        ax.legend()
                        st.pyplot(fig)

                        st.write(f"**Gia tốc rơi tự do trung bình ước lượng: g ≈ {g_mean:.4f} m/s²**")

                        ai_prompt = f"""
                        Đây là kết quả thí nghiệm rơi tự do.

                        Dữ liệu thời gian t: {t.tolist()}
                        Dữ liệu quãng đường s: {s.tolist()}
                        Giá trị g trung bình ước lượng: {g_mean}

                        Hãy:
                        1. Nhận xét sự phù hợp của dữ liệu với công thức s = 1/2 g t^2
                        2. Chỉ ra sai số có thể có
                        3. Giải thích vì sao g thực nghiệm có thể khác 9.8
                        4. Kết luận ngắn gọn cho học sinh
                        """

                        answer = ask_ai([
                            {"role": "system", "content": "Bạn là giáo viên vật lí, trình bày rõ ràng, dễ hiểu."},
                            {"role": "user", "content": ai_prompt}
                        ])

                        st.markdown("### 🤖 Nhận xét của AI")
                        st.markdown(answer)

                except Exception as e:
                    st.error(f"Dữ liệu không hợp lệ: {e}")

        elif experiment == "Dao động điều hòa / Con lắc đơn":
            st.info("Nhập dữ liệu chiều dài l và chu kỳ T để kiểm tra công thức con lắc đơn.")

            l_input = st.text_input("Dữ liệu l (m)", placeholder="0.2 0.4 0.6 0.8 1.0")
            T_input = st.text_input("Dữ liệu T (s)", placeholder="0.9 1.27 1.55 1.79 2.0")

            if st.button("Phân tích thí nghiệm con lắc đơn", key="analyze_pendulum_real"):
                try:
                    l = np.array(list(map(float, l_input.split())))
                    T = np.array(list(map(float, T_input.split())))

                    if len(l) != len(T) or len(l) < 2:
                        st.error("Cần nhập cùng số lượng giá trị cho l và T, tối thiểu 2 điểm.")
                    else:
                        T2 = T ** 2
                        slope, intercept, r, _, _ = linregress(l, T2)

                        fig, ax = plt.subplots()
                        ax.scatter(l, T2, label="Dữ liệu đo")
                        ax.plot(l, slope * l + intercept, label="Đường hồi quy")
                        ax.set_xlabel("l (m)")
                        ax.set_ylabel("T² (s²)")
                        ax.set_title("Đồ thị T² - l")
                        ax.legend()
                        st.pyplot(fig)

                        g_est = (4 * np.pi ** 2) / slope if slope != 0 else None

                        st.write(f"**Hệ số góc = {slope:.4f}**")
                        st.write(f"**Hệ số tương quan R = {r:.4f}**")
                        if g_est is not None:
                            st.write(f"**Gia tốc trọng trường ước lượng: g ≈ {g_est:.4f} m/s²**")

                        ai_prompt = f"""
                        Đây là kết quả thí nghiệm con lắc đơn.

                        Dữ liệu l: {l.tolist()}
                        Dữ liệu T: {T.tolist()}
                        Hệ số góc của đồ thị T^2 theo l: {slope}
                        Hệ số tương quan R: {r}
                        Giá trị g ước lượng: {g_est}

                        Hãy:
                        1. Nhận xét mức độ phù hợp với công thức T = 2*pi*sqrt(l/g)
                        2. Giải thích tại sao dùng đồ thị T^2 theo l
                        3. Nêu các nguyên nhân sai số
                        4. Kết luận ngắn gọn, dễ hiểu
                        """

                        answer = ask_ai([
                            {"role": "system", "content": "Bạn là giáo viên vật lí THPT, trình bày dễ hiểu, có tính sư phạm."},
                            {"role": "user", "content": ai_prompt}
                        ])

                        st.markdown("### 🤖 Nhận xét của AI")
                        st.markdown(answer)

                except Exception as e:
                    st.error(f"Dữ liệu không hợp lệ: {e}")

    # ---------------------------------
    # CHẾ ĐỘ 2: THÍ NGHIỆM ẢO
    # ---------------------------------
    elif lab_mode == "Thí nghiệm ảo":
        st.markdown("### 🖥️ Mô phỏng thí nghiệm ảo")

        if experiment == "Định luật Ohm":
            R = st.slider("Điện trở R (Ω)", 1.0, 100.0, 10.0)
            U_max = st.slider("Điện áp cực đại Umax (V)", 1.0, 24.0, 12.0)

            U_values = np.linspace(0, U_max, 50)
            I_values = U_values / R

            fig, ax = plt.subplots()
            ax.plot(U_values, I_values)
            ax.set_xlabel("U (V)")
            ax.set_ylabel("I (A)")
            ax.set_title("Mô phỏng định luật Ohm")
            st.pyplot(fig)

            st.write(f"Với R = {R:.2f} Ω, ta có I = U / R.")

        elif experiment == "Rơi tự do":
            g = st.slider("Gia tốc g (m/s²)", 1.0, 20.0, 9.8)
            t_max = st.slider("Thời gian quan sát (s)", 1.0, 10.0, 5.0)

            t = np.linspace(0, t_max, 200)
            s = 0.5 * g * t**2

            fig, ax = plt.subplots()
            ax.plot(t, s)
            ax.set_xlabel("t (s)")
            ax.set_ylabel("s (m)")
            ax.set_title("Mô phỏng rơi tự do")
            st.pyplot(fig)

            st.write("Phương trình sử dụng: s = 1/2 g t²")

        elif experiment == "Dao động điều hòa / Con lắc đơn":
            L = st.slider("Chiều dài dây l (m)", 0.1, 2.0, 1.0)
            g = st.slider("Gia tốc trọng trường g (m/s²)", 1.0, 20.0, 9.8)

            T = 2 * np.pi * np.sqrt(L / g)
            t = np.linspace(0, 2 * T, 400)
            x = np.cos(2 * np.pi * t / T)

            fig, ax = plt.subplots()
            ax.plot(t, x)
            ax.set_xlabel("t (s)")
            ax.set_ylabel("Li độ chuẩn hóa")
            ax.set_title("Mô phỏng dao động con lắc đơn")
            st.pyplot(fig)

            st.write(f"Chu kỳ dao động: **T = {T:.4f} s**")

    # ---------------------------------
    # CHẾ ĐỘ 3: AI PHÂN TÍCH THÔNG MINH
    # ---------------------------------
    elif lab_mode == "AI phân tích thông minh":
        st.markdown("### 🤖 Phân tích thí nghiệm bằng AI")
        student_result = st.text_area(
            "Nhập mô tả kết quả thí nghiệm hoặc số liệu và nhận xét của bạn",
            placeholder="Ví dụ: Tôi đo được U = 1 2 3 4 5 và I = 0.11 0.19 0.31 0.39 0.52. Theo tôi dữ liệu gần đúng định luật Ohm..."
        )

        if st.button("AI phân tích thí nghiệm", key="ai_lab_analysis"):
            if not student_result.strip():
                st.warning("Vui lòng nhập nội dung thí nghiệm.")
            else:
                ai_prompt = f"""
                Học sinh đang làm thí nghiệm: {experiment}

                Nội dung học sinh cung cấp:
                {student_result}

                Hãy phản hồi theo đúng cấu trúc:
                1. Mức độ đúng của kết quả
                2. Nhận xét về sự phù hợp với lý thuyết
                3. Sai số có thể gặp
                4. Cách cải thiện thí nghiệm
                5. Kết luận ngắn gọn, dễ hiểu cho học sinh THPT
                """

                answer = ask_ai([
                    {"role": "system", "content": "Bạn là gia sư vật lí AI, phản hồi rõ ràng, dễ hiểu, có tính hướng dẫn học sinh."},
                    {"role": "user", "content": ai_prompt}
                ])

                st.markdown(answer)



# ========================
# TAB 6: CHẤM BÀI
# ========================
with tabs[4]:
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
with tabs[5]:
    st.latex("v=v_0+at")
    st.latex("s=v_0t+\\frac{1}{2}at^2")
    st.latex("I=\\frac{U}{R}")
    st.latex("T=2\\pi\\sqrt{\\frac{l}{g}}")

# ========================
# TAB 8: LỊCH SỬ
# ========================
with tabs[6]:
    st.subheader("Lịch sử học tập")

    # Kiểm tra có history chưa
    if "history" in st.session_state and st.session_state.history:
        # Mỗi câu hỏi là một expander, click mở ra thấy đáp án
        for i, item in enumerate(st.session_state.history):
            with st.expander(item["question"], expanded=False):
                st.markdown(item["answer"])
    else:
        st.info("Chưa có lịch sử hỏi đáp nào.")
