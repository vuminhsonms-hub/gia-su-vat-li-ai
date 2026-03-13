import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from openai import OpenAI

# =========================
# API KEY
# =========================

client = OpenAI(api_key="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")

# =========================
# GIAO DIỆN
# =========================

st.set_page_config(page_title="Gia sư Vật lí AI", layout="wide")

st.title("🔬 Gia sư Vật lí AI")
st.write("Hệ thống hỗ trợ học tập và thí nghiệm Vật lí thông minh cho học sinh THPT")

tabs = st.tabs([
"🤖 Hỏi đáp",
"🧠 Giải bài tập",
"📝 Tạo trắc nghiệm",
"📊 Phân tích thí nghiệm",
"🧪 Thí nghiệm mẫu",
"📚 Công thức nhanh"
])

# =========================
# TAB 1 HỎI ĐÁP
# =========================

with tabs[0]:

    st.subheader("Hỏi đáp kiến thức Vật lí")

    question = st.text_area("Nhập câu hỏi")

    if st.button("AI trả lời"):

        if question:

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là gia sư vật lí cho học sinh THPT Việt Nam. Giải thích dễ hiểu."
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            )

            answer = response.choices[0].message.content

            st.write(answer)

# =========================
# TAB 2 GIẢI BÀI TẬP
# =========================

with tabs[1]:

    st.subheader("AI giải bài tập Vật lí")

    problem = st.text_area("Nhập bài tập")

    if st.button("Giải bài"):

        if problem:

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Giải bài tập vật lí cho học sinh THPT. Trình bày từng bước."
                    },
                    {
                        "role": "user",
                        "content": problem
                    }
                ]
            )

            answer = response.choices[0].message.content

            st.write(answer)

# =========================
# TAB 3 TRẮC NGHIỆM
# =========================

with tabs[2]:

    st.subheader("Tạo câu hỏi trắc nghiệm")

    topic = st.text_input("Chủ đề")

    number = st.slider("Số câu hỏi",1,10,5)

    if st.button("Tạo câu hỏi"):

        prompt = f"Tạo {number} câu hỏi trắc nghiệm vật lí THPT về chủ đề {topic}, có 4 đáp án và đáp án đúng."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"user","content":prompt}
            ]
        )

        answer = response.choices[0].message.content

        st.write(answer)

# =========================
# TAB 4 PHÂN TÍCH THÍ NGHIỆM
# =========================

with tabs[3]:

    st.subheader("Phân tích dữ liệu thí nghiệm")

    st.write("Nhập dữ liệu cách nhau bằng dấu cách")

    x_input = st.text_input("Giá trị X")

    y_input = st.text_input("Giá trị Y")

    if st.button("Phân tích dữ liệu"):

        try:

            x = np.array(list(map(float,x_input.split())))
            y = np.array(list(map(float,y_input.split())))

            slope, intercept, r, p, std = linregress(x,y)

            fig, ax = plt.subplots()

            ax.scatter(x,y)

            line = slope*x + intercept

            ax.plot(x,line)

            ax.set_xlabel("X")

            ax.set_ylabel("Y")

            ax.set_title("Đồ thị thí nghiệm")

            st.pyplot(fig)

            st.write("### Kết quả")

            st.write("Hệ số góc:",round(slope,3))

            st.write("Hệ số tương quan R:",round(r,3))

            st.latex(f"y={slope:.2f}x+{intercept:.2f}")

            explain_prompt = f"""
            Phân tích đồ thị vật lí sau:

            X={x}
            Y={y}

            Hãy giải thích ý nghĩa vật lí.
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":explain_prompt}]
            )

            st.write("### Nhận xét AI")

            st.write(response.choices[0].message.content)

        except:

            st.error("Dữ liệu không hợp lệ")

# =========================
# TAB 5 THÍ NGHIỆM
# =========================

with tabs[4]:

    st.subheader("Một số thí nghiệm vật lí")

    exp = st.selectbox(
        "Chọn thí nghiệm",
        [
            "Định luật Ohm",
            "Con lắc đơn",
            "Rơi tự do"
        ]
    )

    if exp=="Định luật Ohm":

        st.write("""
Mục đích:
Kiểm tra mối quan hệ giữa U và I.

Dụng cụ:
nguồn điện, điện trở, ampe kế, vôn kế.

Tiến hành:
1 thay đổi U
2 đo I
3 vẽ đồ thị I-U

Kết luận:
Đồ thị là đường thẳng.
""")

    if exp=="Con lắc đơn":

        st.write("""
Chu kì:

T = 2π √(l/g)

Đo chiều dài dây và chu kì dao động.
""")

    if exp=="Rơi tự do":

        st.write("""
Công thức:

s = 1/2 g t²

g ≈ 9.8 m/s²
""")

# =========================
# TAB 6 CÔNG THỨC
# =========================

with tabs[5]:

    st.subheader("Công thức nhanh")

    formula = st.selectbox(
        "Chọn chủ đề",
        [
            "Chuyển động thẳng",
            "Điện học",
            "Dao động"
        ]
    )

    if formula=="Chuyển động thẳng":

        st.latex("v=v_0+at")
        st.latex("s=v_0t+1/2at^2")

    if formula=="Điện học":

        st.latex("I=U/R")
        st.latex("P=UI")

    if formula=="Dao động":

        st.latex("T=2\pi\sqrt{l/g}")
        st.latex("f=1/T")