import streamlit as st
import pandas as pd
import numpy as np

# Tiêu đề chính (lớn nhất)
st.title("🎓 Học Streamlit - Từ Cơ Bản đến Nâng Cao")

# Tiêu đề phụ
st.header("📚 Bước 1: Hiển thị Text")

# Text thường
st.write("Đây là cách đơn giản nhất để hiển thị text trong Streamlit")

# Markdown (hỗ trợ formatting)
st.markdown("""
**Streamlit** là framework Python để tạo web app *cực kỳ đơn giản*!

- ✅ Không cần HTML/CSS
- ✅ Không cần JavaScript  
- ✅ Chỉ cần Python thuần
""")

# Đường kẻ ngang
st.markdown("---")

st.header("📊 Bước 2: Hiển thị Dữ liệu")

# Tạo dữ liệu mẫu
data = {
    'Tên': ['An', 'Bình', 'Chi', 'Dũng'],
    'Tuổi': [25, 30, 22, 28],
    'Điểm': [8.5, 9.0, 7.5, 8.8]
}
df = pd.DataFrame(data)

# Hiển thị DataFrame
st.write("**Dữ liệu sinh viên:**")
st.dataframe(df)

# Hiển thị metrics
st.write("**Thống kê:**")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Số sinh viên", len(df))
with col2:
    st.metric("Điểm TB", f"{df['Điểm'].mean():.2f}")
with col3:
    st.metric("Tuổi TB", f"{df['Tuổi'].mean():.1f}")

st.markdown("---")

# ============================================================================
# BƯỚC 3: WIDGETS TƯƠNG TÁC
# ============================================================================

st.header("🎮 Bước 3: Widgets Tương tác")

# Text Input
name = st.text_input("Nhập tên của bạn:", placeholder="Ví dụ: Nguyễn Văn A")
if name:
    st.success(f"Xin chào {name}! 👋")

# Number Input
age = st.number_input("Nhập tuổi:", min_value=1, max_value=100, value=25)
st.info(f"Bạn {age} tuổi")

# Slider
score = st.slider("Chọn điểm số:", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
st.write(f"Điểm của bạn: **{score}**")

# Selectbox (Dropdown)
subject = st.selectbox(
    "Chọn môn học yêu thích:",
    ["Toán", "Lý", "Hóa", "Sinh", "Văn"]
)
st.write(f"Bạn chọn: **{subject}**")

# Checkbox
agree = st.checkbox("Tôi đồng ý với điều khoản")
if agree:
    st.success("✅ Cảm ơn bạn đã đồng ý!")

st.markdown("---")

# ============================================================================
# BƯỚC 4: BUTTON VÀ SESSION STATE
# ============================================================================

st.header("🔘 Bước 4: Button và Session State")

st.write("**Session State** giúp lưu trữ dữ liệu giữa các lần tương tác")

# Khởi tạo session state
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# Button
if st.button("Click tôi! 🎯"):
    st.session_state.counter += 1

# Hiển thị counter
st.metric("Số lần click", st.session_state.counter)

# Reset button
if st.button("Reset"):
    st.session_state.counter = 0
    st.rerun()  # Chạy lại app

st.markdown("---")

# ============================================================================
# BƯỚC 5: LAYOUT - COLUMNS & TABS
# ============================================================================

st.header("📐 Bước 5: Layout - Columns & Tabs")

# Columns
st.subheader("Chia cột:")
col1, col2, col3 = st.columns(3)

with col1:
    st.info("**Cột 1**")
    st.write("Nội dung cột 1")

with col2:
    st.success("**Cột 2**")
    st.write("Nội dung cột 2")

with col3:
    st.warning("**Cột 3**")
    st.write("Nội dung cột 3")

# Tabs
st.subheader("Tabs:")
tab1, tab2, tab3 = st.tabs(["🏠 Tab 1", "📊 Tab 2", "⚙️ Tab 3"])

with tab1:
    st.write("Đây là nội dung Tab 1")
    st.image("https://via.placeholder.com/400x200?text=Tab+1", use_container_width=True)

with tab2:
    st.write("Đây là nội dung Tab 2")
    # Tạo chart đơn giản
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['A', 'B', 'C']
    )
    st.line_chart(chart_data)

with tab3:
    st.write("Đây là nội dung Tab 3")
    st.code("""
# Code example
def hello():
    print("Hello Streamlit!")
    """, language="python")

st.markdown("---")

# ============================================================================
# BƯỚC 6: SIDEBAR
# ============================================================================

st.header("📱 Bước 6: Sidebar")

# Thêm widgets vào sidebar
st.sidebar.title("⚙️ Cài đặt")
st.sidebar.markdown("---")

sidebar_option = st.sidebar.radio(
    "Chọn chế độ:",
    ["Chế độ 1", "Chế độ 2", "Chế độ 3"]
)

st.sidebar.slider("Độ khó:", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.info("💡 Sidebar giúp tổ chức giao diện tốt hơn!")

# Hiển thị lựa chọn từ sidebar
st.write(f"Bạn đã chọn: **{sidebar_option}**")

st.markdown("---")

# ============================================================================
# BƯỚC 7: STATUS MESSAGES
# ============================================================================

st.header("💬 Bước 7: Status Messages")

col1, col2 = st.columns(2)

with col1:
    st.success("✅ Success message")
    st.info("ℹ️ Info message")

with col2:
    st.warning("⚠️ Warning message")
    st.error("❌ Error message")

# Progress bar
st.write("**Progress Bar:**")
progress = st.progress(0)
import time

if st.button("Chạy Progress"):
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)
    st.success("Hoàn thành!")

st.markdown("---")

# ============================================================================
# BƯỚC 8: EXPANDER (Ẩn/Hiện nội dung)
# ============================================================================

st.header("📦 Bước 8: Expander")

with st.expander("🔍 Click để xem thêm"):
    st.write("""
    **Expander** giúp ẩn/hiện nội dung để giao diện gọn gàng hơn.
    
    Bạn có thể đặt bất kỳ nội dung nào vào đây:
    - Text
    - Images
    - Charts
    - Code
    """)
    
    st.code("""
# Example
with st.expander("Tiêu đề"):
    st.write("Nội dung ẩn")
    """, language="python")

st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
---
### 🎉 Chúc mừng!

Bạn đã học xong các tính năng cơ bản của Streamlit!

**Bước tiếp theo:**
1. Đọc file `docs/TUTORIAL_VI.md` để hiểu sâu hơn
2. Thử modify code này và xem kết quả
3. Tạo app của riêng bạn!

**Tài liệu:**
- 📚 [Streamlit Docs](https://docs.streamlit.io)
- 🎨 [Gallery](https://streamlit.io/gallery)
- 💬 [Forum](https://discuss.streamlit.io)
""")

st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Made with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
