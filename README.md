# 🧬 Ứng dụng Clustering dữ liệu biểu hiện Gen

Ứng dụng phân cụm dữ liệu gen với Streamlit - K-Means, Agglomerative, Spectral & Ensemble

## 📝 Mô tả

App clustering với 4 phương pháp unsupervised:

- Upload/Load data → Chọn K & method → Run Clustering → Visualize & Export

## 🚀 Setup nhanh

```bash
# Bước 1: Cài đặt
./scripts/setup.sh

# Bước 2: Chạy app
./scripts/run.sh
```

App mở tại: **http://localhost:8501**

## 📋 Setup thủ công

```bash
# 1. Tạo virtual environment
python3 -m venv venv

# 2. Kích hoạt
source venv/bin/activate

# 3. Cài packages
pip install -r requirements.txt

# 4. Chạy
streamlit run app.py
```

**Built with Streamlit** 🚀
