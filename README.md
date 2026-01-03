# � ML Demo App

Ứng dụng demo Machine Learning với Streamlit - Đơn giản, Clean & Professional

## 📝 Mô tả

App demo Linear Regression với giao diện 2 cột responsive:
- Upload/Load data → Train model → Visualize → Predict

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

## ⚠️ Troubleshooting

```bash
# Thiếu python3-venv
sudo apt install python3-venv

# Thiếu pip
sudo apt install python3-pip

# Port 8501 bị chiếm
streamlit run app.py --server.port 8502
```

## 📚 Tài liệu

- `docs/TUTORIAL_VI.md` - Hướng dẫn Streamlit chi tiết
- `app.py` - Source code với best practices

## 🎯 Mục đích

Project này để:
- Học Streamlit từ cơ bản đến nâng cao
- Demo ML app với clean code structure
- Best practices cho Streamlit development

---

**Built with Streamlit** 🚀
