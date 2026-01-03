#!/bin/bash

echo "🧬 Starting Streamlit App..."

# Kiểm tra venv
if [ ! -d "venv" ]; then
    echo "❌ Lỗi: Chưa setup!"
    echo "Chạy: ./scripts/setup.sh"
    exit 1
fi

# Chạy app
source venv/bin/activate
echo "🚀 App: http://localhost:8501"
echo "💡 Nhấn Ctrl+C để thoát"
echo ""
streamlit run app.py

