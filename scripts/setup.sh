#!/bin/bash

set -e  # Exit on error

echo "🧬 Gene Expression Clustering - Setup"
echo "======================================"
echo ""

# ============================================
# 1. Kiểm tra Python3
# ============================================
echo "📌 [1/4] Kiểm tra Python3..."
if ! command -v python3 &> /dev/null; then
    echo "❌ THIẾU: Python3"
    echo ""
    echo "Vui lòng cài đặt:"
    echo "  sudo apt update"
    echo "  sudo apt install python3"
    exit 1
fi
echo "   ✅ Python: $(python3 --version)"

# ============================================
# 2. Kiểm tra pip
# ============================================
echo ""
echo "📌 [2/4] Kiểm tra pip..."
if ! python3 -m pip --version &> /dev/null; then
    echo "❌ THIẾU: python3-pip"
    echo ""
    echo "Vui lòng cài đặt:"
    echo "  sudo apt install python3-pip"
    exit 1
fi
echo "   ✅ pip: $(python3 -m pip --version)"

# ============================================
# 3. Tạo Virtual Environment
# ============================================
echo ""
echo "📌 [3/4] Tạo virtual environment..."

# Xóa venv cũ nếu có
if [ -d "venv" ]; then
    echo "   �️  Xóa venv cũ..."
    rm -rf venv
fi

# Tạo venv mới
if ! python3 -m venv venv 2>/dev/null; then
    echo "❌ THIẾU: python3-venv"
    echo ""
    echo "Vui lòng cài đặt:"
    echo "  sudo apt install python3-venv"
    exit 1
fi
echo "   ✅ Đã tạo virtual environment"

# ============================================
# 4. Cài đặt Dependencies
# ============================================
echo ""
echo "� [4/4] Cài đặt Python packages..."
echo "   (Quá trình này có thể mất 1-2 phút...)"
echo ""

# Activate venv
source venv/bin/activate

# Upgrade pip (hiển thị progress)
echo "   📦 Upgrade pip..."
python3 -m pip install --upgrade pip --progress-bar on

echo ""
echo "   📦 Cài đặt dependencies..."
# Cài đặt từng package để dễ debug
python3 -m pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn openpyxl scipy --progress-bar on

# ============================================
# Hoàn tất
# ============================================
echo ""
echo "======================================"
echo "✅ Setup hoàn tất!"
echo ""
echo "🚀 Chạy app:"
echo "   ./scripts/run.sh"
echo ""
echo "📚 Đọc hướng dẫn:"
echo "   docs/QUICKSTART.md"
echo "======================================"
