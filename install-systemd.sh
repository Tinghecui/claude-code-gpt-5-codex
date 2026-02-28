#!/bin/bash

# Install Claude Code GPT-5 Proxy as a systemd service
# Usage: ./install-systemd.sh

set -e

SERVICE_NAME="claude-code-proxy"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo ""
echo "=========================================="
echo "  Claude Code GPT-5 Proxy - systemd 安装"
echo "=========================================="
echo ""

# Detect current directory and user
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
CURRENT_USER="$(whoami)"

# Detect uv path
UV_BIN="$(which uv 2>/dev/null || echo "")"
if [ -z "$UV_BIN" ]; then
    echo "❌ 错误: 找不到 uv"
    echo ""
    echo "请先安装 uv:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    exit 1
fi
UV_PATH="$(dirname "$UV_BIN")"

echo "🔧 检测到配置:"
echo "   工作目录: $WORKDIR"
echo "   运行用户: $CURRENT_USER"
echo "   uv 路径:  $UV_BIN"
echo ""

# Check .env file
if [ ! -f "$WORKDIR/.env" ]; then
    echo "⚠️  警告: .env 文件不存在"
    echo "   请确保已配置环境变量 (OPENAI_API_KEY 等)"
    echo ""
fi

# Check if service template exists
if [ ! -f "$WORKDIR/systemd/claude-code-proxy.service" ]; then
    echo "❌ 错误: 服务模板文件不存在"
    echo "   预期路径: $WORKDIR/systemd/claude-code-proxy.service"
    exit 1
fi

# Stop existing service if running
if systemctl is-active --quiet $SERVICE_NAME 2>/dev/null; then
    echo "📦 停止现有服务..."
    sudo systemctl stop $SERVICE_NAME
fi

# Generate service file with replaced variables
echo "📝 生成服务配置文件..."
sed -e "s|__USER__|$CURRENT_USER|g" \
    -e "s|__WORKDIR__|$WORKDIR|g" \
    -e "s|__UV_PATH__|$UV_PATH|g" \
    -e "s|__UV_BIN__|$UV_BIN|g" \
    "$WORKDIR/systemd/claude-code-proxy.service" | sudo tee "$SERVICE_FILE" > /dev/null

sudo chmod 644 "$SERVICE_FILE"

# Reload systemd and enable service
echo "🔄 重载 systemd 配置..."
sudo systemctl daemon-reload

echo "✅ 启用开机自启..."
sudo systemctl enable $SERVICE_NAME

echo ""
echo "=========================================="
echo "  ✅ 安装完成！"
echo "=========================================="
echo ""
echo "常用命令:"
echo "  启动服务:  sudo systemctl start $SERVICE_NAME"
echo "  停止服务:  sudo systemctl stop $SERVICE_NAME"
echo "  重启服务:  sudo systemctl restart $SERVICE_NAME"
echo "  查看状态:  sudo systemctl status $SERVICE_NAME"
echo "  实时日志:  sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "立即启动服务:"
echo "  sudo systemctl start $SERVICE_NAME"
echo ""
