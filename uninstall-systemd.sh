#!/bin/bash

# Uninstall Claude Code GPT-5 Proxy systemd service
# Usage: ./uninstall-systemd.sh

set -e

SERVICE_NAME="claude-code-proxy"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo ""
echo "=========================================="
echo "  Claude Code GPT-5 Proxy - systemd 卸载"
echo "=========================================="
echo ""

# Stop service if running
if systemctl is-active --quiet $SERVICE_NAME 2>/dev/null; then
    echo "🛑 停止服务..."
    sudo systemctl stop $SERVICE_NAME
fi

# Disable service
if systemctl is-enabled --quiet $SERVICE_NAME 2>/dev/null; then
    echo "🔓 禁用开机自启..."
    sudo systemctl disable $SERVICE_NAME
fi

# Remove service file
if [ -f "$SERVICE_FILE" ]; then
    echo "🗑️  删除服务配置文件..."
    sudo rm -f "$SERVICE_FILE"
fi

# Reload systemd
echo "🔄 重载 systemd 配置..."
sudo systemctl daemon-reload

echo ""
echo "=========================================="
echo "  ✅ 卸载完成！"
echo "=========================================="
echo ""
echo "服务 $SERVICE_NAME 已完全移除。"
echo ""
