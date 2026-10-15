#!/usr/bin/env bash
set -e

# ==============================================================================
# Wispo Offline Tarball Uninstaller
# ==============================================================================

if [ "$EUID" -ne 0 ]; then
    echo "Error: This uninstaller must be run with root privileges." >&2
    echo "Usage: sudo /opt/wispo/uninstall.sh" >&2
    exit 1
fi

echo "==> Uninstalling Wispo..."

# 1. Terminate any running Wispo process
pkill -f "wispo.service" 2>/dev/null || true
pkill -f "wispo.cli" 2>/dev/null || true

# 2. Remove launcher wrapper
rm -f /usr/bin/wispo

# 3. Remove the uinput rule left by pre-0.1.0 tarball releases.
if [ -f /etc/udev/rules.d/99-wispo-uinput.rules ]; then
    rm -f /etc/udev/rules.d/99-wispo-uinput.rules
    if command -v udevadm >/dev/null 2>&1; then
        udevadm control --reload-rules || true
    fi
fi

# 4. Remove desktop entry
rm -f /usr/share/applications/com.wispo.Wispo.desktop

# 5. Remove icons
for sz in 16 32 48 64 128 256 512; do
    rm -f "/usr/share/icons/hicolor/${sz}x${sz}/apps/com.wispo.Wispo.png"
    rm -f "/usr/share/icons/hicolor/${sz}x${sz}/apps/com.wispo.Settings.png"
done

# 6. Refresh caches
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q /usr/share/applications || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -q -t /usr/share/icons/hicolor || true
fi

# 7. Remove installation payload
rm -rf /opt/wispo

echo "==> Wispo uninstalled cleanly."
