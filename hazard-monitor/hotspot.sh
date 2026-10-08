#!/usr/bin/env bash
# Turns the Pi's Wi-Fi into a hotspot the ESP32 nodes join.
#
#   bash hotspot.sh on       create/start the hotspot (also starts on every boot)
#   bash hotspot.sh off      stop it and go back to normal Wi-Fi
#   bash hotspot.sh status   show the hotspot and the devices connected to it
#
# The Pi is always 192.168.4.1 on this network, so the nodes send to
#   http://192.168.4.1:3000/api/sensor-data
#
# Run setup.sh FIRST (it needs internet). While the hotspot is on, the Pi's
# Wi-Fi cannot reach the internet; a LAN cable to the router still can.
# Needs Raspberry Pi OS Bookworm or newer (NetworkManager / nmcli).
set -euo pipefail

SSID="${HOTSPOT_SSID:-HAZARD-NET}"
PASSWORD="${HOTSPOT_PASSWORD:-hazard1234}"   # at least 8 characters
PI_IP="192.168.4.1"
COUNTRY="${WIFI_COUNTRY:-IN}"                # Wi-Fi country code
CON="hazard-hotspot"
IFACE="wlan0"

die() { printf '\n\033[1;31mXX  %s\033[0m\n\n' "$*"; exit 1; }
ok()  { printf '\033[1;32mOK\033[0m  %s\n' "$*"; }

command -v nmcli >/dev/null || die "nmcli not found. Use Raspberry Pi OS Bookworm (2023 or newer)."
[ "${#PASSWORD}" -ge 8 ] || die "HOTSPOT_PASSWORD must be at least 8 characters."
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

case "${1:-}" in
  on)
    $SUDO rfkill unblock wifi 2>/dev/null || true
    command -v raspi-config >/dev/null && $SUDO raspi-config nonint do_wifi_country "$COUNTRY" 2>/dev/null || true
    $SUDO nmcli connection delete "$CON" >/dev/null 2>&1 || true
    # 2.4 GHz (band bg): the ESP32 cannot see 5 GHz networks
    $SUDO nmcli connection add type wifi ifname "$IFACE" con-name "$CON" autoconnect yes ssid "$SSID" \
      802-11-wireless.mode ap 802-11-wireless.band bg 802-11-wireless.channel 6 \
      wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$PASSWORD" \
      ipv4.method shared ipv4.addresses "$PI_IP/24" ipv6.method disabled \
      connection.autoconnect-priority 100 >/dev/null
    printf 'Starting hotspot... (an SSH session over Wi-Fi will drop now - reconnect on "%s")\n' "$SSID"
    $SUDO nmcli connection up "$CON" >/dev/null
    ok "Hotspot ON   name: $SSID   password: $PASSWORD"
    ok "Pi address:  $PI_IP   ESP32 URL: http://$PI_IP:3000/api/sensor-data"
    ok "Dashboard:   http://$PI_IP:3000/dashboard  (join $SSID first)"
    ;;
  off)
    $SUDO nmcli connection delete "$CON" >/dev/null 2>&1 || true
    ok "Hotspot OFF - the Pi will rejoin its saved Wi-Fi network"
    ;;
  status)
    nmcli -f NAME,DEVICE,STATE connection show --active
    printf '\nDevices on the hotspot:\n'
    ip neigh show dev "$IFACE" 2>/dev/null | grep -v FAILED || echo "  none"
    ;;
  *)
    echo "usage: bash hotspot.sh on | off | status"
    echo "change name/password:  HOTSPOT_SSID=MyNet HOTSPOT_PASSWORD=secret123 bash hotspot.sh on"
    exit 1
    ;;
esac
