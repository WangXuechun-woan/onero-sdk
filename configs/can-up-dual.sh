#!/bin/bash
# bring-up CAN interfaces for dual A1 arms
# ACM0 -> slcan0 (right, A1R)
# ACM1 -> slcan1 (left,  A1L)
# Usage: sudo bash configs/can-up-dual.sh [right|left|both]
set -e

MODE="${1:-both}"

bring_up() {
    local IFACE="$1"
    local TTY="$2"
    echo "[CAN] Bringing up $IFACE <- /dev/$TTY"
    ip link set down "$IFACE" 2>/dev/null || true
    pkill -f "slcand.*$IFACE" 2>/dev/null || true
    sleep 0.2
    slcand -o -s8 -t hw -S 1000000 "/dev/$TTY" "$IFACE"
    sleep 0.5
    ip link set up "$IFACE"
    echo "[CAN] $IFACE UP"
}

if [[ "$MODE" == "right" || "$MODE" == "both" ]]; then
    bring_up slcan0 ttyACM0
fi

if [[ "$MODE" == "left" || "$MODE" == "both" ]]; then
    bring_up slcan1 ttyACM1
fi

echo ""
ip link show slcan0 2>/dev/null || true
ip link show slcan1 2>/dev/null || true
echo "[CAN] Done."
