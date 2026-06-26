#!/bin/bash
# Called by udev on USB-CAN adapter removal
# Usage: onero-can-down.sh <can_interface>

CAN_IF="$1"
ip link set down $CAN_IF 2>/dev/null
pkill -f "slcand.*$CAN_IF" 2>/dev/null
