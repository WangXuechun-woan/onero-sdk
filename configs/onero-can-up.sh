#!/bin/bash
# Called by udev on USB-CAN adapter plug-in
# Usage: onero-can-up.sh <can_interface> <tty_device>

CAN_IF="$1"
TTY_DEV="$2"

# Kill any existing slcand on this interface
ip link set down $CAN_IF 2>/dev/null
pkill -f "slcand.*$CAN_IF" 2>/dev/null
sleep 0.2

# Use systemd-run to escape udev's cgroup (udev kills all processes in its cgroup after timeout)
systemd-run --no-block --unit=slcand-$CAN_IF --property=Type=forking -- slcand -o -s8 -t hw -S 1000000 /dev/$TTY_DEV $CAN_IF

sleep 0.5

# Bring up interface
ip link set up $CAN_IF
