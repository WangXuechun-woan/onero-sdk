"""Optional SLCAN bulk serial I/O; retains python-can's receive parser."""

from __future__ import annotations

import time

from can import CanOperationError
from can.interfaces.slcan import slcanBus


class BufferedSlcanBus(slcanBus):
    """Read available serial bytes together instead of issuing one read per byte."""

    _READ_CHUNK_SIZE = 4096
    _MAX_UNTERMINATED_LINE = 4096

    def send_batch(self, messages) -> None:
        """Send classic data frames in order using one serial write and flush."""
        lines = []
        for msg in messages:
            if msg.is_fd or msg.is_remote_frame or msg.is_error_frame:
                raise ValueError("SLCAN batch sending requires classic CAN data frames")
            if msg.dlc != len(msg.data) or not 0 <= msg.dlc <= 8:
                raise ValueError("Invalid classic CAN data length")
            if msg.is_extended_id:
                header = f"T{msg.arbitration_id:08X}{msg.dlc:d}"
            else:
                header = f"t{msg.arbitration_id:03X}{msg.dlc:d}"
            lines.append((header + msg.data.hex().upper()).encode() + self.LINE_TERMINATOR)
        if not lines:
            return
        payload = b"".join(lines)
        try:
            # Match the normal send(message) timeout policy.
            if self.serialPortOrig.write_timeout is not None:
                self.serialPortOrig.write_timeout = None
            written = self.serialPortOrig.write(payload)
            if written != len(payload):
                raise CanOperationError("Incomplete SLCAN batch write")
            self.serialPortOrig.flush()
        except CanOperationError:
            raise
        except Exception as exc:
            raise CanOperationError("Could not write SLCAN batch") from exc

    def _read(self, timeout):
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        attempted_read = False
        try:
            while True:
                endings = [index for index in (self._buffer.find(self._OK),
                                                self._buffer.find(self._ERROR)) if index >= 0]
                if endings:
                    end = min(endings) + 1
                    line = bytes(self._buffer[:end]).decode()
                    del self._buffer[:end]
                    return line
                if len(self._buffer) >= self._MAX_UNTERMINATED_LINE:
                    raise CanOperationError('SLCAN input exceeded the unterminated-line limit')
                # A zero-timeout poll may read bytes already waiting once, but
                # never waits for the first byte. Preserve partial lines.
                now = time.monotonic()
                if attempted_read and deadline is not None and now >= deadline:
                    return None
                waiting = self.serialPortOrig.in_waiting
                if waiting:
                    chunk = self.serialPortOrig.read(min(waiting, self._READ_CHUNK_SIZE))
                else:
                    if deadline is not None and now >= deadline:
                        return None
                    old_timeout = self.serialPortOrig.timeout
                    remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                    limit_read = remaining is not None and (old_timeout is None or old_timeout > remaining)
                    try:
                        if limit_read:
                            self.serialPortOrig.timeout = remaining
                        chunk = self.serialPortOrig.read(1)
                    finally:
                        if limit_read:
                            self.serialPortOrig.timeout = old_timeout
                attempted_read = True
                self._buffer.extend(chunk)
        except CanOperationError:
            raise
        except Exception as exc:
            raise CanOperationError('Could not read from SLCAN serial device') from exc
