"""
onero SDK Debug UI — HTTP server
=================================
Wraps one or two Arm instances and exposes a REST + SSE API consumed by the
web frontend (static/index.html + static/app.js).

Architecture
------------
ArmAdapter
    Wraps a single ``Arm`` instance.  All blocking SDK calls (movej, movel,
    home, stop) are dispatched to a single-worker thread so HTTP handlers
    never block.  Real-time servo (servoj 100 Hz) and freedrive run in their
    own daemon threads.

LabServer
    ThreadingHTTPServer subclass.  Routes:

      GET  /                          index.html
      GET  /static/<file>             CSS / JS assets
      GET  /assets/urdf/<model>.urdf  URDF from onero_sdk package
      GET  /assets/meshes/<path>      STL meshes from onero_sdk package
      GET  /api/stream[?hz=N]         SSE — pushes full state at N Hz (≤50)
      GET  /[<side>/]api/state        One-shot state JSON (polling fallback)

      POST /[<side>/]api/enable
      POST /[<side>/]api/disable
      POST /[<side>/]api/stop
      POST /[<side>/]api/home
      POST /[<side>/]api/set_stiffness   {"stiffness": "stiff"|"medium"|"soft"}
      POST /[<side>/]api/movej            {"joints":[…], "speed":0.5}
      POST /[<side>/]api/movel            {"position":[x,y,z], "rotation":[…], "speed":0.1}
      POST /[<side>/]api/servo/start      {"kp":50, "kd":1.0}
      POST /[<side>/]api/servo/target     {"joints":[…], "velocities":[…]}
      POST /[<side>/]api/servo/stop
      POST /[<side>/]api/freedrive/start  {"damping":0.5}
      POST /[<side>/]api/freedrive/stop

Single-arm mode uses prefix ``/api/``; dual-arm uses ``/left/api/`` and
``/right/api/``.  The SSE stream at ``/api/stream`` always sends both sides
(single-arm mode fills the unused side with zeros).
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Package-internal asset paths
# ---------------------------------------------------------------------------
_PKG_DIR = pathlib.Path(__file__).parent.parent   # onero_sdk/
_STATIC_DIR = pathlib.Path(__file__).parent / "static"
_URDF_DIR = _PKG_DIR / "urdf" / "A1"
_MESH_DIR = _PKG_DIR / "meshes"


# ---------------------------------------------------------------------------
# Rotation matrix → quaternion (pure numpy, no scipy)
# ---------------------------------------------------------------------------
def _rot_to_quat(R: np.ndarray) -> List[float]:
    """Convert 3×3 rotation matrix to [qx, qy, qz, qw]."""
    t = float(R[0, 0] + R[1, 1] + R[2, 2])
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return [float((R[2, 1] - R[1, 2]) * s),
                float((R[0, 2] - R[2, 0]) * s),
                float((R[1, 0] - R[0, 1]) * s),
                float(0.25 / s)]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
        return [float(0.25 * s),
                float((R[0, 1] + R[1, 0]) / s),
                float((R[0, 2] + R[2, 0]) / s),
                float((R[2, 1] - R[1, 2]) / s)]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
        return [float((R[0, 1] + R[1, 0]) / s),
                float(0.25 * s),
                float((R[1, 2] + R[2, 1]) / s),
                float((R[0, 2] - R[2, 0]) / s)]
    else:
        s = 2.0 * np.sqrt(max(0.0, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
        return [float((R[0, 2] + R[2, 0]) / s),
                float((R[1, 2] + R[2, 1]) / s),
                float(0.25 * s),
                float((R[1, 0] - R[0, 1]) / s)]


# ---------------------------------------------------------------------------
# ArmAdapter — wraps one Arm instance
# ---------------------------------------------------------------------------
class ArmAdapter:
    """
    Thread-safe adapter around ``onero_sdk.Arm``.

    Blocking motions (movej, movel, home) are dispatched to a single worker
    thread so HTTP handlers return immediately.  A new command cancels any
    running motion via ``arm.stop()``.
    """

    def __init__(self, arm, side: str = "right") -> None:
        self._arm = arm
        self.side = side
        self._lock = threading.Lock()

        # Worker thread for blocking motions
        self._cmd: Optional[dict] = None
        self._cmd_event = threading.Event()
        self._worker_alive = True
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name=f"debugui-worker-{side}")
        self._worker.start()

        # Servo loop (servoj 100 Hz)
        self._servo_active = False
        self._servo_stop = threading.Event()
        self._servo_target: List[float] = []
        self._servo_vel: List[float] = []
        self._servo_kp: Optional[float] = None
        self._servo_kd: Optional[float] = None
        self._servo_thread: Optional[threading.Thread] = None
        self._servo_lock = threading.Lock()

        # Last known state for fast SSE reads
        self._last_joints: List[float] = [0.0] * arm.num_joints
        self._last_torques: List[float] = [0.0] * arm.num_joints
        self._last_pos: List[float] = [0.0, 0.0, 0.0]
        self._last_quat: List[float] = [0.0, 0.0, 0.0, 1.0]
        self._last_msg: str = "ready"
        self._last_fault: str = ""

    # ------------------------------------------------------------------
    # Worker thread (blocking motions: movej, movel, home)
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while self._worker_alive:
            self._cmd_event.wait()
            self._cmd_event.clear()
            with self._lock:
                cmd = self._cmd
                self._cmd = None
            if cmd is None or not self._worker_alive:
                continue
            try:
                self._run_cmd(cmd)
            except Exception as exc:
                self._last_fault = str(exc)
                self._last_msg = f"error: {exc}"
                logger.warning("[debugui %s] cmd error: %s", self.side, exc)

    def _run_cmd(self, cmd: dict) -> None:
        name = cmd["name"]
        arm = self._arm
        if name == "movej":
            self._last_msg = f"movej running…"
            self._last_fault = ""
            state = arm.movej(cmd["joints"],
                              speed=cmd.get("speed"),
                              duration=cmd.get("duration"))
            self._update_state(state)
            self._last_msg = "movej done"
        elif name == "movel":
            self._last_msg = "movel running…"
            self._last_fault = ""
            state = arm.movel(cmd["position"],
                              rotation=cmd.get("rotation"),
                              speed=cmd.get("speed"),
                              duration=cmd.get("duration"))
            self._update_state(state)
            self._last_msg = "movel done"
        elif name == "home":
            self._last_msg = "homing…"
            self._last_fault = ""
            state = arm.home()
            self._update_state(state)
            self._last_msg = "home done"

    def _dispatch(self, cmd: dict) -> None:
        """Cancel any running motion and queue a new one."""
        try:
            self._arm.stop()
        except Exception:
            pass
        with self._lock:
            self._cmd = cmd
        self._cmd_event.set()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _update_state(self, state) -> None:
        if state is None:
            return
        self._last_joints = list(state.joints)
        self._last_torques = list(state.torques) if state.torques else [0.0] * len(state.joints)
        try:
            pos, rot = self._arm.get_pose()
            self._last_pos = pos.tolist()
            self._last_quat = _rot_to_quat(rot)
        except Exception:
            pass

    def _refresh_state(self) -> None:
        """Read current state from hardware (called by SSE loop)."""
        try:
            state = self._arm.get_state()
            self._last_joints = list(state.joints)
            self._last_torques = list(state.torques) if state.torques else [0.0] * len(state.joints)
        except Exception:
            pass
        try:
            pos, rot = self._arm.get_pose()
            self._last_pos = pos.tolist()
            self._last_quat = _rot_to_quat(rot)
        except Exception:
            pass

    def get_state_dict(self) -> dict:
        arm = self._arm
        limits = arm.get_joint_limits()
        return {
            "side": self.side,
            "model": arm.model.name,
            "num_joints": arm.num_joints,
            "enabled": arm.enabled,
            "moving": arm.is_moving,
            "freedrive": arm.is_freedrive,
            "joints": self._last_joints,
            "torques": self._last_torques,
            "pose": {
                "pos": self._last_pos,
                "quat": self._last_quat,
            },
            "limits": [[float(lo), float(hi)] for lo, hi in limits],
            "servo_active": self._servo_active,
            "msg": self._last_msg,
            "fault": self._last_fault,
        }

    # ------------------------------------------------------------------
    # Public API (called from HTTP handlers)
    # ------------------------------------------------------------------
    def enable(self) -> dict:
        try:
            state = self._arm.enable()
            self._update_state(state)
            self._last_msg = "enabled"
            self._last_fault = ""
            return {"ok": True, "msg": "enabled"}
        except Exception as exc:
            self._last_fault = str(exc)
            return {"ok": False, "msg": str(exc)}

    def disable(self) -> dict:
        try:
            self.servo_stop()
            self._arm.disable()
            self._last_msg = "disabled"
            return {"ok": True, "msg": "disabled"}
        except Exception as exc:
            self._last_fault = str(exc)
            return {"ok": False, "msg": str(exc)}

    def stop(self) -> dict:
        try:
            self.servo_stop()
            state = self._arm.stop()
            self._update_state(state)
            self._last_msg = "stopped"
            return {"ok": True, "msg": "stopped"}
        except Exception as exc:
            self._last_fault = str(exc)
            return {"ok": False, "msg": str(exc)}

    def home(self) -> dict:
        self._dispatch({"name": "home"})
        return {"ok": True, "msg": "homing queued"}

    def set_stiffness(self, stiffness: str) -> dict:
        valid = ("stiff", "medium", "soft")
        if stiffness not in valid:
            return {"ok": False, "msg": f"stiffness must be one of {valid}"}
        try:
            self._arm.set_stiffness(stiffness)
            self._last_msg = f"stiffness={stiffness}"
            return {"ok": True, "msg": f"stiffness set to {stiffness}"}
        except Exception as exc:
            return {"ok": False, "msg": str(exc)}

    def movej(self, joints: list, speed: Optional[float] = None,
              duration: Optional[float] = None) -> dict:
        n = self._arm.num_joints
        if len(joints) != n:
            return {"ok": False, "msg": f"expected {n} joints, got {len(joints)}"}
        self._dispatch({"name": "movej", "joints": joints,
                        "speed": speed, "duration": duration})
        return {"ok": True, "msg": "movej queued"}

    def movel(self, position: list, rotation=None,
              speed: Optional[float] = None,
              duration: Optional[float] = None) -> dict:
        if len(position) not in (3, 6):
            return {"ok": False, "msg": "position must be [x,y,z] or [x,y,z,rx,ry,rz]"}
        self._dispatch({"name": "movel", "position": position,
                        "rotation": rotation, "speed": speed, "duration": duration})
        return {"ok": True, "msg": "movel queued"}

    # ------------------------------------------------------------------
    # Servo loop (servoj 100 Hz)
    # ------------------------------------------------------------------
    def servo_start(self, kp: Optional[float] = None,
                    kd: Optional[float] = None) -> dict:
        if self._servo_active:
            return {"ok": True, "msg": "servo already active"}
        if not self._arm.enabled:
            return {"ok": False, "msg": "arm not enabled"}
        # Stop any blocking motion first
        try:
            self._arm.stop()
        except Exception:
            pass
        with self._servo_lock:
            try:
                state = self._arm.get_state()
                self._servo_target = list(state.joints)
            except Exception:
                self._servo_target = [0.0] * self._arm.num_joints
            self._servo_vel = [0.0] * self._arm.num_joints
            self._servo_kp = kp
            self._servo_kd = kd
        self._servo_stop.clear()
        self._servo_active = True
        self._servo_thread = threading.Thread(
            target=self._servo_loop, daemon=True,
            name=f"debugui-servo-{self.side}")
        self._servo_thread.start()
        return {"ok": True, "msg": "servo started"}

    def servo_target(self, joints: list,
                     velocities: Optional[list] = None) -> dict:
        if not self._servo_active:
            return {"ok": False, "msg": "servo not active"}
        n = self._arm.num_joints
        if len(joints) != n:
            return {"ok": False, "msg": f"expected {n} joints"}
        with self._servo_lock:
            self._servo_target = list(joints)
            if velocities and len(velocities) == n:
                self._servo_vel = list(velocities)
            else:
                self._servo_vel = [0.0] * n
        return {"ok": True, "msg": "target updated"}

    def servo_stop(self) -> dict:
        was = self._servo_active
        self._servo_stop.set()
        if self._servo_thread and self._servo_thread.is_alive():
            self._servo_thread.join(timeout=0.5)
        self._servo_active = False
        return {"ok": True, "msg": "servo stopped" if was else "servo was not active"}

    def _servo_loop(self) -> None:
        period = 1.0 / 100.0   # 100 Hz
        next_t = time.monotonic()
        arm = self._arm
        while not self._servo_stop.is_set() and arm.enabled:
            with self._servo_lock:
                q = list(self._servo_target)
                vel = list(self._servo_vel)
                kp = self._servo_kp
                kd = self._servo_kd
            try:
                state = arm.servoj(q, kp=kp, kd=kd, velocities=vel)
                self._last_joints = list(state.joints)
                self._last_torques = list(state.torques) if state.torques else [0.0] * len(state.joints)
            except Exception as exc:
                logger.debug("[debugui servo %s] %s", self.side, exc)
                break
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._servo_stop.wait(delay)
            else:
                next_t = time.monotonic()
        self._servo_active = False

    # ------------------------------------------------------------------
    # Freedrive
    # ------------------------------------------------------------------
    def freedrive_start(self, damping: float = 0.5) -> dict:
        if not self._arm.enabled:
            return {"ok": False, "msg": "arm not enabled"}
        self.servo_stop()
        try:
            self._arm.freedrive_enable(damping=damping)
            self._last_msg = f"freedrive (damping={damping})"
            return {"ok": True, "msg": "freedrive enabled"}
        except Exception as exc:
            self._last_fault = str(exc)
            return {"ok": False, "msg": str(exc)}

    def freedrive_stop(self) -> dict:
        try:
            self._arm.freedrive_disable()
            self._last_msg = "freedrive off"
            return {"ok": True, "msg": "freedrive disabled"}
        except Exception as exc:
            return {"ok": False, "msg": str(exc)}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._worker_alive = False
        self._cmd_event.set()
        self.servo_stop()
        try:
            self._arm.freedrive_disable()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# _NullAdapter — placeholder for unused side in single-arm mode
# ---------------------------------------------------------------------------
class _NullAdapter:
    side = "null"

    def get_state_dict(self) -> dict:
        return {
            "side": self.side, "model": "none", "num_joints": 0,
            "enabled": False, "moving": False, "freedrive": False,
            "joints": [], "torques": [],
            "pose": {"pos": [0, 0, 0], "quat": [0, 0, 0, 1]},
            "limits": [], "servo_active": False,
            "msg": "not configured", "fault": "",
        }

    def _refresh_state(self) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class _LabHandler(BaseHTTPRequestHandler):

    # Injected by LabServer before serving
    adapters: Dict[str, Any] = {}   # {"left": ArmAdapter, "right": ArmAdapter}
    dual_mode: bool = False

    def log_message(self, fmt, *args):  # suppress default access log
        logger.debug("[debugui] " + fmt, *args)

    # ── helpers ──────────────────────────────────────────────────────

    def _send(self, status: int = 200, ctype: str = "application/json",
              body: bytes = b"") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, status: int = 200) -> None:
        self._send(status, "application/json",
                   json.dumps(data, ensure_ascii=False).encode())

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n > 0 else b""
        try:
            return json.loads(raw) if raw else {}
        except Exception:
            return {}

    def _resolve_side(self, path: str):
        """Return (adapter, rest_path) or (None, None) on no match."""
        if self.dual_mode:
            for side in ("left", "right"):
                prefix = f"/{side}/api"
                if path.startswith(prefix):
                    return self.adapters[side], path[len(f"/{side}"):]
            return None, None
        else:
            if path.startswith("/api"):
                # single-arm: serve whichever adapter is configured
                side = "right" if "right" in self.adapters else "left"
                return self.adapters[side], path
            return None, None

    # ── static / asset files ─────────────────────────────────────────

    def _serve_static(self, rel: str) -> None:
        safe = (_STATIC_DIR / rel).resolve()
        if not str(safe).startswith(str(_STATIC_DIR.resolve())):
            self._send(403, "text/plain", b"forbidden")
            return
        if not safe.is_file():
            self._send(404, "text/plain", b"not found")
            return
        ctype = mimetypes.guess_type(str(safe))[0] or "application/octet-stream"
        self._send(200, ctype, safe.read_bytes())

    def _serve_urdf(self, rel: str) -> None:
        """Serve URDF from onero_sdk/urdf/A1/ and rewrite mesh paths to /assets/meshes/."""
        safe = (_URDF_DIR / rel).resolve()
        if not str(safe).startswith(str(_URDF_DIR.resolve())) or not safe.is_file():
            self._send(404, "text/plain", b"urdf not found")
            return
        # Patch mesh filename paths so Three.js urdf-loader can fetch them
        # from /assets/meshes/...
        text = safe.read_text(encoding="utf-8")
        # Replace relative STL references with absolute /assets/meshes/ URLs
        # Pattern: filename="<anything>.STL" (case-insensitive)
        import re
        def _rewrite(m):
            orig = m.group(1)
            # Strip leading ./ or ../
            clean = orig.lstrip("./").lstrip("../")
            # Normalise: A1/A1_r/Link1.STL  →  /assets/meshes/A1/A1_r/Link1.STL
            return f'filename="/assets/meshes/{clean.lstrip("/")}"'
        text = re.sub(r'filename="([^"]+\.(?:STL|stl|dae|DAE))"', _rewrite, text)
        self._send(200, "application/xml", text.encode("utf-8"))

    def _serve_mesh(self, rel: str) -> None:
        safe = (_MESH_DIR / rel).resolve()
        if not str(safe).startswith(str(_MESH_DIR.resolve())) or not safe.is_file():
            self._send(404, "text/plain", b"mesh not found")
            return
        ctype = mimetypes.guess_type(str(safe))[0] or "application/octet-stream"
        self._send(200, ctype, safe.read_bytes())

    # ── SSE stream ───────────────────────────────────────────────────

    def _sse_stream(self, hz: float) -> None:
        """Push combined arm state as SSE at requested frequency."""
        hz = max(1.0, min(50.0, hz))
        period = 1.0 / hz
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                t0 = time.monotonic()
                for adapter in self.adapters.values():
                    adapter._refresh_state()
                payload = {
                    side: self.adapters[side].get_state_dict()
                    for side in self.adapters
                }
                line = "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                self.wfile.write(line.encode())
                self.wfile.flush()
                elapsed = time.monotonic() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ── GET ──────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "":
            self._serve_static("index.html")
            return

        if path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
            return

        if path.startswith("/assets/urdf/"):
            self._serve_urdf(path[len("/assets/urdf/"):])
            return

        if path.startswith("/assets/meshes/"):
            self._serve_mesh(path[len("/assets/meshes/"):])
            return

        if path == "/api/stream":
            qs = parse_qs(parsed.query)
            hz = float(qs.get("hz", ["30"])[0])
            self._sse_stream(hz)
            return

        # Per-arm state
        adapter, rest = self._resolve_side(path)
        if adapter and rest == "/api/state":
            adapter._refresh_state()
            self._json(adapter.get_state_dict())
            return

        self._send(404, "text/plain", b"not found")

    # ── OPTIONS (CORS preflight) ──────────────────────────────────────

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── POST ─────────────────────────────────────────────────────────

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_body()

        adapter, rest = self._resolve_side(path)
        if adapter is None:
            self._send(404, "text/plain", b"not found")
            return

        routes = {
            "/api/enable":           lambda: adapter.enable(),
            "/api/disable":          lambda: adapter.disable(),
            "/api/stop":             lambda: adapter.stop(),
            "/api/home":             lambda: adapter.home(),
            "/api/set_stiffness":    lambda: adapter.set_stiffness(
                                         body.get("stiffness", "medium")),
            "/api/movej":            lambda: adapter.movej(
                                         body.get("joints", []),
                                         speed=body.get("speed"),
                                         duration=body.get("duration")),
            "/api/movel":            lambda: adapter.movel(
                                         body.get("position", []),
                                         rotation=body.get("rotation"),
                                         speed=body.get("speed"),
                                         duration=body.get("duration")),
            "/api/servo/start":      lambda: adapter.servo_start(
                                         kp=body.get("kp"),
                                         kd=body.get("kd")),
            "/api/servo/target":     lambda: adapter.servo_target(
                                         body.get("joints", []),
                                         velocities=body.get("velocities")),
            "/api/servo/stop":       lambda: adapter.servo_stop(),
            "/api/freedrive/start":  lambda: adapter.freedrive_start(
                                         damping=float(body.get("damping", 0.5))),
            "/api/freedrive/stop":   lambda: adapter.freedrive_stop(),
        }

        handler = routes.get(rest)
        if handler is None:
            self._send(404, "text/plain", b"unknown endpoint")
            return
        try:
            result = handler()
            self._json(result)
        except Exception as exc:
            logger.exception("[debugui] handler error: %s", exc)
            self._json({"ok": False, "msg": str(exc)}, status=500)


# ---------------------------------------------------------------------------
# LabServer
# ---------------------------------------------------------------------------
class LabServer(ThreadingHTTPServer):
    """
    The debug UI HTTP server.

    Create via ``start_lab()`` or ``start_lab_dual()``, or construct directly::

        server = LabServer(adapters={"right": ArmAdapter(arm, "right")},
                           dual_mode=False, host="0.0.0.0", port=8090)
        server.serve_forever()
    """

    def __init__(self, adapters: Dict[str, ArmAdapter],
                 dual_mode: bool = False,
                 host: str = "0.0.0.0",
                 port: int = 8090) -> None:
        self._adapters = adapters
        self._dual_mode = dual_mode

        # Patch the handler class (one instance per server)
        class Handler(_LabHandler):
            pass
        Handler.adapters = adapters
        Handler.dual_mode = dual_mode

        super().__init__((host, port), Handler)

    def close(self) -> None:
        for adapter in self._adapters.values():
            adapter.close()
        self.server_close()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def start_lab(arm, side: str = "right",
              host: str = "0.0.0.0", port: int = 8090,
              block: bool = True) -> Optional[LabServer]:
    """
    Start the debug UI for a single ``Arm`` instance.

    Args:
        arm:   An ``onero_sdk.Arm`` instance (any backend — real or sim).
        side:  "right" or "left".  Affects which panel is shown in the UI.
        host:  Bind address.  Default ``"0.0.0.0"`` (all interfaces).
        port:  TCP port.  Default 8090.
        block: If ``True`` (default) block until Ctrl-C.
               If ``False`` start in a background thread and return the server.

    Returns:
        ``LabServer`` when ``block=False``, else ``None``.
    """
    adapter = ArmAdapter(arm, side=side)
    server = LabServer(adapters={side: adapter}, dual_mode=False,
                       host=host, port=port)
    url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}"
    print(f"[debugui] {url}  side={side}  model={arm.model.name}", flush=True)
    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.close()
        return None
    else:
        t = threading.Thread(target=server.serve_forever, daemon=True,
                             name="debugui-server")
        t.start()
        return server


def start_lab_dual(dual, host: str = "0.0.0.0", port: int = 8090,
                   block: bool = True) -> Optional[LabServer]:
    """
    Start the debug UI for a ``DualArm`` instance.

    Args:
        dual:  A ``onero_sdk.DualArm`` instance.
        host:  Bind address.
        port:  TCP port.
        block: Block until Ctrl-C when ``True``.

    Returns:
        ``LabServer`` when ``block=False``, else ``None``.
    """
    adapters = {
        "left":  ArmAdapter(dual.left,  side="left"),
        "right": ArmAdapter(dual.right, side="right"),
    }
    server = LabServer(adapters=adapters, dual_mode=True, host=host, port=port)
    url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}"
    print(f"[debugui] {url}  mode=dual", flush=True)
    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.close()
        return None
    else:
        t = threading.Thread(target=server.serve_forever, daemon=True,
                             name="debugui-server")
        t.start()
        return server
