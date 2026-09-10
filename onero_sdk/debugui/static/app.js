/**
 * onero SDK Debug UI — app.js
 *
 * Sections:
 *   1.  Config & shared state
 *   2.  Three.js scene setup
 *   3.  URDF robot loading  (real mesh + ghost preview)
 *   4.  MoveL target-ball drag
 *   5.  Panel rendering  (MoveJ / Servo / MoveL / Freedrive tabs)
 *   6.  SSE state stream  (+ polling fallback)
 *   7.  API helpers
 *   8.  Bootstrap
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import URDFLoader from '/static/vendor/urdf-loader/URDFLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

// ─────────────────────────────────────────────────────────────
// 1.  Config & shared state
// ─────────────────────────────────────────────────────────────

const SIDES = ['left', 'right'];

// These defaults are overwritten once /api/state is fetched
const CFG = {
  left:  { api: '/left/api',  model: 'a1_l', shoulderY: +0.18, panel: 'leftPanel',  label: '左臂' },
  right: { api: '/right/api', model: 'a1_r', shoulderY: -0.18, panel: 'rightPanel', label: '右臂' },
};

// Whether the server runs in single-arm mode (served at /api instead of /left|right/api)
let singleArmSide = null;   // 'left' | 'right' | null (dual)

// Live arm state — updated by SSE
const armState = {
  left:  { joints: [], limits: null, enabled: false, moving: false, freedrive: false,
           servo_active: false, end_effector: null, model: 'a1_l', num_joints: 7 },
  right: { joints: [], limits: null, enabled: false, moving: false, freedrive: false,
           servo_active: false, end_effector: null, model: 'a1_r', num_joints: 7 },
};

// Preview joint values (driven by UI sliders, not hardware)
const preview = { left: [], right: [] };
const previewLocked = { left: false, right: false };
const dragging = { left: false, right: false };

// Auto-send state (unused timer slot kept for compat)
const autoSendState = {
  left:  {},
  right: {},
};

// Servo 100Hz loop state (client side)
const servoLoop = { left: null, right: null };   // setInterval handle

function fmt(v) { return Number(v).toFixed(4); }
function setHud(msg) { document.getElementById('hud').textContent = msg; }
function setStatus(side, msg) {
  const el = document.getElementById(`${side}_status`);
  if (el) el.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
}

function apiBase(side) {
  if (singleArmSide) return '/api';
  return CFG[side].api;
}

// ─────────────────────────────────────────────────────────────
// 2.  Three.js scene
// ─────────────────────────────────────────────────────────────

const canvas   = document.getElementById('canvas3d');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x10151f);

const camera = new THREE.PerspectiveCamera(50, 2, 0.03, 6);
camera.up.set(0, 0, 1);
camera.position.set(0.9, -1.1, 0.65);
camera.lookAt(0, 0, 0);

const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.target.set(0, 0, -0.05);
orbitControls.update();

// Lighting
scene.add(new THREE.AmbientLight(0x606070));
const dirLight = new THREE.DirectionalLight(0xffffff, 2);
dirLight.position.set(1, -1, 1.5);
scene.add(dirLight);

// Grid & axes
const grid = new THREE.GridHelper(1.2, 24, 0x334466, 0x1a2a3a);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
[
  [new THREE.Vector3(1,0,0), 0xff5555],
  [new THREE.Vector3(0,1,0), 0x55ff55],
  [new THREE.Vector3(0,0,1), 0x5588ff],
].forEach(([dir, color]) =>
  scene.add(new THREE.ArrowHelper(dir, new THREE.Vector3(0,0,0), 0.35, color, 0.05, 0.03)));

// Torso reference
const torso = new THREE.Mesh(
  new THREE.CylinderGeometry(0.035, 0.045, 0.42, 16),
  new THREE.MeshPhongMaterial({ color: 0x64748b }),
);
torso.rotation.x = Math.PI / 2;
torso.position.set(0, 0, -0.16);
scene.add(torso);

for (const s of SIDES) {
  const sh = new THREE.Mesh(
    new THREE.SphereGeometry(0.025, 16, 16),
    new THREE.MeshPhongMaterial({ color: 0xfbbf24 }),
  );
  sh.position.set(0, CFG[s].shoulderY, 0);
  scene.add(sh);
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();
(function animate() { requestAnimationFrame(animate); renderer.render(scene, camera); })();

// ─────────────────────────────────────────────────────────────
// 3.  URDF robot loading
// ─────────────────────────────────────────────────────────────

/**
 * Display transform: operator frame ← URDF frame
 *   op X = URDF Z,  op Y = -URDF Y,  op Z = URDF X
 */
const ROOT_ROT = new THREE.Matrix4().set(
  0,  0, 1, 0,
  0, -1, 0, 0,
  1,  0, 0, 0,
  0,  0, 0, 1,
);

const robots     = { left: null, right: null };
const ghostGroup = { left: null, right: null };
const ghostAxes  = { left: [], right: [] };
const showGhost  = { left: true, right: true };

function jointName(side, i) { return `joint${i + 1}-${CFG[side].model}`; }

function makeRGBAxes(size = 0.06) {
  const g = new THREE.Group();
  [[new THREE.Vector3(size,0,0), 0xff2222],
   [new THREE.Vector3(0,size,0), 0x22ff22],
   [new THREE.Vector3(0,0,size), 0x2255ff]].forEach(([end, color]) => {
    const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), end]);
    g.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
      color, depthTest: false, depthWrite: false, linewidth: 2 })));
  });
  return g;
}

function applyRealMaterial(robot) {
  robot.traverse(o => {
    if (!o.material) return;
    const fn = m => { m.side = THREE.DoubleSide; };
    Array.isArray(o.material) ? o.material.forEach(fn) : fn(o.material);
  });
}

function buildGhostAxes(side, robot) {
  const axes = [];
  for (let i = 0; i < 7; i++) {
    const j = robot.joints[jointName(side, i)];
    if (!j) continue;
    const ax = makeRGBAxes(0.06);
    ghostGroup[side].add(ax);
    axes.push({ ax, joint: j });
  }
  ghostAxes[side] = axes;
  updateGhost(side);
}

function loadRobot(side) {
  const c = CFG[side];
  const group = new THREE.Group();
  group.position.set(0, c.shoulderY, 0);
  group.rotation.z = side === 'left' ? Math.PI / 2 : -Math.PI / 2;
  scene.add(group);

  const gg = new THREE.Group();
  gg.position.copy(group.position);
  gg.rotation.copy(group.rotation);
  gg.visible = showGhost[side];
  scene.add(gg);
  ghostGroup[side] = gg;

  const loader = new URDFLoader();
  // urdf-loader prepends workingPath to every filename;
  // our filenames are already absolute (/assets/meshes/...) so
  // we override loadMeshCb to fetch them directly and skip resolvePath.
  loader.loadMeshCb = (path, manager, done) => {
    // strip any accidental workingPath prefix that snuck in
    const url = path.replace(/^.*(\/assets\/meshes\/)/, '/assets/meshes/');
    new STLLoader(manager).load(
      url,
      geom => done(new THREE.Mesh(geom, new THREE.MeshPhongMaterial({ color: 0xaabbcc, side: THREE.DoubleSide }))),
      null,
      err => done(null, err)
    );
  };
  loader.load(`/assets/urdf/${c.model}.urdf`, robot => {
    robot.setRotationFromMatrix(ROOT_ROT);
    applyRealMaterial(robot);
    group.add(robot);
    robots[side] = robot;
    updateRobot(side);
    buildGhostAxes(side, robot);
    setHud(`URDF loaded: ${c.model}`);
  }, undefined, err => setHud(`URDF load failed (${side}): ${err}`));
}

function updateRobot(side) {
  const r = robots[side];
  if (!r) return;
  armState[side].joints.forEach((q, i) => {
    const j = r.joints[jointName(side, i)];
    if (j) j.setJointValue(q);
  });
}

function updateGhost(side) {
  if (!showGhost[side]) return;
  const axes = ghostAxes[side];
  if (!axes || !axes.length) return;
  const robot = robots[side];
  if (!robot) return;

  // Temporarily apply preview, snap joint axes, restore real pose
  const saved = preview[side].map((q, i) => {
    const j = robot.joints[jointName(side, i)];
    const cur = j ? j.angle : 0;
    if (j) j.setJointValue(q);
    return cur;
  });
  robot.updateMatrixWorld(true);

  axes.forEach(({ ax, joint }) => {
    const wp = new THREE.Vector3();
    const wq = new THREE.Quaternion();
    joint.getWorldPosition(wp);
    joint.getWorldQuaternion(wq);
    ghostGroup[side].updateMatrixWorld(true);
    const inv = new THREE.Matrix4().copy(ghostGroup[side].matrixWorld).invert();
    wp.applyMatrix4(inv);
    ax.position.copy(wp);
    const gq = new THREE.Quaternion();
    ghostGroup[side].getWorldQuaternion(gq);
    ax.quaternion.copy(gq.invert().multiply(wq));
  });

  saved.forEach((q, i) => {
    const j = robot.joints[jointName(side, i)];
    if (j) j.setJointValue(q);
  });
  robot.updateMatrixWorld(true);
}

function setGhostVisible(side, v) {
  showGhost[side] = v;
  if (ghostGroup[side]) ghostGroup[side].visible = v;
}

// ─────────────────────────────────────────────────────────────
// 4.  MoveL target-ball drag
// ─────────────────────────────────────────────────────────────

/**
 * Coordinate transforms (URDF base frame ↔ Three.js scene):
 *   scene.x = urdf.z,   scene.y = -urdf.y + shoulderY,   scene.z = urdf.x
 * Inverse (delta):
 *   urdf.dx = scene.dz,  urdf.dy = -scene.dy,  urdf.dz = scene.dx
 */
function eeToScene(pos, side) {
  if (!pos || pos.length < 3) return null;
  return new THREE.Vector3(pos[2], -pos[1] + CFG[side].shoulderY, pos[0]);
}

function sceneDeltaToArm(from, to) {
  const d = to.clone().sub(from);
  return [d.z, -d.y, d.x];     // [dx, dy, dz] in arm base frame
}

const BALL_COLORS       = { left: 0x3b82f6, right: 0xf97316 };
const CARTESIAN_MAX_STEP = 0.025;

const ballMeshes       = {};
const eeDotMeshes      = {};
const cartesianBallMode = { left: null, right: null };
const cartesianTarget   = { left: new THREE.Vector3(), right: new THREE.Vector3() };

for (const s of SIDES) {
  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(0.025, 20, 20),
    new THREE.MeshPhongMaterial({ color: BALL_COLORS[s], transparent: true, opacity: 0.85 }),
  );
  ball.userData.side = s;
  ball.visible = false;
  scene.add(ball);
  ballMeshes[s] = ball;

  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(0.012, 12, 12),
    new THREE.MeshPhongMaterial({ color: BALL_COLORS[s], transparent: true, opacity: 0.45 }),
  );
  dot.visible = false;
  scene.add(dot);
  eeDotMeshes[s] = dot;
}

function setCartesianInputs(side, mode, pos) {
  if (!pos || pos.length < 3) return;
  ['x', 'y', 'z'].forEach((axis, i) => {
    const input = document.getElementById(`${side}_${mode}_${axis}`);
    if (input) input.value = Number(pos[i]).toFixed(4);
  });
}

function currentCartesianTarget(side, mode) {
  const values = ['x', 'y', 'z'].map(axis => {
    const input = document.getElementById(`${side}_${mode}_${axis}`);
    return input ? Number(input.value) : NaN;
  });
  return values.every(Number.isFinite) ? values : null;
}

function resetCartesianTarget(side, mode) {
  const ee = armState[side].end_effector;
  if (ee && ee.pos) setCartesianInputs(side, mode, ee.pos);
}

function activateCartesianBall(side, mode, active) {
  cartesianBallMode[side] = active ? mode : null;
  ballMeshes[side].visible  = active;
  eeDotMeshes[side].visible = active;
  if (active) {
    const ee = armState[side].end_effector;
    const pos = eeToScene(ee ? ee.pos : null, side);
    if (pos) {
      ballMeshes[side].position.copy(pos);
      cartesianTarget[side].copy(pos);
    }
    if (ee && ee.pos) setCartesianInputs(side, mode, ee.pos);
  }
}

function updateEeDot(side) {
  const ee = armState[side].end_effector;
  const pos = eeToScene(ee ? ee.pos : null, side);
  if (pos) eeDotMeshes[side].position.copy(pos);
}

// Drag state
const raycaster = new THREE.Raycaster();
const mouse2d   = new THREE.Vector2();
let   dragState = null;

function getNDC(ev) {
  const r = canvas.getBoundingClientRect();
  mouse2d.x =  ((ev.clientX - r.left) / r.width)  * 2 - 1;
  mouse2d.y = -((ev.clientY - r.top)  / r.height) * 2 + 1;
}

canvas.addEventListener('pointerdown', ev => {
  if (ev.button !== 0) return;
  getNDC(ev);
  raycaster.setFromCamera(mouse2d, camera);
  const activeBalls = SIDES.filter(s => cartesianBallMode[s]).map(s => ballMeshes[s]);
  if (!activeBalls.length) return;
  const hits = raycaster.intersectObjects(activeBalls);
  if (!hits.length) return;
  const side = hits[0].object.userData.side;
  orbitControls.enabled = false;
  const plane = new THREE.Plane(new THREE.Vector3(0,0,1), -hits[0].object.position.z);
  dragState = {
    side,
    mode: cartesianBallMode[side],
    plane,
    fixedZ: hits[0].object.position.z,
  };
  canvas.setPointerCapture(ev.pointerId);
  ev.stopPropagation();
});

canvas.addEventListener('pointermove', ev => {
  if (!dragState) return;
  getNDC(ev);
  raycaster.setFromCamera(mouse2d, camera);
  const pt = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(dragState.plane, pt)) return;
  pt.z = dragState.fixedZ;
  ballMeshes[dragState.side].position.copy(pt);
  cartesianTarget[dragState.side].copy(pt);

  // Dragging is preview-only. MoveL and MoveP are blocking trajectory
  // commands, not streaming servo commands. Update the selected mode's
  // numeric preview and submit exactly once on pointerup.
  const ee = armState[dragState.side].end_effector;
  const cur = eeToScene(ee ? ee.pos : null, dragState.side);
  if (cur && ee) {
    const delta = sceneDeltaToArm(cur, cartesianTarget[dragState.side]);
    setCartesianInputs(
      dragState.side,
      dragState.mode,
      ee.pos.map((v, i) => v + delta[i]),
    );
  }
});

canvas.addEventListener('pointerup', ev => {
  if (!dragState) return;
  const { side, mode } = dragState;
  orbitControls.enabled = true;
  dragState = null;
  canvas.releasePointerCapture(ev.pointerId);
  sendCartesian(side, mode);
});

function sendCartesian(side, mode) {
  const ee = armState[side].end_effector;
  const target = currentCartesianTarget(side, mode);
  if (!ee || !ee.pos || !target) {
    setStatus(side, `${mode}: invalid target`);
    return;
  }

  const delta = target.map((v, i) => v - ee.pos[i]);
  if (Math.max(...delta.map(Math.abs)) > CARTESIAN_MAX_STEP + 1e-9) {
    setStatus(side, `${mode}: each axis is limited to 2.5 cm per command`);
    return;
  }
  if (delta.reduce((sum, v) => sum + Math.abs(v), 0) < 0.0005) return;

  const speedInput = document.getElementById(
    mode === 'movel' ? `${side}_lspeed` : `${side}_pspeed`,
  );
  const speed = speedInput ? Number(speedInput.value) : undefined;
  api(side, mode, { position: target, speed })
    .then(d => setStatus(side, `${mode}: ${d.ok ? 'ok' : 'fail'} ${d.msg || ''}`));
}

// ─────────────────────────────────────────────────────────────
// 5.  Panel rendering
// ─────────────────────────────────────────────────────────────

function renderPanel(side) {
  const c      = CFG[side];
  const s      = armState[side];
  const limits = s.limits || Array(7).fill([-3.14, 3.14]);
  const panel  = document.getElementById(c.panel);

  panel.innerHTML = `
<div class="panelHeader">
  <span class="connDot" id="${side}_dot"></span>
  <span class="armTitle">${c.label} <small id="${side}_modlabel"></small></span>
  <span id="${side}_connlabel" style="font-size:11px;color:#6b7a91">--</span>
</div>

<div class="stiffRow">
  <span style="font-size:11px;color:#6b7a91;margin-right:2px">刚度：</span>
  ${['stiff','medium','soft'].map(k =>
    `<button class="stiffBtn" data-stiff="${k}" id="${side}_stiff_${k}">${k}</button>`
  ).join('')}
</div>

<div class="controlRow">
  <button class="btn ok"   id="${side}_en">Enable</button>
  <button class="btn warn" id="${side}_dis">Disable</button>
  <button class="btn warn" id="${side}_stop">Stop</button>
  <button class="btn ghost" id="${side}_home">Home</button>
</div>

<div class="modeTabs">
  <button class="modeTab active" data-mode="movej">MoveJ</button>
  <button class="modeTab" data-mode="servo">Servo</button>
  <button class="modeTab" data-mode="movel">MoveL</button>
  <button class="modeTab" data-mode="movep">MoveP</button>
  <button class="modeTab" data-mode="freedrive">Freedrive</button>
</div>

<!-- MoveJ -->
<div class="modeCard active" id="${side}_card_movej">
  <div class="cardToolbar">
    <button class="btn" id="${side}_mj_app">Apply</button>
    <label style="font-size:12px"><input id="${side}_mj_auto" type="checkbox"/> Auto</label>
    <label style="font-size:12px;margin-left:4px"><input id="${side}_mj_ghost" type="checkbox" checked/> Ghost</label>
    <span style="margin-left:auto;font-size:11px;color:#6b7a91">speed(rad/s)</span>
    <input id="${side}_mj_speed" class="speed" type="number" min="0.05" max="5" step="0.05" value="0.5"/>
  </div>
  <div class="joints" id="${side}_mj_joints"></div>
</div>

<!-- Servo -->
<div class="modeCard" id="${side}_card_servo">
  <div class="cardToolbar">
    <span class="servoActive" id="${side}_sv_led"></span>
    <button class="btn ok"   id="${side}_sv_start">Start</button>
    <button class="btn warn" id="${side}_sv_stop">Stop</button>
    <label style="font-size:12px;margin-left:4px"><input id="${side}_sv_ghost" type="checkbox" checked/> Ghost</label>
  </div>
  <div class="gainRow">
    <label>kp</label>
    <input id="${side}_sv_kp" type="number" min="0" max="500" step="5" value="50"/>
    <label style="margin-left:8px">kd</label>
    <input id="${side}_sv_kd" type="number" min="0" max="20"  step="0.1" value="1.0"/>
    <span style="margin-left:auto;font-size:11px;color:#6b7a91">100 Hz servoj</span>
  </div>
  <div class="cardInfo">⚠ 实时控制 — 滑条拖动即时跟踪。启动后勿切换其他运动模式。</div>
  <div class="joints" id="${side}_sv_joints"></div>
</div>

<!-- MoveL -->
<div class="modeCard" id="${side}_card_movel">
  <div class="cardToolbar">
    <button class="btn ok" id="${side}_lv_act">Drag Ball</button>
    <button class="btn" id="${side}_lv_current">Current</button>
    <button class="btn ok" id="${side}_lv_apply">Apply</button>
    <button class="btn warn" id="${side}_lv_deact">Stop</button>
  </div>
  <div class="gainRow">
    <label>X</label><input id="${side}_movel_x" type="number" step="0.001"/>
    <label>Y</label><input id="${side}_movel_y" type="number" step="0.001"/>
    <label>Z</label><input id="${side}_movel_z" type="number" step="0.001"/>
  </div>
  <div class="gainRow">
    <label>speed (m/s)</label>
    <input id="${side}_lspeed" type="number" min="0.02" max="0.5" step="0.01" value="0.1"/>
  </div>
  <div class="cardInfo">笛卡尔直线路径。拖球只做预览，松手后发送一次；每轴单次最大 2.5 cm。未填写姿态时保持当前姿态。</div>
  <div class="poseRow" id="${side}_pose_row">EE: --</div>
</div>

<!-- MoveP -->
<div class="modeCard" id="${side}_card_movep">
  <div class="cardToolbar">
    <button class="btn ok" id="${side}_pv_act">Drag Ball</button>
    <button class="btn" id="${side}_pv_current">Current</button>
    <button class="btn ok" id="${side}_pv_apply">Apply</button>
    <button class="btn warn" id="${side}_pv_stop">Stop</button>
  </div>
  <div class="gainRow">
    <label>X</label><input id="${side}_movep_x" type="number" step="0.001"/>
    <label>Y</label><input id="${side}_movep_y" type="number" step="0.001"/>
    <label>Z</label><input id="${side}_movep_z" type="number" step="0.001"/>
  </div>
  <div class="gainRow">
    <label>speed (rad/s)</label>
    <input id="${side}_pspeed" type="number" min="0.05" max="5" step="0.05" value="0.5"/>
  </div>
  <div class="cardInfo">点到点笛卡尔运动：先解目标 IK，再执行关节插值；路径不保证直线。每轴单次最大 2.5 cm。</div>
  <div class="poseRow" id="${side}_movep_pose">EE: --</div>
</div>

<!-- Freedrive -->
<div class="modeCard" id="${side}_card_freedrive">
  <div class="cardToolbar">
    <button class="btn ok"   id="${side}_fd_start">Enable</button>
    <button class="btn warn" id="${side}_fd_stop">Disable</button>
    <span style="margin-left:auto;font-size:11px;color:#6b7a91">damping</span>
    <input id="${side}_fd_damp" class="speed" type="number" min="0" max="5" step="0.1" value="0.5"/>
  </div>
  <div class="cardInfo">零重力示教模式 — 手动引导机械臂，释放后自动保持位置。<br>
    需要安装 <code>onero-sdk[dynamics]</code>（pinocchio）。</div>
  <div class="poseRow" id="${side}_fd_pose">EE: --</div>
</div>

<div class="status" id="${side}_status"></div>`;

  // ── MoveJ sliders ──────────────────────────────────
  _buildSliders(`${side}_mj_joints`, side, limits, s.joints, (i, v) => {
    // input: update preview & ghost only, do NOT send during drag
    preview[side][i] = v;
    previewLocked[side] = true;
    updateGhost(side);
  }, () => {
    // change (pointerup / release): send once if Auto is checked
    const auto = panel.querySelector(`#${side}_mj_auto`);
    if (auto && auto.checked) sendMoveJ(side);
  });

  // ── Servo sliders ──────────────────────────────────
  _buildSliders(`${side}_sv_joints`, side, limits, s.joints, (i, v) => {
    preview[side][i] = v;
    updateGhost(side);
    if (s.servo_active) {
      api(side, 'servo/target', { joints: preview[side].slice() });
    }
  }, null);

  // ── Tab switching ──────────────────────────────────
  panel.querySelectorAll('.modeTab').forEach(btn => {
    btn.onclick = () => {
      panel.querySelectorAll('.modeTab').forEach(b => b.classList.remove('active'));
      panel.querySelectorAll('.modeCard').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.dataset.mode;
      panel.querySelector(`#${side}_card_${mode}`).classList.add('active');
      // Deactivate the Cartesian target ball when leaving its active mode and
      // initialize targets from the latest measured pose, not stale UI values.
      if (cartesianBallMode[side] && cartesianBallMode[side] !== mode) {
        activateCartesianBall(side, cartesianBallMode[side], false);
      }
      if (mode === 'movel' || mode === 'movep') resetCartesianTarget(side, mode);
    };
  });

  // ── Stiffness ──────────────────────────────────────
  panel.querySelectorAll('.stiffBtn').forEach(btn => {
    btn.onclick = () => {
      panel.querySelectorAll('.stiffBtn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      api(side, 'set_stiffness', { stiffness: btn.dataset.stiff })
        .then(d => setStatus(side, d.msg));
    };
  });

  // ── Control buttons ────────────────────────────────
  panel.querySelector(`#${side}_en`)  .onclick = () =>
    api(side, 'enable').then(d => setStatus(side, d.msg));
  panel.querySelector(`#${side}_dis`) .onclick = () =>
    api(side, 'disable').then(d => setStatus(side, d.msg));
  panel.querySelector(`#${side}_stop`).onclick = () =>
    api(side, 'stop').then(d => setStatus(side, d.msg));
  panel.querySelector(`#${side}_home`).onclick = () =>
    api(side, 'home').then(d => setStatus(side, d.msg));

  // ── MoveJ buttons ──────────────────────────────────
  panel.querySelector(`#${side}_mj_app`).onclick = () => sendMoveJ(side);
  panel.querySelector(`#${side}_mj_ghost`).onchange = ev =>
    setGhostVisible(side, ev.target.checked);

  // ── Servo buttons ──────────────────────────────────
  panel.querySelector(`#${side}_sv_start`).onclick = () => {
    const kp = parseFloat(panel.querySelector(`#${side}_sv_kp`).value);
    const kd = parseFloat(panel.querySelector(`#${side}_sv_kd`).value);
    api(side, 'servo/start', { kp, kd })
      .then(d => setStatus(side, d.msg));
  };
  panel.querySelector(`#${side}_sv_stop`).onclick = () =>
    api(side, 'servo/stop').then(d => setStatus(side, d.msg));
  panel.querySelector(`#${side}_sv_ghost`).onchange = ev =>
    setGhostVisible(side, ev.target.checked);

  // ── Cartesian motion buttons ──────────────────────
  panel.querySelector(`#${side}_lv_act`).onclick = () => activateCartesianBall(side, 'movel', true);
  panel.querySelector(`#${side}_lv_current`).onclick = () => resetCartesianTarget(side, 'movel');
  panel.querySelector(`#${side}_lv_apply`).onclick = () => sendCartesian(side, 'movel');
  panel.querySelector(`#${side}_lv_deact`).onclick = () => {
    activateCartesianBall(side, 'movel', false);
    api(side, 'stop').then(d => setStatus(side, d.msg));
  };
  panel.querySelector(`#${side}_pv_act`).onclick = () => activateCartesianBall(side, 'movep', true);
  panel.querySelector(`#${side}_pv_current`).onclick = () => resetCartesianTarget(side, 'movep');
  panel.querySelector(`#${side}_pv_apply`).onclick = () => sendCartesian(side, 'movep');
  panel.querySelector(`#${side}_pv_stop`).onclick = () => {
    activateCartesianBall(side, 'movep', false);
    api(side, 'stop').then(d => setStatus(side, d.msg));
  };

  // ── Freedrive buttons ─────────────────────────────
  panel.querySelector(`#${side}_fd_start`).onclick = () => {
    const damp = parseFloat(panel.querySelector(`#${side}_fd_damp`).value);
    api(side, 'freedrive/start', { damping: damp })
      .then(d => setStatus(side, d.msg));
  };
  panel.querySelector(`#${side}_fd_stop`).onclick = () =>
    api(side, 'freedrive/stop').then(d => setStatus(side, d.msg));
}

/** Build a grid of joint sliders in the given container. */
function _buildSliders(containerId, side, limits, initJoints, onInput, onChange) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = '';
  limits.forEach((lim, i) => {
    const q = (initJoints && initJoints[i] !== undefined) ? initJoints[i] : 0;
    const div = document.createElement('div');
    div.className = 'joint';
    div.innerHTML = `
      <label>J${i+1}: <span id="${containerId}_v${i}">${fmt(q)}</span></label>
      <input type="range" min="${lim[0]}" max="${lim[1]}" step="0.001" value="${q}"
             id="${containerId}_r${i}"/>`;
    const inp = div.querySelector('input');
    inp.addEventListener('pointerdown', () => { dragging[side] = true; });
    inp.addEventListener('pointerup',   () => { dragging[side] = false; });
    inp.addEventListener('input', () => {
      const v = Number(inp.value);
      document.getElementById(`${containerId}_v${i}`).textContent = fmt(v);
      if (onInput) onInput(i, v);
    });
    if (onChange) inp.addEventListener('change', () => onChange(i, Number(inp.value)));
    el.appendChild(div);
  });
}

/** Sync slider positions to match hardware state. */
function syncSlidersToState(side, joints) {
  if (dragging[side] || previewLocked[side]) return;
  ['mj', 'sv'].forEach(prefix => {
    const cid = `${side}_${prefix}_joints`;
    joints.forEach((q, i) => {
      const r = document.getElementById(`${cid}_r${i}`);
      const v = document.getElementById(`${cid}_v${i}`);
      if (r) r.value = q;
      if (v) v.textContent = fmt(q);
      if (!previewLocked[side]) preview[side][i] = q;
    });
  });
}

// ─────────────────────────────────────────────────────────────
// MoveJ send
// ─────────────────────────────────────────────────────────────

const MOVEJ_MIN_DELTA = 0.005;  // rad — ignore sends with tiny displacement

function sendMoveJ(side) {
  // Skip if displacement from current hardware state is negligible
  const cur = armState[side].joints;
  const tgt = preview[side];
  if (cur.length && tgt.length) {
    const maxD = Math.max(...tgt.map((q, i) => Math.abs(q - (cur[i] || 0))));
    if (maxD < MOVEJ_MIN_DELTA) return;
  }
  const panel = document.getElementById(CFG[side].panel);
  const speedEl = panel && panel.querySelector(`#${side}_mj_speed`);
  const speed = speedEl ? parseFloat(speedEl.value) || 0.5 : 0.5;
  api(side, 'movej', { joints: tgt.slice(), speed })
    .then(d => setStatus(side, `movej: ${d.ok ? 'ok' : 'fail'} ${d.msg || ''}`));
}

// ─────────────────────────────────────────────────────────────
// Overlay cards
// ─────────────────────────────────────────────────────────────

function updateOverlay(side, state) {
  const el = document.getElementById(`overlay_${side}`);
  if (!el) return;

  const dot   = document.getElementById(`${side}_dot`);
  const conn  = document.getElementById(`${side}_connlabel`);
  const modlb = document.getElementById(`${side}_modlabel`);
  const svled = document.getElementById(`${side}_sv_led`);

  if (dot) {
    dot.className = 'connDot' +
      (state.fault ? ' fault' : state.enabled ? ' enabled' : state.connected !== false ? ' connected' : '');
  }
  if (conn)  conn.textContent  = state.enabled ? 'enabled' : 'disabled';
  if (modlb) modlb.textContent = state.model || '';
  if (svled) svled.className   = 'servoActive' + (state.servo_active ? ' on' : '');

  // Pose rows
  const ee = state.end_effector;
  const poseStr = ee
    ? `EE: (${fmt(ee.pos[0])}, ${fmt(ee.pos[1])}, ${fmt(ee.pos[2])})`
    : 'EE: --';
  ['pose_row', 'movep_pose', 'fd_pose'].forEach(id => {
    const pel = document.getElementById(`${side}_${id}`);
    if (pel) { pel.innerHTML = `<span>Pose</span>${poseStr}`; }
  });

  updateEeDot(side);

  // Stiffness buttons — no persistent highlight (server doesn't return current stiffness)
  // Overlay status card
  const msgs = [state.model || '', state.msg || ''];
  if (state.fault) msgs.push(`⚠ ${state.fault}`);
  el.className = 'overlayCard' + (state.fault ? ' fault' : '');
  el.innerHTML = `<div class="title">${CFG[side].label}</div>${msgs.filter(Boolean).join('<br>')}`;
  if (state.num_joints === 0) el.style.display = 'none';
  else el.style.display = '';
}

// ─────────────────────────────────────────────────────────────
// 6.  SSE state stream
// ─────────────────────────────────────────────────────────────

let pollTimer = null;
let sseActive = false;

function applyState(side, data) {
  const s = armState[side];
  if (data.joints && data.joints.length > 0) {
    s.joints        = data.joints;
    s.limits        = data.limits || s.limits;
    s.enabled       = data.enabled;
    s.moving        = data.moving;
    s.freedrive     = data.freedrive;
    s.servo_active  = data.servo_active;
    s.end_effector  = data.pose || null;
    s.model         = data.model || s.model;
    s.num_joints    = data.num_joints || s.num_joints;
    updateRobot(side);
    if (!previewLocked[side]) syncSlidersToState(side, data.joints);
    updateOverlay(side, data);
  }
}

function startSSE(hz = 30) {
  const url = `/api/stream?hz=${hz}`;
  const es  = new EventSource(url);
  sseActive = true;
  setHud('SSE connected');

  es.onmessage = ev => {
    try {
      const payload = JSON.parse(ev.data);
      for (const side of SIDES) {
        if (payload[side]) applyState(side, payload[side]);
      }
    } catch (_) {}
  };

  es.onerror = () => {
    sseActive = false;
    es.close();
    setHud('SSE disconnected, polling…');
    startPolling(1000);
  };
}

function startPolling(interval = 1000) {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    for (const side of SIDES) {
      try {
        const r = await fetch(`${apiBase(side)}/state`);
        const d = await r.json();
        applyState(side, d);
      } catch (_) {}
    }
  }, interval);
}

// ─────────────────────────────────────────────────────────────
// 7.  API helpers
// ─────────────────────────────────────────────────────────────

function api(side, endpoint, body) {
  return fetch(`${apiBase(side)}/${endpoint}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body || {}),
  }).then(r => r.json());
}

// ─────────────────────────────────────────────────────────────
// 8.  Bootstrap
// ─────────────────────────────────────────────────────────────

async function boot() {
  // Detect single vs dual mode from the first reachable state endpoint.
  // Dual-mode server has /left/api/state; single-arm has /api/state.
  let dualMode = true;
  try {
    const r = await fetch('/left/api/state');
    if (!r.ok) throw new Error();
    const d = await r.json();
    if (!d || d.num_joints === 0) throw new Error();
  } catch (_) {
    dualMode = false;
  }

  if (!dualMode) {
    // Single-arm: figure out which side
    try {
      const r  = await fetch('/api/state');
      const d  = await r.json();
      const sd = d.side || 'right';
      singleArmSide = sd;
      CFG[sd].api   = '/api';
      // Hide the other panel
      const other = sd === 'right' ? 'left' : 'right';
      document.getElementById(CFG[other].panel).style.display = 'none';
      document.getElementById(`overlay_${other}`).style.display = 'none';
      // Apply initial state
      applyState(sd, d);
      armState[sd].limits = d.limits || armState[sd].limits;
      preview[sd] = (d.joints || []).slice();
    } catch (_) {}
  } else {
    // Dual arm
    for (const sd of SIDES) {
      try {
        const r = await fetch(`${CFG[sd].api}/state`);
        const d = await r.json();
        applyState(sd, d);
        armState[sd].limits = d.limits || armState[sd].limits;
        preview[sd] = (d.joints || []).slice();
      } catch (_) {}
    }
  }

  // Render panels
  for (const side of SIDES) {
    const panel = document.getElementById(CFG[side].panel);
    if (panel.style.display === 'none') continue;
    renderPanel(side);
    // Init preview from initial state
    if (!preview[side].length) preview[side] = Array(armState[side].num_joints || 7).fill(0);
  }

  // Load URDF models
  for (const side of SIDES) {
    const panel = document.getElementById(CFG[side].panel);
    if (panel.style.display !== 'none') {
      CFG[side].model = armState[side].urdf_file || armState[side].model.toLowerCase().replace('a1r','a1_r').replace('a1l','a1_l') || CFG[side].model;
      loadRobot(side);
    }
  }

  // Start state stream
  startSSE(30);
}

boot().catch(err => setHud('boot error: ' + err));
