// Live Go1 joystick in the browser: MuJoCo (WebAssembly) physics + the trained PPO policy
// in plain JavaScript + three.js rendering. Mirrors export_web.py's CpuGo1 / NumpyPolicy.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const b64bytes = s => {
  const bin = atob(s), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
};
const b64f32 = s => new Float32Array(b64bytes(s).buffer);

// ---- Policy: normalise -> MLP (silu) -> tanh(loc)
export class Policy {
  constructor(P) {
    this.mean = b64f32(P.obs_mean);
    this.std = b64f32(P.obs_std);
    this.layers = P.layers.map(l => ({ n: l.in, m: l.out, w: b64f32(l.w), b: b64f32(l.b) }));
    this.bufs = this.layers.map(l => new Float32Array(l.m));
  }
  act(obs) {
    let x = new Float32Array(obs.length);
    for (let i = 0; i < obs.length; i++) x[i] = (obs[i] - this.mean[i]) / this.std[i];
    this.layers.forEach((L, li) => {
      const y = this.bufs[li];
      y.set(L.b);
      for (let i = 0; i < L.n; i++) {
        const xi = x[i];
        if (xi === 0) continue;
        const row = i * L.m;
        for (let j = 0; j < L.m; j++) y[j] += xi * L.w[row + j];
      }
      if (li < this.layers.length - 1) for (let j = 0; j < L.m; j++) y[j] = y[j] / (1 + Math.exp(-y[j]));
      x = y;
    });
    const out = new Float32Array(x.length / 2);
    for (let i = 0; i < out.length; i++) out[i] = Math.tanh(x[i]);
    return out;
  }
}

export async function startSim({ canvas, base = "web/", onStatus = () => {}, onFrame = () => {} }) {
  onStatus("Loading the physics engine (10 MB, first visit only)…");
  const json = f => fetch(base + f).then(r => { if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`); return r.json(); });
  const [factory, wasmBinary, P, S] = await Promise.all([
    import(new URL(base + "vendor/mujoco.js", document.baseURI).href).then(m => m.default),
    fetch(base + "vendor/mujoco.wasm").then(r => { if (!r.ok) throw new Error(`mujoco.wasm: HTTP ${r.status}`); return r.arrayBuffer(); }),
    json("policy.json"),
    json("go1_scene.json"),
  ]);
  const meshBuf = b64bytes(S.mesh_data).buffer;
  onStatus("Starting MuJoCo…");
  const mujoco = await factory({ wasmBinary });
  const vfs = new mujoco.MjVFS();
  vfs.addBuffer("go1.mjb", b64bytes(S.mjb));
  const model = mujoco.MjModel.from_binary_path("go1.mjb", vfs);
  const data = new mujoco.MjData(model);
  const policy = new Policy(P);
  if (P.ref_obs) {
    // Self-check against the NumPy reference written by export_web.py.
    const a = policy.act(Float32Array.from(P.ref_obs));
    const err = Math.max(...Array.from(a, (v, i) => Math.abs(v - P.ref_act[i])));
    if (err > 1e-3) console.warn(`Policy port differs from NumPy reference by ${err}`);
    policy.refError = err;
  }
  const def = Float32Array.from(P.default_pose);
  const [linAdr] = P.sensors.local_linvel, [gyroAdr] = P.sensors.gyro;

  // ---- Simulation state
  let lastAct = new Float32Array(12);
  const cmd = [0, 0, 0];          // smoothed command sent to the policy
  const target = [0, 0, 0];       // what the keys ask for
  const actual = [0, 0, 0];       // measured body-frame vx, vy, yaw rate
  let simTime = 0, fallen = false, fallTimer = 0, falls = 0;

  function reset() {
    mujoco.mj_resetDataKeyframe(model, data, P.home_key);
    const ctrl = data.ctrl;
    for (let i = 0; i < 12; i++) ctrl[i] = def[i];
    mujoco.mj_forward(model, data);
    lastAct = new Float32Array(12);
    cmd.fill(0);
    fallen = false; fallTimer = 0; simTime = 0;
  }

  function observe() {
    const s = data.sensordata, q = data.qpos, v = data.qvel, R = data.site_xmat;
    const k = P.imu_site * 9;
    const o = new Float32Array(48);
    o[0] = s[linAdr]; o[1] = s[linAdr + 1]; o[2] = s[linAdr + 2];
    o[3] = s[gyroAdr]; o[4] = s[gyroAdr + 1]; o[5] = s[gyroAdr + 2];
    // gravity in the IMU frame: R^T [0, 0, -1]
    o[6] = -R[k + 6]; o[7] = -R[k + 7]; o[8] = -R[k + 8];
    for (let i = 0; i < 12; i++) {
      o[9 + i] = q[7 + i] - def[i];
      o[21 + i] = v[6 + i];
      o[33 + i] = lastAct[i];
    }
    o[45] = cmd[0]; o[46] = cmd[1]; o[47] = cmd[2];
    return o;
  }

  function controlStep() {
    // Ease the command toward the keys (a real joystick does not jump instantly).
    const rate = 4 * P.ctrl_dt;
    for (let i = 0; i < 3; i++) cmd[i] += Math.max(-rate, Math.min(rate, target[i] - cmd[i]));
    const a = policy.act(observe());
    const ctrl = data.ctrl;
    for (let i = 0; i < 12; i++) ctrl[i] = def[i] + a[i] * P.action_scale;
    for (let n = 0; n < P.n_substeps; n++) mujoco.mj_step(model, data);
    lastAct = a;
    simTime += P.ctrl_dt;
    const s = data.sensordata;
    for (let i = 0; i < 2; i++) actual[i] += 0.2 * (s[linAdr + i] - actual[i]);
    actual[2] += 0.2 * (s[gyroAdr + 2] - actual[2]);
    // Fallen: torso upside down or on the ground.
    const R = data.site_xmat, up = R[P.imu_site * 9 + 8], z = data.qpos[2];
    if (!fallen && (up < 0.3 || z < 0.12)) { fallen = true; falls++; }
  }

  function shove() {
    const ang = Math.random() * Math.PI * 2, mag = 0.8 + Math.random() * 0.6;
    const v = data.qvel;
    v[0] += Math.cos(ang) * mag;
    v[1] += Math.sin(ang) * mag;
  }

  // ---- three.js scene (MuJoCo is z-up)
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(1.4, -1.6, 0.8);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 0.8;
  controls.maxDistance = 6;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.target.set(0, 0, 0.25);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x444455, 1.1);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2, near: 0.1, far: 10 });
  scene.add(sun, sun.target);

  // Ground: checker texture so motion is readable; 1 m tiles.
  const tex = (() => {
    const c = document.createElement("canvas");
    c.width = c.height = 128;
    const g = c.getContext("2d");
    return { c, g };
  })();
  const groundMat = new THREE.MeshStandardMaterial({ roughness: 0.95 });
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), groundMat);
  ground.receiveShadow = true;
  scene.add(ground);

  function applyTheme() {
    const attr = document.documentElement.getAttribute("data-theme");
    const dark = attr === "dark" || (attr !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
    const a = dark ? "#1a1d2c" : "#e8eaf2", b = dark ? "#212538" : "#dfe2ec", line = dark ? "#2d3350" : "#cfd3e3";
    const { c, g } = tex;
    g.fillStyle = a; g.fillRect(0, 0, 128, 128);
    g.fillStyle = b; g.fillRect(0, 0, 64, 64); g.fillRect(64, 64, 64, 64);
    g.strokeStyle = line; g.lineWidth = 2; g.strokeRect(0, 0, 128, 128);
    const t = new THREE.CanvasTexture(c);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(100, 100);
    t.anisotropy = 8;
    t.colorSpace = THREE.SRGBColorSpace;
    groundMat.map = t; groundMat.needsUpdate = true;
    scene.background = new THREE.Color(dark ? "#10121d" : "#f2f3f7");
    scene.fog = new THREE.Fog(scene.background, 8, 30);
  }
  applyTheme();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
  new MutationObserver(applyTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  // Robot: one group per body, geoms at their local pose.
  const bodies = [];
  for (let b = 0; b < S.nbody; b++) { const g = new THREE.Group(); scene.add(g); bodies.push(g); }
  const meshGeo = {};
  for (const [id, mm] of Object.entries(S.meshes)) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(meshBuf, mm.v_off, mm.v_count), 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint16Array(meshBuf, mm.f_off, mm.f_count), 1));
    geo.computeVertexNormals();
    meshGeo[id] = geo;
  }
  for (const g of S.geoms) {
    let geo;
    if (g.mesh !== undefined) geo = meshGeo[g.mesh];
    else if (g.type === 6) geo = new THREE.BoxGeometry(g.size[0] * 2, g.size[1] * 2, g.size[2] * 2);
    else if (g.type === 2) geo = new THREE.SphereGeometry(g.size[0], 20, 12);
    else continue;
    const c = new THREE.Color().setRGB(g.rgba[0], g.rgba[1], g.rgba[2], THREE.SRGBColorSpace);
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: c, roughness: 0.55, metalness: 0.15 }));
    mesh.position.set(...g.pos);
    mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    mesh.castShadow = true;
    bodies[g.body].add(mesh);
  }

  // Ground arrows: amber = joystick command, indigo = measured velocity (dashboard colours).
  // The command arrow has a wider head, so it shows as an amber halo when tracking is good.
  function arrow(color, head) {
    const a = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.5, color, head[0], head[1]);
    a.userData.head = head;
    scene.add(a);
    return a;
  }
  const cmdArrow = arrow(css("--cmd") || "#b86f00", [0.16, 0.14]);
  const actArrow = arrow(css("--act") || "#3a45a6", [0.12, 0.07]);

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * renderer.getPixelRatio()) || canvas.height !== Math.round(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
  }

  const torsoPrev = new THREE.Vector3();
  function draw() {
    const xp = data.xpos, xq = data.xquat;
    for (let b = 1; b < S.nbody; b++) {
      bodies[b].position.set(xp[3 * b], xp[3 * b + 1], xp[3 * b + 2]);
      bodies[b].quaternion.set(xq[4 * b + 1], xq[4 * b + 2], xq[4 * b + 3], xq[4 * b]);
    }
    const t = P.torso_body;
    const torso = new THREE.Vector3(xp[3 * t], xp[3 * t + 1], xp[3 * t + 2]);
    // Camera follows the robot; the user can still orbit around it.
    const delta = torso.clone().sub(torsoPrev);
    delta.z = 0;
    camera.position.add(delta);
    controls.target.set(torso.x, torso.y, 0.25);
    torsoPrev.copy(torso);
    sun.position.set(torso.x + 2, torso.y - 1.5, 4);
    sun.target.position.set(torso.x, torso.y, 0);

    const yaw = Math.atan2(2 * (xq[4 * t] * xq[4 * t + 3] + xq[4 * t + 1] * xq[4 * t + 2]),
      1 - 2 * (xq[4 * t + 2] ** 2 + xq[4 * t + 3] ** 2));
    const place = (arr, vx, vy, z) => {
      const wx = Math.cos(yaw) * vx - Math.sin(yaw) * vy, wy = Math.sin(yaw) * vx + Math.cos(yaw) * vy;
      const len = Math.hypot(wx, wy);
      arr.visible = len > 0.05;
      if (!arr.visible) return;
      arr.position.set(torso.x, torso.y, z);
      arr.setDirection(new THREE.Vector3(wx / len, wy / len, 0));
      arr.setLength(0.25 + len * 0.6, ...arr.userData.head);
    };
    place(cmdArrow, cmd[0], cmd[1], 0.012);
    place(actArrow, actual[0], actual[1], 0.018);
    controls.update();
    renderer.render(scene, camera);
  }

  // ---- Main loop: fixed 50 Hz control, rendering every animation frame.
  reset();
  torsoPrev.set(data.xpos[3 * P.torso_body], data.xpos[3 * P.torso_body + 1], 0);
  let acc = 0, last = performance.now(), paused = false, stepsThisSec = 0, rtf = 1, rtfT = last;
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (!paused) {
      acc += dt;
      let n = 0;
      while (acc >= P.ctrl_dt && n < 8) { controlStep(); acc -= P.ctrl_dt; n++; stepsThisSec++; }
      if (n === 8) acc = 0;  // too slow to keep up: drop time rather than spiral
      if (fallen) { fallTimer += dt; if (fallTimer > 2.5) reset(); }
    }
    if (now - rtfT > 1000) { rtf = (stepsThisSec * P.ctrl_dt) / ((now - rtfT) / 1000); stepsThisSec = 0; rtfT = now; }
    resize();
    draw();
    onFrame({ cmd, target, actual, simTime, fallen, falls, rtf, paused });
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  onStatus(null);

  return {
    meta: P,
    policyRefError: policy.refError,
    setTarget(vx, vy, wz) { target[0] = vx; target[1] = vy; target[2] = wz; },
    reset,
    shove,
    setPaused(p) { paused = p; last = performance.now(); },
    // Step the simulation synchronously (used by the headless self-test).
    advance(seconds) {
      for (let n = Math.round(seconds / P.ctrl_dt); n > 0; n--) controlStep();
      resize();
      draw();
      return { cmd: [...cmd], actual: [...actual], simTime, fallen, falls };
    },
  };
}
