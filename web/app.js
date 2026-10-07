"use strict";

const IMAGE_SIZES = [
  ["1:1 · 1024", 1024, 1024],
  ["1:1 · 2048", 2048, 2048],
  ["4:3", 2400, 1792],
  ["3:4", 1792, 2400],
  ["3:2", 2528, 1696],
  ["2:3", 1696, 2528],
  ["16:9", 2752, 1536],
  ["9:16", 1536, 2752],
  ["依參考圖比例", 0, 0],
  ["自訂", -1, -1],
];
const VIDEO_SIZES = [
  ["864×480", 864, 480],
  ["1344×768（官方）", 1344, 768],
  ["480×864", 480, 864],
  ["736×1312 · 直式", 736, 1312],
  ["自動", 0, 0],
  ["自訂", -1, -1],
];
const STEP_DEFAULT = { image: 40, video: 30 };
const STATUS_TEXT = {
  queued: "排隊中",
  running: "生成中",
  succeeded: "完成",
  failed: "失敗",
  cancelled: "已取消",
};
const STAGE_TEXT = { load: "載入模型", encode: "準備素材", denoise: "取樣", decode: "輸出作品", decode_audio: "輸出音訊", decode_video: "輸出影片", done: "完成" };

const $ = (id) => document.getElementById(id);
const state = {
  mode: "image",
  size: 0,
  refs: [],
  vrefs: [],
  models: [],
  jobs: [],
  activeId: null,
  viewId: null,
  shown: "",
  source: null,
  settled: false,
  histSig: "",
  loraRows: [], loraSelections: {}, fileSelections: { image: {}, video: {} },
  fileSignature: "", apiKey: "", pollTimer: null, refreshing: false, submitting: false, authRequired: false, modelChanging: false,
};

function sizes() {
  return state.mode === "video" ? VIDEO_SIZES : IMAGE_SIZES;
}

function isTerminal(status) {
  return status === "succeeded" || status === "failed" || status === "cancelled";
}

function showFormError(message) {
  const el = $("form-error");
  el.hidden = !message;
  el.textContent = message || "";
}

async function api(path, opts) {
  const headers = Object.assign({}, opts?.headers);
  if (state.apiKey) headers.Authorization = "Bearer " + state.apiKey;
  const res = await fetch(path, Object.assign({ cache: "no-store", signal: AbortSignal.timeout(opts?.timeout || 15000) }, opts, { headers }));
  const text = await res.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch (e) {
      data = null;
    }
  }
  if (!res.ok) {
    const msg = data && data.error && data.error.message ? data.error.message : "請求失敗（" + res.status + "）";
    const error = new Error(msg);
    error.status = res.status;
    throw error;
  }
  return data;
}

function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("讀不到「" + file.name + "」"));
    reader.readAsDataURL(file);
  });
}

function modelId(kind) {
  const list = state.models.filter((m) => m.kind === kind);
  const loaded = list.find((m) => m.loaded);
  return (loaded || list[0] || {}).id || "";
}

// 參考生成（Ref2VA）：依序的圖片／影片／音訊，標籤依種類各自編號（同提示詞裡的寫法）。
const VREF_TAG = { image: "Picture", video: "Video", audio: "Audio" };
const VREF_LIMIT = { image: 9, video: 3, audio: 3 };

function vrefKind(file) {
  const t = file.type || "";
  if (t.startsWith("image/") || /\.(png|jpe?g)$/i.test(file.name)) return "image";
  if (t.startsWith("video/") || /\.(mp4|mov|webm|mkv)$/i.test(file.name)) return "video";
  if (t.startsWith("audio/") || /\.(wav|mp3|m4a|flac|ogg)$/i.test(file.name)) return "audio";
  return "";
}

function renderVrefs() {
  updateRequestPreview();
  const list = $("vref-list");
  list.replaceChildren();
  const count = { image: 0, video: 0, audio: 0 };
  state.vrefs.forEach((ref, index) => {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.className = "ref-name";
    const tag = document.createElement("span");
    tag.className = "ref-tag";
    tag.textContent = "<" + VREF_TAG[ref.kind] + " " + ++count[ref.kind] + ">";
    label.append(tag, document.createTextNode(ref.name));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "移除";
    remove.setAttribute("aria-label", "移除 " + ref.name);
    remove.addEventListener("click", () => {
      state.vrefs.splice(index, 1);
      renderVrefs();
    });
    li.append(label, remove);
    list.append(li);
  });
}

async function addVrefs(fileList) {
  for (const file of Array.from(fileList || [])) {
    const kind = vrefKind(file);
    if (!kind) {
      showFormError("「" + file.name + "」不是圖片、影片或音訊");
      continue;
    }
    if (state.vrefs.length >= 12) {
      showFormError("參考最多 12 個");
      break;
    }
    if (state.vrefs.filter((r) => r.kind === kind).length >= VREF_LIMIT[kind]) {
      showFormError(VREF_TAG[kind] + " 參考最多 " + VREF_LIMIT[kind] + " 個");
      continue;
    }
    try { state.vrefs.push({ kind, name: file.name, url: await readFile(file) }); }
    catch (e) { showFormError(e.message); }
  }
  renderVrefs();
}

function applyMode() {
  const image = state.mode === "image";
  $("mode-image").setAttribute("aria-pressed", image ? "true" : "false");
  $("mode-video").setAttribute("aria-pressed", image ? "false" : "true");
  $("negative-wrap").hidden = !image;
  $("cfg-wrap").hidden = !image;
  $("refs-wrap").hidden = !image;
  $("frames-wrap").hidden = image;
  $("frame-files").hidden = image;
  $("shift-wrap").hidden = image;
  $("sched-wrap").hidden = !image;
  $("vrefs-wrap").hidden = image;
  renderModelFiles();
  updateTasks();
  renderSizes();
  const s = selectedSize();
  $("skeleton").style.aspectRatio = s.w + " / " + s.h;
}

function setMode(mode) {
  if (state.mode === mode) return;
  const steps = Number($("steps").value);
  if (steps === STEP_DEFAULT[state.mode]) $("steps").value = String(STEP_DEFAULT[mode]);
  state.mode = mode;
  clearAcceleration();
  state.size = 0;
  $("width").max = $("height").max = mode === "image" ? "2752" : "1920";
  $("width").value = mode === "image" ? "1024" : "736";
  $("height").value = mode === "image" ? "1024" : "1312";
  applyMode();
  updateRequestPreview();
}

function setBusy(busy) {
  $("cancel").hidden = !busy;
  const fields = $("form").querySelectorAll("input, textarea, select, button");
  fields.forEach((el) => {
    if (el.id === "cancel" || el.id === "generate") return;
    el.disabled = busy;
  });
}

function updateRun(job) {
  const button = $("generate");
  const status = $("status");
  const meter = $("meter");
  const bar = $("meter-bar");
  status.classList.remove("warn");
  if (!job || isTerminal(job.status)) {
    setBusy(state.modelChanging);
    button.textContent = "生成";
    meter.hidden = true;
    bar.style.width = "0%";
    status.textContent = job && job.status === "cancelled" ? "已取消" : "";
    return;
  }
  setBusy(true);
  const prog = job.progress || {};
  const step = prog.step || 0;
  const total = prog.total || 0;
  const label = total ? step + "/" + total : "";
  const stage = STAGE_TEXT[prog.stage] || "";
  meter.hidden = false;
  if (job.status === "queued") {
    button.textContent = "排隊中";
    status.textContent = "排隊中，等待顯示卡";
    status.classList.add("warn");
    bar.style.width = "0%";
    return;
  }
  button.textContent = label ? "生成中 " + label : "生成中";
  status.textContent = stage && label ? stage + " " + label : "生成中";
  bar.style.width = total ? Math.min(100, (step / total) * 100) + "%" : "0%";
}

function clearViewer() {
  $("download").hidden = true;
  $("result-img").hidden = true;
  const video = $("result-video");
  video.pause();
  video.removeAttribute("src");
  video.hidden = true;
  video.load();
  $("skeleton").hidden = true;
  $("viewer-error").hidden = true;
  $("viewer-error").textContent = "";
  $("caption").hidden = true;
}

function showJob(job) {
  if (!job) return;
  if (job.status === "queued" || job.status === "running") {
    if (state.shown.startsWith(job.id + ":run")) return;
    state.shown = job.id + ":run";
    clearViewer();
    $("empty").hidden = true;
    const s = selectedSize();
    $("skeleton").style.aspectRatio = s.w + " / " + s.h;
    $("skeleton").hidden = false;
    return;
  }
  if (job.status !== "succeeded" || !job.result || !job.result.url) {
    state.shown = job.id + ":bad";
    clearViewer();
    $("empty").hidden = true;
    const err = $("viewer-error");
    err.hidden = false;
    err.textContent = (job.error && job.error.message) || STATUS_TEXT[job.status] || "生成失敗";
    return;
  }
  const key = job.id + ":" + job.result.url;
  if (state.shown === key) return;
  state.shown = key;
  clearViewer();
  $("empty").hidden = true;
  const url = job.result.url;
  $("download").href = url;
  $("download").download = job.id + (job.type === "video" ? ".mp4" : ".png");
  $("download").hidden = false;
  if (job.type === "video") {
    const video = $("result-video");
    video.src = url;
    video.hidden = false;
  } else {
    const img = $("result-img");
    img.alt = job.prompt || "生成結果";
    img.src = url;
    img.hidden = false;
  }
  const cap = $("caption");
  const sec = job.result.seconds;
  let text = "種子 " + job.result.seed;
  if (typeof sec === "number") text += " · 生成耗時 " + sec.toFixed(1) + " 秒";
  const params = job.parameters || {};
  if (params.width && params.height) text += " · " + params.width + "×" + params.height;
  if (params.duration_seconds) text += " · 影片 " + params.duration_seconds.toFixed(2) + " 秒 / 24 fps";
  cap.textContent = text;
  cap.hidden = false;
}

function onJob(job) {
  if (job.id === state.viewId) showJob(job);
  if (job.id === state.activeId) {
    updateRun(job);
    if (isTerminal(job.status)) state.activeId = null;
  }
}

function closeSource() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
  if (!state.source) return;
  state.source.close();
  state.source = null;
}

function watch(id) {
  closeSource();
  state.settled = false;
  if (state.apiKey) return pollWatch(id);
  const es = new EventSource("/v1/jobs/" + encodeURIComponent(id) + "/events");
  state.source = es;
  es.onmessage = (ev) => {
    let job = null;
    try {
      job = JSON.parse(ev.data);
    } catch (e) {
      return;
    }
    onJob(job);
    if (isTerminal(job.status)) {
      state.settled = true;
      es.close();
      if (state.source === es) state.source = null;
      refreshMeta();
    }
  };
  es.onerror = () => {
    es.close();
    if (state.source === es) state.source = null;
    if (!state.settled) pollWatch(id);
  };
}

function renderHistory() {
  const sig = state.jobs
    .map((j) => j.id + j.status + (j.progress ? j.progress.step : 0) + (j.result ? j.result.url : ""))
    .join("|");
  if (sig === state.histSig) return;
  state.histSig = sig;
  const box = $("history");
  box.replaceChildren();
  if (!state.jobs.length) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "完成的圖像與影片會排在這裡。";
    box.append(p);
    return;
  }
  for (const job of state.jobs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "hist" + (job.id === state.viewId ? " current" : "");
    button.title = job.prompt || job.id;
    if (job.status === "succeeded" && job.result && job.result.url && job.type !== "video") {
      const img = document.createElement("img");
      img.alt = "";
      img.src = job.result.url;
      button.append(img);
    } else {
      const fallback = document.createElement("div");
      fallback.className = "thumb-fallback";
      fallback.textContent = job.type === "video" ? "影片" : STATUS_TEXT[job.status] || "圖像";
      button.append(fallback);
    }
    const meta = document.createElement("div");
    meta.className = "history-meta";
    const left = document.createElement("span");
    left.textContent = STATUS_TEXT[job.status] || job.status;
    const right = document.createElement("span");
    right.textContent = job.result && typeof job.result.seconds === "number" ? job.result.seconds.toFixed(1) + " 秒" : "";
    meta.append(left, right);
    button.append(meta);
    button.addEventListener("click", () => {
      state.viewId = job.id;
      state.shown = "";
      state.histSig = "";
      showJob(job);
      renderHistory();
      if (!isTerminal(job.status) && job.id === state.activeId) watch(job.id);
    });
    box.append(button);
  }
}

function renderModels(data) {
  const previous = state.models.map((m) => m.id).join();
  state.models = (data && data.data) || [];
  if (previous !== state.models.map((m) => m.id).join()) renderLoras(state.loraRows);
  const box = $("models");
  box.replaceChildren();
  state.models.forEach((m) => {
    const el = document.createElement("span");
    el.className = "model";
    const dot = document.createElement("span");
    dot.className = "dot" + (m.loaded ? " on" : "");
    const name = document.createElement("span");
    name.textContent = m.id + (m.loaded ? " · 已載入" : " · 按需載入");
    const detail = document.createElement("span");
    detail.className = "model-detail";
    detail.textContent = m.detail;
    el.append(dot, name, detail);
    box.append(el);
  });
  updateModelSummary();
}

async function refreshMeta() {
  if (state.refreshing || document.hidden || state.authRequired) return;
  state.refreshing = true;
  const start = performance.now();
  try {
    const [models, sys, jobs, network] = await Promise.all([api("/v1/models"), api("/v1/system"), api("/v1/jobs"), api("/v1/network")]);
    renderNetwork(network);
    renderModels(models);
    renderModelFiles();
    renderSystem(sys);
    $("connection").textContent = "API 已連線";
    $("connection").className = "connected";
    $("api-state").textContent = "回應 " + Math.round(performance.now() - start) + " ms";
    $("api-error").textContent = "";
    state.jobs = (jobs && jobs.data) || [];
    const running = state.jobs.find((j) => !isTerminal(j.status));
    if (!state.activeId && running) { state.activeId = running.id; state.viewId = running.id; watch(running.id); }
    updateModelActions();
    renderHistory();
    const viewed = state.jobs.find((j) => j.id === state.viewId);
    if (viewed) onJob(viewed);
    if (state.activeId) updateRun(state.jobs.find((j) => j.id === state.activeId));
  } catch (e) {
    if (e.status === 401) {
      state.authRequired = true;
      document.querySelectorAll("[data-view]").forEach((b) => { const active = b.dataset.view === "api-panel"; b.setAttribute("aria-pressed", String(active)); $(b.dataset.view).hidden = !active; });
    }
    $("connection").textContent = "API 無法連線";
    $("connection").className = "warn";
    $("api-state").textContent = "無法連線";
    $("monitor-error").textContent = "資料可能已過期：" + e.message;
    $("api-error").textContent = e.status === 401 ? "服務需要 API 金鑰，請輸入後按測試連線。" : e.message;
  } finally { state.refreshing = false; }
}

function pollWatch(id) {
  if (state.pollTimer) clearInterval(state.pollTimer);
  let pending = false;
  state.pollTimer = setInterval(async () => {
    if (pending) return;
    pending = true;
    try {
      const job = await api("/v1/jobs/" + encodeURIComponent(id));
      onJob(job);
      if (isTerminal(job.status)) { closeSource(); refreshMeta(); }
    } catch (e) { showFormError(e.message); }
    finally { pending = false; }
  }, 1000);
}

async function loadLoras() {
  try {
    const data = await api("/v1/loras");
    state.loraRows = (data && data.data) || [];
    renderLoras(state.loraRows);
    if (state.activeId) setBusy(true);
  } catch (e) {
    const box = $("loras");
    box.replaceChildren();
    const p = document.createElement("p");
    p.className = "error";
    p.textContent = e.message;
    box.append(p);
  }
}

function renderRefs() {
  const list = $("ref-list");
  $("refs-count").textContent = state.refs.length + " / 10";
  updateRequestPreview();
  list.replaceChildren();
  state.refs.forEach((ref, index) => {
    const li = document.createElement("li");
    const thumbnail = document.createElement("img");
    thumbnail.src = ref.url;
    thumbnail.alt = "參考圖 " + (index + 1);
    thumbnail.className = "ref-thumb";
    const name = document.createElement("span");
    name.textContent = ref.name;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "移除";
    remove.setAttribute("aria-label", "移除 " + ref.name);
    remove.addEventListener("click", () => {
      state.refs.splice(index, 1);
      renderRefs();
    });
    li.append(thumbnail, name, remove);
    list.append(li);
  });
}

async function addFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  const room = 10 - state.refs.length;
  if (room <= 0) {
    showFormError("參考圖最多 10 張");
    return;
  }
  if (files.length > room) showFormError("參考圖最多 10 張，多的已略過");
  for (const file of files.slice(0, room)) {
    const ok = file.type === "image/png" || file.type === "image/jpeg" || /\.(png|jpe?g)$/i.test(file.name);
    if (!ok) {
      showFormError("只接受 PNG 或 JPEG");
      continue;
    }
    try { state.refs.push({ name: file.name, url: await readFile(file) }); }
    catch (e) { showFormError(e.message); }
  }
  renderRefs();
}

async function onSubmit(ev) {
  ev.preventDefault();
  if (state.activeId || state.submitting || state.modelChanging) return;
  state.submitting = true;
  setBusy(true);
  $("generate").disabled = true;
  showFormError("");
  try {
    const body = await buildRequest();
    const job = await api("/v1/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.activeId = job.id;
    state.viewId = job.id;
    state.shown = "";
    state.histSig = "";
    showJob(job);
    updateRun(job);
    watch(job.id);
  } catch (e) {
    showFormError(e.message || "送出失敗");
  } finally {
    state.submitting = false;
    $("generate").disabled = false;
    if (!state.activeId) setBusy(false);
  }
}

function bind() {
  $("mode-image").addEventListener("click", () => setMode("image"));
  $("mode-video").addEventListener("click", () => setMode("video"));
  $("form").addEventListener("submit", onSubmit);
  $("prompt").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {
      ev.preventDefault();
      $("form").requestSubmit();
    }
  });
  $("randomize").addEventListener("click", () => {
    $("seed").value = String(Math.floor(Math.random() * 2147483648));
  });
  $("cancel").addEventListener("click", async () => {
    if (!state.activeId) return;
    try {
      onJob(await api("/v1/jobs/" + encodeURIComponent(state.activeId), { method: "DELETE" }));
    } catch (e) {
      showFormError(e.message);
    }
  });
  $("ref-input").addEventListener("change", (ev) => {
    addFiles(ev.target.files).then(() => {
      ev.target.value = "";
    });
  });
  $("vref-input").addEventListener("change", (ev) => {
    addVrefs(ev.target.files).then(() => {
      ev.target.value = "";
    });
  });
  const vdrop = $("vdrop");
  ["dragenter", "dragover"].forEach((name) => {
    vdrop.addEventListener(name, (ev) => {
      ev.preventDefault();
      vdrop.classList.add("hot");
    });
  });
  vdrop.addEventListener("dragleave", () => vdrop.classList.remove("hot"));
  vdrop.addEventListener("drop", (ev) => {
    ev.preventDefault();
    vdrop.classList.remove("hot");
    addVrefs(ev.dataTransfer.files);
  });
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((name) => {
    drop.addEventListener(name, (ev) => {
      ev.preventDefault();
      drop.classList.add("hot");
    });
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("hot"));
  drop.addEventListener("drop", (ev) => {
    ev.preventDefault();
    drop.classList.remove("hot");
    addFiles(ev.dataTransfer.files);
  });
}

applyMode();
bind();
bindControls();
refreshMeta();
loadLoras();
setInterval(refreshMeta, 3000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshMeta(); });
