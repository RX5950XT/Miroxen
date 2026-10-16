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
  ["1344×768 官方", 1344, 768],
  ["480×864", 480, 864],
  ["736×1312 直式", 736, 1312],
  // 以下是官方其他比例在短邊 768、上限 1344×768 像素時的畫布（同 h3_canvas）
  ["21:9 · 1536×672", 1536, 672],
  ["4:3 · 1024×768", 1024, 768],
  ["1:1 · 768", 768, 768],
  ["3:4 · 768×1024", 768, 1024],
  ["自動", 0, 0],
  ["自訂", -1, -1],
];
const TURBO_ID = "qwen-image-2.1-turbo";
const MODEL_NAME = { "qwen-image-2.1": "Qwen 標準", [TURBO_ID]: "Qwen Turbo", "minimax-h3": "H3 影音" };
function stepDefault(mode) { return mode === "video" ? 30 : state.imageModel === TURBO_ID ? 8 : 40; }
const STATUS_TEXT = { queued: "排隊中", running: "生成中", succeeded: "完成", failed: "失敗", cancelled: "已取消" };
const STAGE_TEXT = { load: "載入模型", encode: "準備素材", denoise: "取樣", decode: "輸出作品", decode_audio: "輸出音訊", decode_video: "輸出影片", done: "完成" };

const $ = (id) => document.getElementById(id);
const state = {
  mode: "image", size: 0, refs: [], vrefs: [], frames: {}, models: [], jobs: [],
  activeId: null, viewId: null, shown: "", source: null, settled: false, histSig: "",
  imageModel: "qwen-image-2.1", libraryFilter: "all",
  loraRows: [], loraSelections: {}, fileSelections: {}, installPoll: null,
  fileSignature: "", apiKey: "", pollTimer: null, refreshing: false, submitting: false, authRequired: false, modelChanging: false,
};

const THEME_COLORS = ["bg", "panel", "surface", "surface-2", "ink", "ink-muted", "line", "accent", "accent-ink", "danger", "warn", "ok"];

function applyTheme(theme, colors) {
  const root = document.documentElement;
  if (theme === "light" || theme === "dark") root.dataset.theme = theme;
  if (!colors || typeof colors !== "object" || Array.isArray(colors)) return;
  for (const name of THEME_COLORS) {
    const value = colors[name];
    if (typeof value === "string" && CSS.supports("color", value)) root.style.setProperty("--" + name, value);
  }
}

function initEmbedding() {
  const params = new URLSearchParams(location.search);
  const embed = params.get("embed") === "1";
  if (embed) document.documentElement.dataset.embed = "1";
  applyTheme(params.get("theme"), Object.fromEntries(THEME_COLORS.map((name) => [name, params.get("c-" + name)])));
  if (embed) window.addEventListener("message", (event) => {
    if (event.source !== window.parent || !event.data || event.data.type !== "miroxen:theme") return;
    applyTheme(event.data.theme, event.data.colors);
  });
  if (embed) embedModelPopover();
  // 外殼（AxonDeck）管引擎程序與檔案總管：按鈕只通知 parent，訊息不帶任何秘密
  if (embed && window.parent !== window) for (const button of document.querySelectorAll("[data-host]")) {
    button.hidden = false;
    button.addEventListener("click", () => window.parent.postMessage({ type: "miroxen:host", action: button.dataset.host }, "*"));
  }
  const key = /^#key=([A-Za-z0-9_-]{16,128})$/.exec(location.hash);
  if (key) {
    state.apiKey = key[1];
    history.replaceState(history.state, "", location.pathname + location.search);
  }
}

// 內嵌時「模型與硬體」「API」交給外殼：模型清單改成右上角模型按鈕的彈出面板（搬 DOM，id 不變，渲染照舊）
function embedModelPopover() {
  const section = $("models").closest(".panel");
  section.id = "model-pop";
  section.classList.add("model-pop");
  section.popover = "auto";
  document.body.append(section);
  const toggle = $("model-toggle");
  delete toggle.dataset.goto;
  toggle.setAttribute("aria-label", "模型狀態，切換或卸載模型");
  toggle.setAttribute("aria-haspopup", "dialog");
  toggle.popoverTargetElement = section;
}

function sizes() { return state.mode === "video" ? VIDEO_SIZES : IMAGE_SIZES; }
function isTerminal(status) { return status === "succeeded" || status === "failed" || status === "cancelled"; }

function showFormError(message) {
  const box = $("form-error");
  box.hidden = !message;
  box.textContent = message || "";
}

async function api(path, opts) {
  const headers = Object.assign({}, opts?.headers);
  if (state.apiKey) headers.Authorization = "Bearer " + state.apiKey;
  const res = await fetch(path, Object.assign({ cache: "no-store", signal: AbortSignal.timeout(opts?.timeout || 15000) }, opts, { headers }));
  const text = await res.text();
  let data = null;
  if (text) { try { data = JSON.parse(text); } catch (e) { data = null; } }
  if (!res.ok) {
    const error = new Error(data && data.error && data.error.message ? data.error.message : "請求失敗（" + res.status + "）");
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
  if (kind === "image") return state.imageModel;
  return (state.models.find((m) => m.kind === kind) || {}).id || "minimax-h3";
}
function fileSelection(id) { return state.fileSelections[id] || (state.fileSelections[id] = {}); }

// ---- 參考素材 ----
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

function refItem(content, name, onRemove) {
  const li = document.createElement("li"), label = document.createElement("span"), remove = document.createElement("button");
  label.className = "ref-name";
  label.append(content, document.createTextNode(name));
  remove.type = "button"; remove.className = "ghost small"; remove.textContent = "移除";
  remove.setAttribute("aria-label", "移除 " + name);
  remove.addEventListener("click", onRemove);
  li.append(label, remove);
  return li;
}

function renderVrefs() {
  const count = { image: 0, video: 0, audio: 0 };
  $("vrefs-count").textContent = state.vrefs.length + " / 12";
  $("vref-list").replaceChildren(...state.vrefs.map((ref, index) => {
    const tag = document.createElement("code");
    tag.className = "ref-tag";
    tag.textContent = "<" + VREF_TAG[ref.kind] + " " + ++count[ref.kind] + ">";
    tag.title = "按一下插入提示詞";
    tag.addEventListener("click", (ev) => { ev.preventDefault(); insertPrompt(tag.textContent); });
    return refItem(tag, ref.name, () => { state.vrefs.splice(index, 1); renderVrefs(); });
  }));
  updateRequestPreview();
}

function insertPrompt(text) {
  const p = $("prompt"), at = p.selectionStart ?? p.value.length;
  p.value = p.value.slice(0, at) + text + p.value.slice(p.selectionEnd ?? at);
  p.focus(); p.selectionStart = p.selectionEnd = at + text.length;
  updateRequestPreview();
}

async function addVrefs(fileList) {
  for (const file of Array.from(fileList || [])) {
    const kind = vrefKind(file);
    if (!kind) { showFormError("「" + file.name + "」不是圖片、影片或音訊"); continue; }
    if (state.vrefs.length >= 12) { showFormError("參考最多 12 個"); break; }
    if (state.vrefs.filter((r) => r.kind === kind).length >= VREF_LIMIT[kind]) { showFormError(VREF_TAG[kind] + " 參考最多 " + VREF_LIMIT[kind] + " 個"); continue; }
    try { state.vrefs.push({ kind, name: file.name, url: await readFile(file) }); }
    catch (e) { showFormError(e.message); }
  }
  renderVrefs();
}

function renderRefs() {
  $("refs-count").textContent = state.refs.length + " / 10";
  $("ref-list").replaceChildren(...state.refs.map((ref, index) => {
    const thumb = document.createElement("img");
    thumb.src = ref.url; thumb.alt = "參考圖 " + (index + 1); thumb.className = "ref-thumb";
    return refItem(thumb, ref.name, () => { state.refs.splice(index, 1); renderRefs(); });
  }));
  syncMaskPaint();
  updateRequestPreview();
}

async function addFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  const room = 10 - state.refs.length;
  if (room <= 0) { showFormError("參考圖最多 10 張"); return; }
  if (files.length > room) showFormError("參考圖最多 10 張，多的已略過");
  for (const file of files.slice(0, room)) {
    const ok = file.type === "image/png" || file.type === "image/jpeg" || /\.(png|jpe?g)$/i.test(file.name);
    if (!ok) { showFormError("只接受 PNG 或 JPEG"); continue; }
    try { state.refs.push({ name: file.name || "貼上的圖片.png", url: await readFile(file) }); }
    catch (e) { showFormError(e.message); }
  }
  renderRefs();
}

// ---- 遮罩畫筆：在第一張參考圖上塗抹，輸出同尺寸黑底白色 PNG 當第二張參考圖 ----
function syncMaskPaint() { $("mask-paint").hidden = state.mode !== "image" || imageTask() !== "mask" || !state.refs.length; }

let maskSource = null;  // 畫布對應的原圖；換了原圖才清空筆跡
function openMaskPainter() {
  const img = $("mask-base"), canvas = $("mask-canvas"), url = state.refs[0].url;
  if (url === maskSource) { $("mask-dialog").showModal(); return; }
  img.onload = () => {
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    maskSource = url;
    $("mask-dialog").showModal();
  };
  img.src = url;
}

function bindMaskPainter() {
  const canvas = $("mask-canvas"), ctx = canvas.getContext("2d");
  let last = null;
  const point = (ev) => { const r = canvas.getBoundingClientRect(); return [(ev.clientX - r.left) * canvas.width / r.width, (ev.clientY - r.top) * canvas.height / r.height]; };
  const stroke = (to) => {
    const r = canvas.getBoundingClientRect();
    ctx.globalCompositeOperation = $("mask-erase").checked ? "destination-out" : "source-over";
    ctx.strokeStyle = "rgb(255 40 90)";  // 只影響預覽顏色 ctx.lineCap = ctx.lineJoin = "round";
    ctx.lineWidth = Number($("mask-brush").value) * canvas.width / r.width;  // 筆刷以螢幕像素計
    ctx.beginPath(); ctx.moveTo(...(last || to)); ctx.lineTo(...to); ctx.stroke();
    last = to;
  };
  canvas.addEventListener("pointerdown", (ev) => { canvas.setPointerCapture(ev.pointerId); last = null; stroke(point(ev)); });
  canvas.addEventListener("pointermove", (ev) => { if (canvas.hasPointerCapture(ev.pointerId)) stroke(point(ev)); });
  canvas.addEventListener("pointerup", () => { last = null; });
  $("mask-brush").addEventListener("input", () => { $("mask-brush-value").textContent = $("mask-brush").value; });
  $("mask-clear").addEventListener("click", () => ctx.clearRect(0, 0, canvas.width, canvas.height));
  $("mask-cancel").addEventListener("click", () => $("mask-dialog").close());
  $("mask-paint").addEventListener("click", openMaskPainter);
  $("mask-done").addEventListener("click", () => {
    const px = ctx.getImageData(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < px.data.length; i += 4) {  // 有塗到（alpha > 0）就是白，其餘黑
      const v = px.data[i + 3] > 0 ? 255 : 0;
      px.data[i] = px.data[i + 1] = px.data[i + 2] = v; px.data[i + 3] = 255;
    }
    const out = document.createElement("canvas");
    out.width = canvas.width; out.height = canvas.height;
    out.getContext("2d").putImageData(px, 0, 0);
    const mask = { name: "遮罩.png", url: out.toDataURL("image/png") };
    if (state.refs.length >= 2) state.refs[1] = mask; else state.refs.push(mask);
    $("mask-dialog").close();
    renderRefs();
  });
}

// ---- 模式與執行狀態 ----
function applyMode() {
  const image = state.mode === "image";
  $("mode-image").setAttribute("aria-pressed", String(image));
  $("mode-video").setAttribute("aria-pressed", String(!image));
  $("image-models").hidden = !image;
  $("negative-wrap").hidden = !image;
  $("cfg-wrap").hidden = !image;
  $("frames-wrap").hidden = image;
  $("shift-wrap").hidden = image;
  $("sched-wrap").hidden = !image || state.imageModel === TURBO_ID;
  state.fileSignature = "";
  renderModelFiles();
  updateTasks();
  renderSizes();
}

function setMode(mode) {
  if (state.mode === mode) return;
  const steps = Number($("steps").value);
  if (steps === stepDefault(state.mode)) $("steps").value = String(stepDefault(mode));
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
  $("form").querySelectorAll("input, textarea, select, button").forEach((node) => {
    if (node.id === "cancel" || node.id === "generate") return;
    node.disabled = busy;
  });
}

function updateRun(job) {
  const button = $("generate"), status = $("status"), meter = $("meter"), bar = $("meter-bar");
  status.classList.remove("warn");
  if (!job || isTerminal(job.status)) {
    $("cancel").hidden = true;
    setBusy(state.modelChanging);
    button.textContent = "生成";
    meter.hidden = true;
    bar.style.width = "0%";
    status.textContent = job ? (job.status === "succeeded" ? "完成" + (job.result && typeof job.result.seconds === "number" ? "，耗時 " + job.result.seconds.toFixed(1) + " 秒" : "") : STATUS_TEXT[job.status]) : "";
    return;
  }
  setBusy(true);
  $("cancel").hidden = false;
  const prog = job.progress || {}, step = prog.step || 0, total = prog.total || 0;
  const label = total ? step + "/" + total : "", stage = STAGE_TEXT[prog.stage] || "";
  meter.hidden = false;
  if (job.status === "queued") {
    const ahead = state.jobs.filter((j) => !isTerminal(j.status) && j.id < job.id).length;
    button.textContent = "排隊中";
    status.textContent = ahead ? "排隊中，前面還有 " + ahead + " 件" : "排隊中，等待顯示卡";
    status.classList.add("warn");
    bar.style.width = "0%";
    $("skeleton-text").textContent = status.textContent;
    return;
  }
  button.textContent = label ? "生成中 " + label : "生成中";
  status.textContent = stage && label ? stage + " " + label : stage || "生成中";
  $("skeleton-text").textContent = status.textContent;
  bar.style.width = total ? Math.min(100, (step / total) * 100) + "%" : "0%";
}

// ---- 結果檢視 ----
function clearViewer() {
  $("result-tools").hidden = true;
  $("result-img").hidden = true;
  const video = $("result-video");
  video.pause(); video.removeAttribute("src"); video.hidden = true; video.load();
  $("skeleton").hidden = true;
  $("viewer-error").hidden = true;
  $("viewer-error").textContent = "";
  $("detail").hidden = true;
}

function showJob(job) {
  if (!job) return;
  $("stage-title").textContent = job.type === "video" ? "影片結果" : "圖片結果";
  if (job.status === "queued" || job.status === "running") {
    if (state.shown.startsWith(job.id + ":run")) return;
    state.shown = job.id + ":run";
    clearViewer();
    $("empty").hidden = true;
    const p = job.parameters || {}, s = selectedSize();
    const w = p.width || s.w, h = p.height || s.h;
    if (w > 0 && h > 0) $("skeleton").style.aspectRatio = w + " / " + h;
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
    $("result-tools").hidden = false;
    ["use-seed", "open-new", "download"].forEach((id) => { $(id).hidden = true; });
    return;
  }
  const key = job.id + ":" + job.result.url;
  if (state.shown === key) return;
  state.shown = key;
  clearViewer();
  $("empty").hidden = true;
  const url = job.result.url;
  ["use-seed", "open-new", "download"].forEach((id) => { $(id).hidden = false; });
  $("download").href = url;
  $("download").download = job.id + (job.type === "video" ? ".mp4" : ".png");
  $("open-new").href = url;
  $("result-tools").hidden = false;
  if (job.type === "video") {
    const video = $("result-video");
    video.src = url; video.hidden = false;
  } else {
    const img = $("result-img");
    img.alt = job.prompt || "生成結果"; img.src = url; img.hidden = false;
  }
  renderDetail(job);
}

function viewJob(job) {
  state.viewId = job.id;
  state.shown = "";
  state.histSig = "";
  showJob(job);
  renderHistory();
  if (!isTerminal(job.status) && job.id === state.activeId) watch(job.id);
}

function onJob(job) {
  const i = state.jobs.findIndex((j) => j.id === job.id);
  if (i >= 0) state.jobs[i] = job;
  if (job.id === state.viewId) showJob(job);
  if (job.id === state.activeId) {
    updateRun(job);
    if (isTerminal(job.status)) state.activeId = null;
  }
}

async function cancelJob(id) {
  try { onJob(await api("/v1/jobs/" + encodeURIComponent(id), { method: "DELETE" })); refreshMeta(); }
  catch (e) { showFormError(e.message); }
}

// ---- 進度追蹤 ----
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
  if (state.apiKey) return pollWatch(id);  // EventSource 不能帶 Authorization 標頭
  const es = new EventSource("/v1/jobs/" + encodeURIComponent(id) + "/events");
  state.source = es;
  es.onmessage = (ev) => {
    let job = null;
    try { job = JSON.parse(ev.data); } catch (e) { return; }
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

// ---- 佇列與近期 ----
function renderHistory() {
  const sig = state.viewId + "#" + state.jobs.map((j) => j.id + j.status + (j.progress ? j.progress.step : 0)).join("|");
  if (sig === state.histSig) return;
  state.histSig = sig;
  renderLibrary();
  const box = $("history");
  if (!state.jobs.length) {
    const li = document.createElement("li");
    li.className = "empty-line";
    li.textContent = "生成的作品會依序排在這裡。";
    box.replaceChildren(li);
    return;
  }
  box.replaceChildren(...state.jobs.slice(0, 30).map((job) => {
    const li = document.createElement("li"), button = document.createElement("button"), text = document.createElement("span");
    li.className = "hist" + (job.id === state.viewId ? " current" : "") + " s-" + job.status;
    button.type = "button";
    button.title = job.prompt || job.id;
    button.append(jobThumb(job));
    const prompt = document.createElement("span"), meta = document.createElement("span");
    text.className = "hist-text";
    prompt.className = "hist-prompt"; prompt.textContent = job.prompt || job.id;
    meta.className = "hist-meta num";
    const p = job.progress || {};
    meta.textContent = STATUS_TEXT[job.status] + (job.status === "running" && p.total ? " " + p.step + "/" + p.total : "") +
      (job.result && typeof job.result.seconds === "number" ? " · " + job.result.seconds.toFixed(1) + " 秒" : "");
    text.append(prompt, meta);
    button.append(text);
    button.addEventListener("click", () => viewJob(job));
    li.append(button);
    if (!isTerminal(job.status)) {
      const cancel = document.createElement("button");
      cancel.type = "button"; cancel.className = "ghost small danger-text"; cancel.textContent = "取消";
      cancel.setAttribute("aria-label", "取消 " + (job.prompt || job.id));
      cancel.addEventListener("click", () => cancelJob(job.id));
      li.append(cancel);
    }
    return li;
  }));
}

// ---- 輪詢 ----
function setConnection(ok, text) {
  const c = $("connection");
  c.className = "pill " + (ok ? "connected" : "warn");
  c.lastElementChild.textContent = text;
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
    setConnection(true, "已連線");
    $("api-state").textContent = "回應 " + Math.round(performance.now() - start) + " ms";
    state.jobs = (jobs && jobs.data) || [];
    const running = state.jobs.find((j) => !isTerminal(j.status));
    if (!state.activeId && running) { state.activeId = running.id; if (!state.viewId) state.viewId = running.id; watch(running.id); }
    updateModelActions();
    renderHistory();
    const viewed = state.jobs.find((j) => j.id === state.viewId);
    if (viewed) showJob(viewed);
    if (state.activeId) updateRun(state.jobs.find((j) => j.id === state.activeId));
  } catch (e) {
    if (e.status === 401) { state.authRequired = true; showView("api-panel"); }
    setConnection(false, e.status === 401 ? "需要金鑰" : "無法連線");
    $("api-state").textContent = "無法連線";
    $("monitor-error").textContent = "資料可能已過期：" + e.message;
    $("api-error").textContent = e.status === 401 ? "服務需要 API 金鑰，請輸入後按「測試連線」。" : e.message;
  } finally { state.refreshing = false; }
}

async function loadLoras() {
  try {
    const data = await api("/v1/loras");
    state.loraRows = (data && data.data) || [];
    renderLoras(state.loraRows);
  } catch (e) {
    const p = document.createElement("p");
    p.className = "error";
    p.textContent = e.message;
    $("loras").replaceChildren(p);
  }
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
    const job = await api("/v1/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), timeout: 120000 });
    state.activeId = job.id;
    state.jobs = [job, ...state.jobs.filter((j) => j.id !== job.id)];
    viewJob(job);
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
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); $("form").requestSubmit(); }
  });
  $("randomize").addEventListener("click", () => { $("seed").value = String(Math.floor(Math.random() * 2147483648)); updateAdvancedSummary(); updateRequestPreview(); });
  $("cancel").addEventListener("click", () => { if (state.activeId) cancelJob(state.activeId); });
  $("ref-input").addEventListener("change", (ev) => { addFiles(ev.target.files).then(() => { ev.target.value = ""; }); });
  $("vref-input").addEventListener("change", (ev) => { addVrefs(ev.target.files).then(() => { ev.target.value = ""; }); });
  bindDrop($("vdrop"), addVrefs);
  bindDrop($("drop"), addFiles);
  document.addEventListener("paste", (ev) => {
    if ($("studio").hidden || $("form").querySelector(":disabled#prompt")) return;
    const files = Array.from(ev.clipboardData?.files || []).filter((f) => f.type.startsWith("image/") || f.type.startsWith("video/") || f.type.startsWith("audio/"));
    if (!files.length) return;
    ev.preventDefault();
    if (state.mode === "image") addFiles(files);
    else if (videoTask() === "reference") addVrefs(files);
    else setFrame(state.frames.first ? "last" : "first", files[0]);
  });
}

initEmbedding();
applyMode();
bind();
bindControls();
bindViews();
refreshMeta();
loadLoras();
setInterval(refreshMeta, 3000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshMeta(); });
