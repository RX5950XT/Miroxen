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
];
const VIDEO_SIZES = [
  ["864×480", 864, 480],
  ["1344×768（官方）", 1344, 768],
  ["480×864", 480, 864],
  ["自動", 0, 0],
];
const STEP_DEFAULT = { image: 40, video: 30 };
const STATUS_TEXT = {
  queued: "排隊中",
  running: "生成中",
  succeeded: "完成",
  failed: "失敗",
  cancelled: "已取消",
};
const STAGE_TEXT = { encode: "編碼", denoise: "去噪", decode: "解碼" };

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
};

function sizes() {
  return state.mode === "video" ? VIDEO_SIZES : IMAGE_SIZES;
}

function selectedSize() {
  const list = sizes();
  const item = list[Math.min(state.size, list.length - 1)];
  return { label: item[0], w: item[1], h: item[2] };
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
  const res = await fetch(path, Object.assign({ cache: "no-store" }, opts));
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
    throw new Error(msg);
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

function renderSizes() {
  const box = $("sizes");
  box.replaceChildren();
  sizes().forEach((item, i) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item[0];
    button.setAttribute("aria-pressed", i === state.size ? "true" : "false");
    button.addEventListener("click", () => {
      state.size = i;
      renderSizes();
      const s = selectedSize();
      if (s.w) $("skeleton").style.aspectRatio = s.w + " / " + s.h;
    });
    box.append(button);
  });
  const s = selectedSize(), help = $("size-help");
  help.hidden = s.w !== 0;
  help.textContent = state.mode === "image"
    ? "依最後一張參考圖的比例，約 1024×1024 像素。"
    : "有首幀時依首幀比例、否則 16:9，短邊 768（官方畫布，較慢）。";
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
    state.vrefs.push({ kind, name: file.name, url: await readFile(file) });
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
  renderSizes();
  const s = selectedSize();
  $("skeleton").style.aspectRatio = s.w + " / " + s.h;
}

function setMode(mode) {
  if (state.mode === mode) return;
  const steps = Number($("steps").value);
  if (steps === STEP_DEFAULT[state.mode]) $("steps").value = String(STEP_DEFAULT[mode]);
  state.mode = mode;
  state.size = 0;
  applyMode();
}

function setBusy(busy) {
  $("cancel").hidden = !busy;
  const fields = $("form").querySelectorAll("input, textarea, button");
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
    setBusy(false);
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
  if (typeof sec === "number") text += " · " + sec.toFixed(1) + " 秒";
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
  if (!state.source) return;
  state.source.close();
  state.source = null;
}

function watch(id) {
  closeSource();
  state.settled = false;
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
  es.onerror = async () => {
    es.close();
    if (state.source === es) state.source = null;
    if (state.settled) return;
    try {
      onJob(await api("/v1/jobs/" + encodeURIComponent(id)));
    } catch (e) {
      showFormError(e.message);
    }
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
  state.models = (data && data.data) || [];
  const box = $("models");
  box.replaceChildren();
  state.models.forEach((m) => {
    const el = document.createElement("span");
    el.className = "model";
    const dot = document.createElement("span");
    dot.className = "dot" + (m.loaded ? " on" : "");
    const name = document.createElement("span");
    name.textContent = m.id;
    el.append(dot, name);
    box.append(el);
  });
}

function renderSystem(data) {
  $("queue").textContent = "佇列 " + ((data && data.queue_length) || 0);
  const box = $("gpus");
  box.replaceChildren();
  (data.gpus || []).forEach((g) => {
    const row = document.createElement("div");
    row.className = "gpu";
    const label = document.createElement("span");
    label.className = "gpu-label";
    const name = g.name || "GPU " + g.index;
    const used = g.total > 0 ? Math.max(0, Math.min(1, 1 - g.free / g.total)) : 0;
    label.textContent = name;
    label.title = name + " · 已用 " + Math.round(used * 100) + "%";
    const bar = document.createElement("div");
    bar.className = "bar";
    const span = document.createElement("span");
    span.style.width = (used * 100).toFixed(1) + "%";
    bar.append(span);
    row.append(label, bar);
    box.append(row);
  });
}

async function refreshMeta() {
  try {
    const [models, sys, jobs] = await Promise.all([api("/v1/models"), api("/v1/system"), api("/v1/jobs")]);
    renderModels(models);
    renderSystem(sys);
    state.jobs = (jobs && jobs.data) || [];
    renderHistory();
    if (state.viewId && state.viewId !== state.activeId) {
      const found = state.jobs.find((j) => j.id === state.viewId);
      if (found) showJob(found);
    }
  } catch (e) {
    $("queue").textContent = "離線";
  }
}

function renderLoras(rows) {
  const box = $("loras");
  box.replaceChildren();
  if (!rows.length) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "目錄裡沒有 .safetensors";
    box.append(p);
    return;
  }
  rows.forEach((row) => {
    const line = document.createElement("div");
    line.className = "lora";
    const lab = document.createElement("label");
    const boxIn = document.createElement("input");
    boxIn.type = "checkbox";
    boxIn.value = row.name;
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = row.name;
    name.title = row.name;
    lab.append(boxIn, name);
    const range = document.createElement("input");
    range.type = "range";
    range.min = "0";
    range.max = "2";
    range.step = "0.05";
    range.value = "1";
    const scale = document.createElement("span");
    scale.className = "scale";
    scale.textContent = "1.00";
    range.addEventListener("input", () => {
      scale.textContent = Number(range.value).toFixed(2);
    });
    line.append(lab, range, scale);
    box.append(line);
  });
}

async function loadLoras() {
  try {
    const data = await api("/v1/loras");
    renderLoras((data && data.data) || []);
  } catch (e) {
    const box = $("loras");
    box.replaceChildren();
    const p = document.createElement("p");
    p.className = "error";
    p.textContent = e.message;
    box.append(p);
  }
}

function selectedLoras() {
  const out = [];
  document.querySelectorAll("#loras .lora").forEach((row) => {
    const box = row.querySelector('input[type="checkbox"]');
    if (!box || !box.checked) return;
    out.push({ name: box.value, scale: Number(row.querySelector('input[type="range"]').value) });
  });
  return out;
}

function renderRefs() {
  const list = $("ref-list");
  list.replaceChildren();
  state.refs.forEach((ref, index) => {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.textContent = ref.name;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "移除";
    remove.addEventListener("click", () => {
      state.refs.splice(index, 1);
      renderRefs();
    });
    li.append(name, remove);
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
    state.refs.push({ name: file.name, url: await readFile(file) });
  }
  renderRefs();
}

function integerIn(raw, min, max) {
  if (!/^-?\d+$/.test(String(raw).trim())) return null;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < min || n > max) return null;
  return n;
}

async function onSubmit(ev) {
  ev.preventDefault();
  if (state.activeId) return;
  showFormError("");
  const prompt = $("prompt").value.trim();
  if (!prompt) return showFormError("請先寫提示詞");
  if ([...prompt].length > 8000) return showFormError("提示詞超過 8000 字");
  const steps = integerIn($("steps").value, 1, 200);
  if (steps == null) return showFormError("步數必須是 1 到 200 的整數");
  const seedText = $("seed").value.trim();
  const seed = seedText === "" ? -1 : integerIn(seedText, -1, 9223372036854775807);
  if (seed == null) return showFormError("種子必須是整數，-1 表示隨機");
  const kind = state.mode;
  const model = modelId(kind);
  if (!model) return showFormError("沒有可用的模型");
  const size = selectedSize();
  const body = { type: kind, model, prompt, size: size.w ? size.w + "x" + size.h : "auto", steps, seed };
  try {
    if (kind === "image") {
      const cfg = Number($("cfg").value);
      if (!Number.isFinite(cfg) || cfg < 0 || cfg > 20) return showFormError("CFG 必須介於 0 與 20");
      body.cfg_scale = cfg;
      const negative = $("negative").value.trim();
      if (negative) body.negative_prompt = negative;
      if (state.refs.length) body.images = state.refs.map((ref) => ref.url);
      if (!size.w && !state.refs.length) return showFormError("「依參考圖比例」需要先加參考圖");
      const sigText = $("sigmas").value.trim();
      if (sigText) {
        const sig = sigText.split(/[,，\s]+/).filter(Boolean).map(Number);
        if (!sig.every((v) => Number.isFinite(v) && v > 0 && v <= 1)) return showFormError("sigma 節點必須是 0 到 1 之間的數字");
        body.sigmas = sig;
      }
      if (!$("shift-terminal").checked) body.shift_terminal = false;
    } else {
      const frames = integerIn($("frames").value, 5, 361);
      if (frames == null) return showFormError("幀數必須是 5 到 361 的整數");
      body.num_frames = frames;
      const first = $("first-frame").files[0];
      const last = $("last-frame").files[0];
      if (first) body.first_frame = await readFile(first);
      if (last) body.last_frame = await readFile(last);
      const shift = Number($("video-shift").value);
      if (!Number.isFinite(shift) || shift < 0.5 || shift > 50) return showFormError("影片 shift 必須介於 0.5 與 50");
      if (shift !== 12) body.video_shift = shift;
      if (state.vrefs.length) {
        if (first || last) return showFormError("參考生成不能和首尾幀同時使用");
        if (state.vrefs.every((r) => r.kind === "audio")) return showFormError("音訊參考需要搭配圖片或影片");
        body.references = state.vrefs.map((r) => ({ type: r.kind, data: r.url }));
      }
    }
    const loras = selectedLoras();
    if (loras.length) body.loras = loras;
    const compute = $("compute").value;
    if (compute) body.compute = compute;
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
refreshMeta();
loadLoras();
setInterval(refreshMeta, 3000);
