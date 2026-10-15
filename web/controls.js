"use strict";

const FILE_LABEL = { dit: "DiT 權重", dit_ref: "參考生成 DiT", text_encoder: "文字編碼器", vae: "圖片 VAE", vision: "視覺編碼器", video_vae: "影片 VAE", audio_vae: "音訊 VAE" };
const TASK_HELP = {
  generate: "加入參考圖即可編輯；不加參考圖則生成新圖。",
  transparent: "生成含透明度的 PNG，背景以棋盤格預覽。",
  extract: "至少加入一張原圖，提示詞指定要保留的主體。",
  mask: "加入原圖與遮罩圖，提示詞指定白色區域要改什麼。",
  first_last: "只寫提示詞就能生成；也可加首幀、尾幀控制開頭與結尾畫面。",
  reference: "依參考圖片、影片、音訊生成；提示詞用 <Picture 1> 這類標籤指名素材。",
};
// 圖片任務送出時包在使用者提示詞外的固定文字；重用設定時用同一張表拆回來。
const TASK_WRAP = {
  transparent: ["This is an RGBA image with transparency. ", " The background is transparent."],
  extract: ["This is an RGBA image with transparency. Extract the subject from the input photo. ", " The background is transparent."],
  mask: ["Picture 2 is a mask. In Picture 1, modify only inside the white area of the mask. ", ""],
};

function radioValue(name) { return (document.querySelector('input[name="' + name + '"]:checked') || {}).value; }
function setRadio(name, value) { const el = document.querySelector('input[name="' + name + '"][value="' + value + '"]'); if (el) el.checked = true; }
function imageTask() { return radioValue("image-task") || "generate"; }
function videoTask() { return radioValue("video-task") || "first_last"; }

function alignedFrames(frames) { return Math.ceil((Math.max(22, frames) - 5) / 17) * 17 + 5; }

function updateDuration() {
  const f = Number($("frames").value), n = alignedFrames(f);
  $("duration-help").textContent = Number.isFinite(f) ? "24 fps · 實際 " + n + " 幀 / " + (n / 24).toFixed(2) + " 秒。最短 22 幀，之後以 17 幀一組對齊。" : "請填入幀數。";
}

function selectedSize() {
  const item = sizes()[Math.min(state.size, sizes().length - 1)];
  if (item[1] === -1) return { label: item[0], w: Number($("width").value), h: Number($("height").value) };
  return { label: item[0], w: item[1], h: item[2] };
}

function renderSizes() {
  const box = $("sizes");
  box.replaceChildren();
  sizes().forEach((item, i) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = item[0];
    button.setAttribute("aria-pressed", String(i === state.size));
    button.addEventListener("click", () => { state.size = i; renderSizes(); updateRequestPreview(); });
    box.append(button);
  });
  const item = sizes()[state.size], s = selectedSize();
  $("custom-size").hidden = item[1] !== -1;
  $("size-help").textContent = item[1] === 0
    ? (state.mode === "image" ? "依最後一張參考圖比例，約 1024² 像素。" : "依首幀比例；沒有首幀時用官方 16:9 畫布。")
    : item[1] === -1 ? "寬高須為 32 的倍數；720×1280 可改用 736×1312，輸出不會自動裁切。" : s.w + " × " + s.h + " px";
  if (s.w > 0 && s.h > 0) $("skeleton").style.aspectRatio = s.w + " / " + s.h;
}

function updateTasks() {
  const image = state.mode === "image", reference = videoTask() === "reference";
  $("image-task").hidden = !image;
  $("video-task").hidden = image;
  $("refs-wrap").hidden = !image;
  $("video-inputs").hidden = image;
  $("frame-files").hidden = image || reference;
  $("vrefs-wrap").hidden = image || !reference;
  $("mask-help").hidden = imageTask() !== "mask";
  syncMaskPaint();
  $("steps").min = image ? "1" : "2";
  $("task-help").textContent = TASK_HELP[image ? imageTask() : videoTask()];
  $("steps-help").textContent = !image ? "實際計算 " + Math.max(0, Number($("steps").value) - 1) + " 次；8 步 LoRA 填 9。"
    : state.imageModel === TURBO_ID ? "Turbo 固定 8 步。" : "自訂 sigma 優先於此步數。";
  $("steps").readOnly = image && state.imageModel === TURBO_ID;
  updateAdvancedSummary();
  renderLoras(state.loraRows);
  updateDuration();
}

function visibleLoras() {
  const base = (id) => id.startsWith("qwen-image-2.1") ? "qwen-image-2.1" : id;  // Turbo 與標準共用 LoRA
  const model = base(modelId(state.mode)), task = videoTask();
  return state.loraRows.filter((row) => (row.model === "unknown" || row.model === model) &&
    (state.mode === "image" || row.task === "any" || row.task === task));
}

function selectedLoras() {
  return visibleLoras().filter((row) => state.loraSelections[row.name]).map((row) => ({ name: row.name, scale: row.purpose === "style" ? state.loraSelections[row.name].scale : 1 }));
}

function applyLoraPreset(row) {
  const preset = row.preset;
  if (!preset) return;
  $("steps").value = String(preset.steps);
  if (state.mode === "image") {
    $("cfg").value = String(preset.cfg_scale || 1);
    $("sigmas").value = (preset.sigmas || []).join(", ");
    $("shift-terminal").checked = preset.shift_terminal !== false;
  } else $("video-shift").value = String(preset.video_shift);
  $("lora-help").textContent = "已套用作者建議設定：" + (state.mode === "video" ? preset.steps - 1 : preset.steps) + " 次計算。加速 LoRA 強度固定為 1。";
  updateTasks();
}

function clearAcceleration() {
  state.loraRows.filter((row) => row.purpose === "acceleration").forEach((row) => delete state.loraSelections[row.name]);
  $("steps").value = stepDefault(state.mode);
  $("sigmas").value = "";
  $("shift-terminal").checked = true;
  $("video-shift").value = "12";
  $("lora-help").textContent = "";
}

function renderLoras(rows) {
  const box = $("loras");
  box.replaceChildren();
  const available = visibleLoras();
  if (!available.length) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = rows.length ? "目前任務沒有相容的 LoRA。" : "尚無 LoRA。可從下方安裝社群 LoRA。";
    box.append(p);
  }
  available.forEach((row) => {
    const line = document.createElement("div"), lab = document.createElement("label"), check = document.createElement("input"), text = document.createElement("span");
    line.className = "lora";
    check.type = "checkbox";
    check.checked = Boolean(state.loraSelections[row.name]);
    text.className = "name";
    text.textContent = row.name.replace(/\.safetensors$/i, "");
    text.title = row.name;
    const tag = document.createElement("small");
    tag.textContent = row.purpose === "acceleration" ? "加速 · 強度固定 1" : row.purpose === "style" ? "畫風" : "用途待確認";
    lab.append(check, text, tag);
    const range = document.createElement("input"), scale = document.createElement("span");
    range.type = "range"; range.min = "0"; range.max = "2"; range.step = "0.05";
    range.value = String(state.loraSelections[row.name]?.scale ?? 1);
    range.setAttribute("aria-label", row.name + " 強度");
    scale.className = "scale num";
    scale.textContent = Number(range.value).toFixed(2);
    range.addEventListener("input", () => {
      scale.textContent = Number(range.value).toFixed(2);
      if (check.checked) state.loraSelections[row.name] = { scale: Number(range.value) };
      updateRequestPreview();
    });
    check.addEventListener("change", () => {
      if (check.checked) {
        if (row.purpose === "acceleration") clearAcceleration();
        state.loraSelections[row.name] = { scale: row.purpose === "style" ? Number(range.value) : 1 };
        applyLoraPreset(row);
      } else {
        delete state.loraSelections[row.name];
        if (row.purpose === "acceleration") { clearAcceleration(); $("lora-help").textContent = "已回復原版排程。"; }
      }
      renderLoras(rows); updateTasks(); updateRequestPreview();
    });
    line.append(lab);
    if (row.purpose === "style") line.append(range, scale);
    box.append(line);
  });
  const n = selectedLoras().length;
  $("lora-summary").textContent = n ? "已選 " + n + " 個" : available.length ? available.length + " 個可用" : "";
  if (state.activeId || state.modelChanging) box.querySelectorAll("input").forEach((el) => { el.disabled = true; });
}

function renderModelFiles() {
  const model = state.models.find((m) => m.id === modelId(state.mode));
  if (!model) return;
  const signature = JSON.stringify([model.id, model.files, model.defaults]);
  if (signature === state.fileSignature) return;
  state.fileSignature = signature;
  const box = $("model-files");
  box.replaceChildren();
  const entries = Object.entries(model.files || {});
  $("files-wrap").hidden = !entries.length;
  entries.forEach(([role, choices]) => {
    const label = document.createElement("label"), select = document.createElement("select");
    label.className = "field";
    label.append(document.createTextNode(FILE_LABEL[role] || role));
    select.dataset.role = role;
    select.append(new Option("預設 · " + model.defaults[role], ""), ...choices.map((file) => new Option(file, file)));
    select.value = fileSelection(model.id)[role] || "";
    select.addEventListener("change", () => { fileSelection(model.id)[role] = select.value; updateRequestPreview(); });
    label.append(select); box.append(label);
  });
}

function requestSettings() {
  const kind = state.mode, s = selectedSize(), task = imageTask();
  let prompt = $("prompt").value.trim();
  if (kind === "image" && TASK_WRAP[task]) prompt = TASK_WRAP[task][0] + prompt + TASK_WRAP[task][1];
  const body = { type: kind, model: modelId(kind), prompt, size: sizes()[state.size][1] === 0 ? "auto" : s.w + "x" + s.h, steps: Number($("steps").value), seed: Number($("seed").value.trim() || -1) };
  if (kind === "image") {
    body.cfg_scale = Number($("cfg").value);
    if ($("negative").value.trim()) body.negative_prompt = $("negative").value.trim();
    if (body.model !== TURBO_ID) {  // Turbo 用固定 8 步排程
      body.shift_terminal = $("shift-terminal").checked;
      if ($("sigmas").value.trim()) body.sigmas = $("sigmas").value.split(/[,，\s]+/).filter(Boolean).map(Number);
    }
  } else {
    body.num_frames = Number($("frames").value);
    body.video_shift = Number($("video-shift").value);
    if (videoTask() === "reference") { body.ref_image_short_edge = Number($("ref-edge").value); body.ref_native_canvas = $("ref-native").checked; }
  }
  body.loras = selectedLoras();
  if ($("compute").value) body.compute = $("compute").value;
  const files = Object.fromEntries(Object.entries(fileSelection(body.model)).filter(([, name]) => name));
  if (Object.keys(files).length) body.model_files = files;
  return body;
}

async function buildRequest() {
  if (!$("prompt").value.trim()) throw new Error("請先寫提示詞");
  const body = requestSettings(), image = body.type === "image", s = selectedSize();
  const limit = image ? 2752 : 1920;
  if (body.size !== "auto" && (![s.w, s.h].every((v) => Number.isInteger(v) && v >= 256 && v <= limit && v % 32 === 0))) throw new Error("寬高須為 256～" + limit + " 之間的 32 倍數");
  const minSteps = image ? 1 : 2;
  if (!Number.isInteger(body.steps) || body.steps < minSteps || body.steps > 200) throw new Error("步數須為 " + minSteps + "～200 的整數");
  if (!Number.isSafeInteger(body.seed) || body.seed < -1) throw new Error("種子須為 -1～9007199254740991 的整數");
  if ([...body.prompt].length > 8000) throw new Error("包含任務提示後超過 8000 字，請縮短提示詞");
  if (image) {
    if (!Number.isFinite(body.cfg_scale) || body.cfg_scale < 0 || body.cfg_scale > 20) throw new Error("CFG 須為 0～20");
    if (body.sigmas && (body.sigmas.length > 200 || !body.sigmas.every((v, i, a) => Number.isFinite(v) && v > 0 && v <= 1 && (i === 0 ? v === 1 : v < a[i - 1])))) throw new Error("sigma 須從 1 開始、嚴格遞減，最多 200 個");
    if (state.refs.length) body.images = state.refs.map((ref) => ref.url);
    if (s.w === 0 && !state.refs.length) throw new Error("依參考圖比例需要先加參考圖");
    if (imageTask() === "extract" && !state.refs.length) throw new Error("去背需要加入原圖");
    if (imageTask() === "mask" && state.refs.length < 2) throw new Error("局部修改需要原圖與遮罩圖");
  } else {
    if (!Number.isInteger(body.num_frames) || body.num_frames < 5 || body.num_frames > 361) throw new Error("目標幀數須為 5～361");
    if (!Number.isFinite(body.video_shift) || body.video_shift < 0.5 || body.video_shift > 50) throw new Error("影片 shift 須為 0.5～50");
    if (videoTask() === "reference") {
      if (!state.vrefs.length || state.vrefs.every((r) => r.kind === "audio")) throw new Error("參考生成需要至少一張圖片或一段影片");
      if (!Number.isInteger(body.ref_image_short_edge) || body.ref_image_short_edge < 256 || body.ref_image_short_edge > 4096) throw new Error("參考圖短邊須為 256～4096 的整數");
      body.references = state.vrefs.map((r) => ({ type: r.kind, data: r.url }));
    } else {
      if (state.frames.first) body.first_frame = state.frames.first.url;
      if (state.frames.last) body.last_frame = state.frames.last.url;
    }
  }
  return body;
}

function previewBody() {
  const body = requestSettings();
  if (state.mode === "image" && state.refs.length) body.images = state.refs.map((r) => "[base64: " + r.name + "]");
  if (state.mode === "video") {
    if (videoTask() === "reference") body.references = state.vrefs.map((r) => ({ type: r.kind, data: "[base64: " + r.name + "]" }));
    else ["first", "last"].forEach((key) => { if (state.frames[key]) body[key + "_frame"] = "[base64: " + state.frames[key].name + "]"; });
  }
  return body;
}

function updateRequestPreview() {
  $("request-preview").textContent = JSON.stringify(previewBody(), null, 2);
  $("prompt-count").textContent = [...$("prompt").value].length + " / 8000";
}

function updateAdvancedSummary() {
  const seed = $("seed").value.trim();
  const parts = [$("steps").value + " 步"];
  if (state.mode === "image") parts.push("CFG " + $("cfg").value);
  parts.push(seed === "-1" || !seed ? "隨機種子" : "種子 " + seed);
  if ($("compute").value) parts.push($("compute").value.toUpperCase());
  $("adv-summary").textContent = parts.join(" · ");
}

function applyImageModel(id) {
  const prev = state.imageModel, turbo = id === TURBO_ID;
  state.imageModel = id;
  setRadio("image-model", id);
  if (Number($("steps").value) === (prev === TURBO_ID ? 8 : 40)) $("steps").value = String(stepDefault("image"));
  if (turbo) { $("steps").value = "8"; $("cfg").value = "1"; $("sigmas").value = ""; }
  $("sched-wrap").hidden = turbo || state.mode !== "image";
  state.fileSignature = "";
  renderModelFiles(); updateTasks(); updateRequestPreview();
}

// ---- 首尾幀 ----
async function setFrame(key, file) {
  const slot = $(key + "-slot"), img = slot.querySelector("img");
  if (!file) { delete state.frames[key]; img.hidden = true; img.removeAttribute("src"); slot.classList.remove("filled"); updateRequestPreview(); return; }
  if (!/^image\/(png|jpeg)$/.test(file.type) && !/\.(png|jpe?g)$/i.test(file.name)) { showFormError("首尾幀只接受 PNG 或 JPEG"); return; }
  try {
    const url = await readFile(file);
    state.frames[key] = { name: file.name, url };
    img.src = url; img.alt = (key === "first" ? "首幀 " : "尾幀 ") + file.name; img.hidden = false; slot.classList.add("filled");
  } catch (e) { showFormError(e.message); }
  updateRequestPreview();
}

// ---- 社群 LoRA 安裝 ----
const INSTALL_STATE = { queued: "排隊中", downloading: "下載中", installing: "驗證中", done: "已安裝", failed: "失敗" };

function installRow(text, detail, kind) {
  const li = document.createElement("li"), title = document.createElement("span"), meta = document.createElement("span");
  li.className = "install " + (kind || "");
  title.className = "install-name"; title.textContent = text;
  meta.className = "install-meta num"; meta.textContent = detail;
  li.append(title, meta);
  return li;
}

function mb(n) { return (n / 1048576).toFixed(n > 1048576 * 100 ? 0 : 1) + " MB"; }

async function refreshInstalls() {
  let data;
  try { data = await api("/v1/loras/installs"); } catch (e) { $("lora-help").textContent = e.message; return; }
  const rows = (data && data.data) || [];
  const box = $("lora-installs");
  box.querySelectorAll(".install.remote").forEach((el) => el.remove());
  rows.slice(0, 5).forEach((r) => {
    const name = r.name || decodeURIComponent(r.url.split("/").pop().split("?")[0]) || r.url;
    const detail = INSTALL_STATE[r.state] + (r.bytes ? " · " + mb(r.bytes) : "") + (r.model ? " · " + (r.model === "minimax-h3" ? "影片" : "圖片") : "") + (r.error ? " · " + r.error : "");
    box.append(installRow(name, detail, "remote " + r.state));
  });
  const active = rows.some((r) => r.state === "queued" || r.state === "downloading" || r.state === "installing");
  if (rows.some((r) => r.state === "done" && !state.loraRows.some((l) => l.name === r.name))) loadLoras();
  if (active && !state.installPoll) state.installPoll = setInterval(refreshInstalls, 1500);
  if (!active && state.installPoll) { clearInterval(state.installPoll); state.installPoll = null; }
}

async function installFromUrl() {
  const url = $("lora-url").value.trim();
  if (!url) { $("lora-help").textContent = "請先貼上網址。"; return; }
  $("lora-url-install").disabled = true;
  try {
    await api("/v1/loras/install", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) });
    $("lora-url").value = "";
    $("lora-help").textContent = "開始下載，完成後會自動出現在清單。";
    await refreshInstalls();
  } catch (e) { $("lora-help").textContent = e.message; }
  finally { $("lora-url-install").disabled = false; }
}

function uploadLora(file) {
  if (!file) return;
  if (!/\.safetensors$/i.test(file.name)) { $("lora-help").textContent = "只接受 .safetensors 檔案。"; return; }
  const row = installRow(file.name, "上傳 0%", "local"), meta = row.querySelector(".install-meta");
  $("lora-installs").prepend(row);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/v1/loras/upload?name=" + encodeURIComponent(file.name));
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  if (state.apiKey) xhr.setRequestHeader("Authorization", "Bearer " + state.apiKey);
  xhr.upload.onprogress = (ev) => { if (ev.lengthComputable) meta.textContent = "上傳 " + Math.round(100 * ev.loaded / ev.total) + "% · " + mb(ev.loaded); };
  xhr.onload = () => {
    let data = null;
    try { data = JSON.parse(xhr.responseText); } catch (e) { data = null; }
    if (xhr.status === 201 && data) {
      meta.textContent = "已安裝 · " + (data.model === "minimax-h3" ? "影片" : "圖片") + " · " + mb(data.size);
      row.classList.add("done");
      loadLoras();
    } else {
      meta.textContent = "失敗 · " + ((data && data.error && data.error.message) || "伺服器回應 " + xhr.status);
      row.classList.add("failed");
    }
  };
  xhr.onerror = () => { meta.textContent = "失敗 · 連線中斷"; row.classList.add("failed"); };
  xhr.send(file);
}

function bindDrop(el, onFiles) {
  ["dragenter", "dragover"].forEach((name) => el.addEventListener(name, (ev) => { ev.preventDefault(); el.classList.add("hot"); }));
  el.addEventListener("dragleave", () => el.classList.remove("hot"));
  el.addEventListener("drop", (ev) => { ev.preventDefault(); el.classList.remove("hot"); onFiles(ev.dataTransfer.files); });
}

function bindControls() {
  document.querySelectorAll('input[name="image-model"]').forEach((radio) => radio.addEventListener("change", () => { if (radio.checked) applyImageModel(radio.value); }));
  document.querySelectorAll('input[name="image-task"]').forEach((radio) => radio.addEventListener("change", () => { updateTasks(); updateRequestPreview(); }));
  document.querySelectorAll('input[name="video-task"]').forEach((radio) => radio.addEventListener("change", () => { clearAcceleration(); updateTasks(); updateRequestPreview(); }));
  $("lora-url-install").addEventListener("click", installFromUrl);
  $("lora-url").addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); installFromUrl(); } });
  $("lora-file").addEventListener("change", (ev) => { uploadLora(ev.target.files[0]); ev.target.value = ""; });
  bindDrop($("lora-drop"), (files) => uploadLora(files[0]));
  $("lora-install").addEventListener("toggle", () => { if ($("lora-install").open) refreshInstalls(); });
  ["first", "last"].forEach((key) => {
    $(key + "-frame").addEventListener("change", (ev) => { setFrame(key, ev.target.files[0]); ev.target.value = ""; });
    bindDrop($(key + "-slot"), (files) => setFrame(key, files[0]));
  });
  $("clear-frames").addEventListener("click", () => { setFrame("first", null); setFrame("last", null); });
  $("duration").addEventListener("input", () => { const n = Number($("duration").value); if (Number.isFinite(n)) $("frames").value = Math.max(5, Math.ceil(n * 24)); updateDuration(); });
  $("frames").addEventListener("input", () => { $("duration").value = (Number($("frames").value) / 24).toFixed(2); updateDuration(); });
  $("form").addEventListener("input", () => { updateRequestPreview(); updateAdvancedSummary(); });
  $("seed-reset").addEventListener("click", () => { $("seed").value = "-1"; updateAdvancedSummary(); updateRequestPreview(); });
  $("reload-loras").addEventListener("click", loadLoras);
  bindMaskPainter();
  updateTasks(); updateRequestPreview(); updateAdvancedSummary();
}
