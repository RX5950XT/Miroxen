"use strict";

const FILE_LABEL = { dit: "DiT 權重", dit_ref: "參考生成 DiT", text_encoder: "文字編碼器", vae: "圖片 VAE", vision: "視覺編碼器", video_vae: "影片 VAE", audio_vae: "音訊 VAE" };
const IMAGE_TASK = {
  generate: "加入參考圖即可編輯；不加參考圖則生成新圖。",
  transparent: "生成含透明度的 PNG，背景以棋盤格預覽。",
  extract: "至少加入一張原圖，提示詞指定要保留的主體。",
  mask: "加入原圖與遮罩圖，提示詞指定白色區域要改什麼。",
};

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
    : "寬高須為 32 的倍數。" + (item[1] === -1 ? "720×1280 可改用 736×1312；輸出不會自動裁切。" : s.w + " × " + s.h + " px");
  if (s.w > 0 && s.h > 0) $("skeleton").style.aspectRatio = s.w + " / " + s.h;
}

function updateTasks() {
  const image = state.mode === "image", reference = $("video-task").value === "reference";
  $("image-task-wrap").hidden = !image;
  $("video-inputs").hidden = image;
  $("frame-files").hidden = image || reference;
  $("vrefs-wrap").hidden = image || !reference;
  $("mask-help").hidden = $("image-task").value !== "mask";
  $("steps").min = image ? "1" : "2";
  $("task-help").textContent = IMAGE_TASK[$("image-task").value];
  $("steps-help").textContent = image ? "自訂 sigma 優先於此步數。" : "實際計算 " + Math.max(0, Number($("steps").value) - 1) + " 次；8 步 LoRA 填 9。";
  renderLoras(state.loraRows);
  updateDuration();
}

function visibleLoras() {
  const model = modelId(state.mode), task = $("video-task").value;
  return state.loraRows.filter((row) => (row.model === "unknown" || row.model === model) &&
    (state.mode === "image" || row.task === "any" || row.task === task));
}

function selectedLoras() {
  return visibleLoras().filter((row) => state.loraSelections[row.name]).map((row) => ({ name: row.name, scale: state.loraSelections[row.name].scale }));
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
  $("lora-help").textContent = "已套用加速設定：" + (state.mode === "video" ? preset.steps - 1 : preset.steps) + " 次計算。強度維持 1 最符合訓練設定。";
  updateTasks();
}

function clearAcceleration() {
  state.loraRows.filter((row) => row.purpose === "acceleration").forEach((row) => delete state.loraSelections[row.name]);
  $("steps").value = STEP_DEFAULT[state.mode];
  $("sigmas").value = "";
  $("shift-terminal").checked = true;
  $("video-shift").value = "12";
  $("lora-help").textContent = "已回復原版排程。";
}

function renderLoras(rows) {
  const box = $("loras");
  box.replaceChildren();
  const available = visibleLoras();
  if (!available.length) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = rows.length ? "目前任務沒有相容的 LoRA。" : "尚無 LoRA。將檔案放進 models/loras，再按重新掃描。";
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
    tag.textContent = row.purpose === "acceleration" ? "加速" : row.purpose === "style" ? "畫風" : "用途待確認";
    lab.append(check, text, tag);
    const range = document.createElement("input"), scale = document.createElement("span");
    range.type = "range"; range.min = "0"; range.max = "2"; range.step = "0.05";
    range.value = String(state.loraSelections[row.name]?.scale ?? 1);
    range.setAttribute("aria-label", row.name + " 強度");
    scale.className = "scale";
    scale.textContent = Number(range.value).toFixed(2);
    range.addEventListener("input", () => {
      scale.textContent = Number(range.value).toFixed(2);
      if (check.checked) state.loraSelections[row.name] = { scale: Number(range.value) };
      updateRequestPreview();
    });
    check.addEventListener("change", () => {
      if (check.checked) {
        if (row.purpose === "acceleration") clearAcceleration();
        state.loraSelections[row.name] = { scale: Number(range.value) };
        applyLoraPreset(row);
      } else {
        delete state.loraSelections[row.name];
        if (row.purpose === "acceleration") clearAcceleration();
      }
      renderLoras(rows); updateTasks(); updateRequestPreview();
    });
    line.append(lab, range, scale);
    box.append(line);
  });
}

function renderModelFiles() {
  const model = state.models.find((m) => m.kind === state.mode);
  if (!model) return;
  $("current-model").textContent = model.id;
  const signature = JSON.stringify([state.mode, model.files, model.defaults]);
  if (signature === state.fileSignature) return;
  state.fileSignature = signature;
  const box = $("model-files");
  box.replaceChildren();
  Object.entries(model.files || {}).forEach(([role, choices]) => {
    const label = document.createElement("label"), select = document.createElement("select");
    label.className = "field";
    label.append(document.createTextNode(FILE_LABEL[role] || role));
    select.dataset.role = role;
    const def = document.createElement("option");
    def.value = ""; def.textContent = "預設 · " + model.defaults[role]; select.append(def);
    choices.forEach((file) => {
      const option = document.createElement("option"); option.value = file; option.textContent = file; select.append(option);
    });
    select.value = state.fileSelections[state.mode][role] || "";
    select.addEventListener("change", () => { state.fileSelections[state.mode][role] = select.value; updateRequestPreview(); });
    label.append(select); box.append(label);
  });
}

function requestSettings() {
  const kind = state.mode, s = selectedSize(), task = $("image-task").value;
  let prompt = $("prompt").value.trim();
  if (kind === "image" && task === "transparent") prompt = "This is an RGBA image with transparency. " + prompt + " The background is transparent.";
  if (kind === "image" && task === "extract") prompt = "This is an RGBA image with transparency. Extract the subject from the input photo. " + prompt + " The background is transparent.";
  if (kind === "image" && task === "mask") prompt = "Picture 2 is a mask. In Picture 1, modify only inside the white area of the mask. " + prompt;
  const body = { type: kind, model: modelId(kind), prompt, size: sizes()[state.size][1] === 0 ? "auto" : s.w + "x" + s.h, steps: Number($("steps").value), seed: Number($("seed").value.trim() || -1) };
  if (kind === "image") {
    body.cfg_scale = Number($("cfg").value);
    if ($("negative").value.trim()) body.negative_prompt = $("negative").value.trim();
    body.shift_terminal = $("shift-terminal").checked;
    if ($("sigmas").value.trim()) body.sigmas = $("sigmas").value.split(/[,，\s]+/).filter(Boolean).map(Number);
  } else {
    body.num_frames = Number($("frames").value);
    body.video_shift = Number($("video-shift").value);
    if ($("video-task").value === "reference") { body.ref_image_short_edge = Number($("ref-edge").value); body.ref_native_canvas = $("ref-native").checked; }
  }
  body.loras = selectedLoras();
  if ($("compute").value) body.compute = $("compute").value;
  const files = Object.fromEntries(Object.entries(state.fileSelections[kind]).filter(([, name]) => name));
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
    if ($("image-task").value === "extract" && !state.refs.length) throw new Error("去背需要加入原圖");
    if ($("image-task").value === "mask" && state.refs.length < 2) throw new Error("局部修改需要原圖與遮罩圖");
  } else {
    if (!Number.isInteger(body.num_frames) || body.num_frames < 5 || body.num_frames > 361) throw new Error("目標幀數須為 5～361");
    if (!Number.isFinite(body.video_shift) || body.video_shift < 0.5 || body.video_shift > 50) throw new Error("影片 shift 須為 0.5～50");
    if ($("video-task").value === "reference") {
      if (!state.vrefs.length || state.vrefs.every((r) => r.kind === "audio")) throw new Error("參考生成需要至少一張圖片或一段影片");
      if (!Number.isInteger(body.ref_image_short_edge) || body.ref_image_short_edge < 256 || body.ref_image_short_edge > 4096) throw new Error("參考圖短邊須為 256～4096 的整數");
      body.references = state.vrefs.map((r) => ({ type: r.kind, data: r.url }));
    } else {
      if ($("first-frame").files[0]) body.first_frame = await readFile($("first-frame").files[0]);
      if ($("last-frame").files[0]) body.last_frame = await readFile($("last-frame").files[0]);
    }
  }
  return body;
}

function updateRequestPreview() {
  const body = requestSettings();
  if (state.mode === "image" && state.refs.length) body.images = state.refs.map((r) => "[base64: " + r.name + "]");
  if (state.mode === "video") {
    if ($("video-task").value === "reference") body.references = state.vrefs.map((r) => ({ type: r.kind, data: "[base64: " + r.name + "]" }));
    else ["first", "last"].forEach((key) => { const file = $(key + "-frame").files[0]; if (file) body[key + "_frame"] = "[base64: " + file.name + "]"; });
  }
  $("request-preview").textContent = JSON.stringify(body, null, 2);
}

function bytes(value) { return Number.isFinite(value) ? (value / 1073741824).toFixed(2) + " GiB" : "暫無資料"; }
function metric(label, value) {
  const dl = document.createElement("dl"), dt = document.createElement("dt"), dd = document.createElement("dd");
  dt.textContent = label; dd.textContent = value; dl.append(dt, dd); return dl;
}
function percent(value) { return Number.isFinite(value) ? value.toFixed(1) + "%" : "暫無資料"; }

function renderSystem(data) {
  $("queue").textContent = "工作 " + data.queue_length;
  const host = data.host || {}, proc = data.process || {}, apiData = data.api || {};
  $("host-metrics").replaceChildren(metric("CPU 使用率", percent(host.cpu_percent)), metric("系統 RAM", bytes(host.ram_used) + " / " + bytes(host.ram_total)), metric("Miroxen RAM", bytes(proc.ram_used)), metric("Miroxen CPU", percent(proc.cpu_percent)));
  $("api-metrics").replaceChildren(metric("服務運行", Math.floor((apiData.uptime_seconds || 0) / 60) + " 分鐘"), metric("API 請求", String(apiData.requests || 0)), metric("錯誤回應", String(apiData.errors || 0)), metric("驗證", apiData.authentication ? "需要金鑰" : "未設金鑰"));
  const box = $("gpus"); box.replaceChildren();
  (data.gpus || []).forEach((g) => {
    const section = document.createElement("section"), heading = document.createElement("h2"), metrics = document.createElement("div"), bar = document.createElement("div"), fill = document.createElement("span");
    section.className = "gpu-row"; metrics.className = "host-metrics";
    heading.textContent = "GPU " + g.index + " · " + g.name;
    metrics.append(metric("GPU 使用率", percent(g.utilization_percent)), metric("顯存", bytes(g.total - g.free) + " / " + bytes(g.total)), metric("溫度", Number.isFinite(g.temperature_c) ? g.temperature_c + " °C" : "暫無資料"), metric("功耗", Number.isFinite(g.power_w) ? g.power_w.toFixed(1) + " W" : "暫無資料"));
    bar.className = "bar"; fill.style.width = (g.total ? Math.min(100, 100 * (1 - g.free / g.total)) : 0) + "%"; bar.append(fill);
    bar.setAttribute("role", "meter"); bar.setAttribute("aria-label", g.name + " 顯存使用率"); bar.setAttribute("aria-valuemin", "0"); bar.setAttribute("aria-valuemax", "100"); bar.setAttribute("aria-valuenow", String(g.total ? Math.round(100 * (1 - g.free / g.total)) : 0));
    section.append(heading, metrics, bar); box.append(section);
  });
  $("sample-time").textContent = "更新於 " + new Date().toLocaleTimeString("zh-TW");
  $("monitor-error").textContent = "";
  $("api-state").textContent = "已連線";
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  const field = document.createElement("textarea"), previous = document.activeElement;
  field.value = text; field.style.position = "fixed"; field.style.opacity = "0";
  document.body.append(field); field.select();
  const copied = document.execCommand("copy"); field.remove(); previous?.focus({ preventScroll: true });
  if (!copied) throw new Error("瀏覽器未允許複製，請選取網址複製。");
}

function renderNetwork(data) {
  $("network-state").textContent = data.lan_enabled ? "區域網路已開放 · 下列為各網卡網址。" : "僅限本機連線";
  const urls = data.urls || [];
  const lan = location.hostname !== "127.0.0.1" && location.hostname !== "localhost" ? location.origin :
    (urls.find((url) => /^192\.168\.|^10\./.test(new URL(url).hostname)) || urls[0]);
  $("lan-api-example").textContent = lan ? lan + "/v1" : "尚無區網網址；服務需開放區域網路。";
  $("lan-health-link").href = (lan || location.origin) + "/health";
  $("network-urls").replaceChildren(...urls.map((url) => {
    const li = document.createElement("li"), row = document.createElement("div"), a = document.createElement("a"), button = document.createElement("button"), studio = document.createElement("a");
    row.className = "address-row"; a.className = "mono"; a.href = url + "/v1/models"; a.textContent = url + "/v1"; a.title = "查看 API 模型清單";
    button.type = "button"; button.textContent = "複製"; button.setAttribute("aria-label", "複製 API " + url);
    button.addEventListener("click", async () => {
      try { await copyText(url + "/v1"); $("address-copy-status").textContent = "已複製 " + url + "/v1"; }
      catch (e) { $("address-copy-status").textContent = e.message; }
    });
    studio.href = url; studio.textContent = "工作台";
    row.append(a, button, studio); li.append(row); return li;
  }));
}

function updateModelSummary() {
  const loaded = state.models.find((m) => m.loaded);
  $("model-summary").textContent = state.modelChanging ? "模型切換中…" : loaded ? (loaded.kind === "image" ? "Qwen" : "H3") + " · 已載入" : "模型未載入";
  $("model-dot").classList.toggle("on", Boolean(loaded));
}

function updateModelActions() {
  updateModelSummary();
  const busy = state.modelChanging || state.submitting || state.jobs.some((j) => !isTerminal(j.status));
  $("load-model").disabled = busy;
  $("manage-model").disabled = busy;
  $("unload-model").disabled = busy || !state.models.some((m) => m.loaded);
}

async function controlModel(load) {
  if (state.modelChanging) return;
  state.modelChanging = true;
  updateModelActions(); setBusy(true); $("generate").disabled = true;
  $("model-action-status").textContent = load ? "載入中，請稍候…" : "卸載中…";
  try {
    const kind = $("manage-model").value;
    const body = { model: modelId(kind), model_files: Object.fromEntries(Object.entries(state.fileSelections[kind]).filter(([, name]) => name)) };
    if ($("compute").value) body.compute = $("compute").value;
    renderModels(await api("/v1/models/" + (load ? "load" : "unload"), { method: "POST", timeout: 120000, headers: { "Content-Type": "application/json" }, body: JSON.stringify(load ? body : {}) }));
    $("model-action-status").textContent = load ? "已載入 " + body.model : "模型已卸載；下次生成會自動載入。";
  } catch (e) { $("model-action-status").textContent = e.message; }
  finally { state.modelChanging = false; $("generate").disabled = false; setBusy(Boolean(state.activeId)); updateModelActions(); await refreshMeta(); }
}

function bindControls() {
  $("load-model").addEventListener("click", () => controlModel(true));
  $("unload-model").addEventListener("click", () => controlModel(false));
  document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll("[data-view]").forEach((b) => { b.setAttribute("aria-pressed", String(b === button)); $(b.dataset.view).hidden = b !== button; });
    updateRequestPreview(); refreshMeta();
  }));
  $("image-task").addEventListener("change", () => { updateTasks(); updateRequestPreview(); });
  $("video-task").addEventListener("change", () => { clearAcceleration(); updateTasks(); updateRequestPreview(); });
  $("clear-frames").addEventListener("click", () => { $("first-frame").value = ""; $("last-frame").value = ""; updateRequestPreview(); });
  $("duration").addEventListener("input", () => { const n = Number($("duration").value); if (Number.isFinite(n)) $("frames").value = Math.max(5, Math.ceil(n * 24)); updateDuration(); });
  $("frames").addEventListener("input", () => { $("duration").value = (Number($("frames").value) / 24).toFixed(2); updateDuration(); });
  $("form").addEventListener("input", () => { updateRequestPreview(); $("steps-help").textContent = state.mode === "video" ? "實際計算 " + Math.max(0, Number($("steps").value) - 1) + " 次" : "自訂 sigma 優先於此步數。"; });
  $("reload-loras").addEventListener("click", loadLoras);
  $("api-base").textContent = location.origin + "/v1";
  $("api-base").href = location.origin + "/v1/models";
  $("api-base").title = "查看 API 模型清單";
  $("copy-api-base").addEventListener("click", async () => {
    try { await copyText($("api-base").textContent); $("address-copy-status").textContent = "已複製 " + $("api-base").textContent; }
    catch (e) { $("address-copy-status").textContent = e.message; }
  });
  $("api-connection").addEventListener("submit", (ev) => { ev.preventDefault(); $("connect-api").click(); });
  $("connect-api").addEventListener("click", async () => { state.apiKey = $("api-key").value.trim(); state.authRequired = false; closeSource(); await refreshMeta(); await loadLoras(); if (state.activeId) watch(state.activeId); });
  $("copy-request").addEventListener("click", async () => {
    try { await copyText(JSON.stringify(await buildRequest(), null, 2)); $("copy-status").textContent = "已複製完整 JSON，可 POST 到 /v1/jobs。"; }
    catch (e) { $("copy-status").textContent = e.message || "複製失敗，請確認瀏覽器允許剪貼簿。"; }
  });
  updateTasks(); updateRequestPreview();
}
