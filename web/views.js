"use strict";

const TIMING_LABEL = {
  lora: "LoRA", input: "讀取素材", text_vision_encode: "文字／影像編碼", vae_encode: "VAE 編碼", dit_prepare: "DiT 準備",
  denoise: "取樣", latent_layout_transfer: "潛空間轉換", vae_decode: "VAE 解碼", png_encode: "PNG 輸出", lora_switch: "LoRA 切換",
  model_lora: "模型／LoRA 切換", input_encode: "素材編碼", text_encode: "文字編碼", latent_readback: "釋放取樣暫存",
  audio_decode_wav: "音訊解碼", video_prepare: "影片準備", video_vae_decode: "影片 VAE 解碼", ffmpeg_pipe: "ffmpeg 寫入", ffmpeg_finish: "ffmpeg 收尾",
};

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function bytes(value) { return Number.isFinite(value) ? (value / 1073741824).toFixed(1) + " GiB" : "—"; }
function percent(value) { return Number.isFinite(value) ? value.toFixed(0) + "%" : "—"; }
function metric(label, value, fill) {
  const box = el("div", "metric"), dt = el("span", "metric-label", label), dd = el("span", "metric-value num", value);
  box.append(dt, dd);
  if (Number.isFinite(fill)) box.append(meterBar(fill, label));
  return box;
}
function meterBar(pct, label) {
  const bar = el("div", "bar"), span = el("span");
  const v = Math.max(0, Math.min(100, pct));
  span.style.width = v + "%";
  if (v >= 90) bar.classList.add("hot");
  bar.setAttribute("role", "meter"); bar.setAttribute("aria-label", label);
  bar.setAttribute("aria-valuemin", "0"); bar.setAttribute("aria-valuemax", "100"); bar.setAttribute("aria-valuenow", String(Math.round(v)));
  bar.append(span);
  return bar;
}

function showView(name) {
  document.querySelectorAll("[data-view]").forEach((b) => { b.setAttribute("aria-pressed", String(b.dataset.view === name)); $(b.dataset.view).hidden = b.dataset.view !== name; });
  if (name === "api-panel") updateRequestPreview();
  if (name === "library") renderLibrary();
  window.scrollTo(0, 0);
  refreshMeta();
}

// ---- 硬體與 API 統計 ----
function renderSystem(data) {
  $("queue").textContent = data.queue_length ? "工作 " + data.queue_length : "閒置";
  $("queue").classList.toggle("busy", data.queue_length > 0);
  const gpus = data.gpus || [], gb = (g) => Math.round(g.total / 1073741824);
  $("gpu-chip").hidden = !gpus.length;
  $("gpu-chip").textContent = gpus.length > 1 ? "雙卡 " + gpus.map((g) => gb(g) + "G").join("＋") : gpus.length ? "單卡 " + gb(gpus[0]) + "G" : "";
  $("gpu-chip").title = gpus.map((g) => "GPU " + g.index + " · " + g.name).join("\n");
  const host = data.host || {}, proc = data.process || {}, apiData = data.api || {};
  const ramPct = host.ram_total ? 100 * host.ram_used / host.ram_total : NaN;
  $("host-metrics").replaceChildren(
    metric("CPU", percent(host.cpu_percent), host.cpu_percent),
    metric("系統 RAM", bytes(host.ram_used) + " / " + bytes(host.ram_total), ramPct),
    metric("Miroxen CPU", percent(proc.cpu_percent)),
    metric("Miroxen RAM", bytes(proc.ram_used)));
  const up = apiData.uptime_seconds || 0;
  $("api-metrics").replaceChildren(
    metric("服務運行", up >= 3600 ? Math.floor(up / 3600) + " 小時 " + Math.floor(up % 3600 / 60) + " 分" : Math.floor(up / 60) + " 分鐘"),
    metric("API 請求", String(apiData.requests || 0)), metric("錯誤回應", String(apiData.errors || 0)),
    metric("驗證", apiData.authentication ? "需要金鑰" : "未設金鑰"));
  $("gpus").replaceChildren(...gpus.map((g) => {
    const card = el("section", "gpu-row"), head = el("div", "block-head");
    head.append(el("h3", "", "GPU " + g.index + " · " + g.name), el("span", "muted num", Number.isFinite(g.temperature_c) ? g.temperature_c + " °C" : ""));
    const used = g.total - g.free, grid = el("div", "metrics");
    grid.append(metric("顯存", bytes(used) + " / " + bytes(g.total), g.total ? 100 * used / g.total : 0),
      metric("使用率", percent(g.utilization_percent), g.utilization_percent),
      metric("功耗", Number.isFinite(g.power_w) ? g.power_w.toFixed(0) + " W" : "—"));
    card.append(head, grid);
    return card;
  }));
  $("sample-time").textContent = "更新於 " + new Date().toLocaleTimeString("zh-TW");
  $("monitor-error").textContent = "";
}

// ---- 模型 ----
function renderModels(data) {
  const previous = state.models.map((m) => m.id + m.loaded).join();
  state.models = (data && data.data) || [];
  if (previous !== state.models.map((m) => m.id + m.loaded).join()) renderLoras(state.loraRows);
  $("models").replaceChildren(...state.models.map((m) => {
    const li = el("li", "model-row" + (m.loaded ? " loaded" : "")), info = el("div", "model-info"), title = el("div", "model-name");
    title.append(el("span", "dot" + (m.loaded ? " on" : "")), el("span", "", MODEL_NAME[m.id] || m.id), el("span", "muted", m.loaded ? "已載入" : "按需載入"));
    info.append(title, el("span", "muted mono small", m.id));
    const files = Object.entries(m.active_files || {}).map(([role, path]) => (FILE_LABEL[role] || role) + "：" + String(path).split(/[\\/]/).pop());
    if (files.length) info.append(el("span", "muted small", files.join(" · ")));
    else if (m.detail) info.append(el("span", "muted small", m.detail));
    const button = el("button", m.loaded ? "ghost" : "", m.loaded ? "重新載入" : "載入");
    button.type = "button"; button.dataset.model = m.id;
    button.addEventListener("click", () => controlModel(true, m.id));
    li.append(info, button);
    return li;
  }));
  updateModelActions();
}

function updateModelSummary() {
  const loaded = state.models.find((m) => m.loaded);
  $("model-summary").textContent = state.modelChanging ? "模型切換中…" : loaded ? (MODEL_NAME[loaded.id] || loaded.id) : "模型未載入";
  $("model-dot").classList.toggle("on", Boolean(loaded));
}

function updateModelActions() {
  updateModelSummary();
  const busy = state.modelChanging || state.submitting || state.jobs.some((j) => !isTerminal(j.status));
  $("models").querySelectorAll("button").forEach((b) => { b.disabled = busy; });
  $("unload-model").disabled = busy || !state.models.some((m) => m.loaded);
}

async function controlModel(load, id) {
  if (state.modelChanging) return;
  state.modelChanging = true;
  updateModelActions(); setBusy(true); $("generate").disabled = true;
  $("model-action-status").textContent = load ? "載入 " + (MODEL_NAME[id] || id) + "，請稍候…" : "卸載中…";
  try {
    const body = { model: id, model_files: Object.fromEntries(Object.entries(fileSelection(id)).filter(([, name]) => name)) };
    if ($("compute").value) body.compute = $("compute").value;
    renderModels(await api("/v1/models/" + (load ? "load" : "unload"), { method: "POST", timeout: 180000, headers: { "Content-Type": "application/json" }, body: JSON.stringify(load ? body : {}) }));
    $("model-action-status").textContent = load ? "已載入 " + (MODEL_NAME[id] || id) : "模型已卸載；下次生成會自動載入。";
  } catch (e) { $("model-action-status").textContent = e.message; }
  finally { state.modelChanging = false; $("generate").disabled = false; setBusy(Boolean(state.activeId)); updateModelActions(); await refreshMeta(); }
}

// ---- 網路與 API ----
async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  const field = document.createElement("textarea"), previous = document.activeElement;
  field.value = text; field.style.position = "fixed"; field.style.opacity = "0";
  document.body.append(field); field.select();
  const copied = document.execCommand("copy"); field.remove(); previous?.focus({ preventScroll: true });
  if (!copied) throw new Error("瀏覽器未允許複製，請手動選取。");
}

function renderNetwork(data) {
  $("network-state").textContent = data.lan_enabled ? "區域網路已開放，下列為各網卡網址：" : "目前僅限本機連線（啟動時用 --host 0.0.0.0 開放區網）。";
  const urls = data.urls || [];
  const lan = location.hostname !== "127.0.0.1" && location.hostname !== "localhost" ? location.origin :
    (urls.find((url) => /^192\.168\.|^10\./.test(new URL(url).hostname)) || urls[0]);
  $("lan-api-example").textContent = lan ? lan + "/v1" : "尚無區網網址";
  $("lan-health-link").href = (lan || location.origin) + "/health";
  $("network-urls").replaceChildren(...urls.map((url) => {
    const li = el("li"), row = el("div", "address-row"), a = el("a", "mono", url + "/v1"), button = el("button", "ghost small", "複製"), studio = el("a", "", "工作台");
    a.href = url + "/v1/models"; a.title = "查看 API 模型清單";
    button.type = "button"; button.setAttribute("aria-label", "複製 API " + url);
    button.addEventListener("click", () => copyWithStatus(url + "/v1", "address-copy-status"));
    studio.href = url;
    row.append(a, button, studio); li.append(row); return li;
  }));
}

async function copyWithStatus(text, statusId, done) {
  try { await copyText(text); $(statusId).textContent = done || "已複製"; }
  catch (e) { $(statusId).textContent = e.message; }
}

function curlCommand(body) {
  const auth = state.apiKey ? " -H \"Authorization: Bearer <API key>\"" : "";
  return "curl -X POST " + location.origin + "/v1/jobs -H \"Content-Type: application/json\"" + auth + " -d @request.json";
}

// ---- 作品詳情 ----
function renderDetail(job) {
  const p = job.parameters || {}, r = job.result || {};
  $("detail-prompt").textContent = job.prompt || "";
  const rows = [
    ["模型", MODEL_NAME[p.model || (job.type === "video" ? "minimax-h3" : "")] || p.model || "H3 影音"],
    ["尺寸", p.width && p.height ? p.width + " × " + p.height : "自動"],
    ["步數", p.steps], ["種子", r.seed],
    ["生成耗時", typeof r.seconds === "number" ? r.seconds.toFixed(1) + " 秒" : null],
    ["影片長度", p.duration_seconds ? p.duration_seconds.toFixed(2) + " 秒 · " + p.num_frames + " 幀" : null],
    ["CFG", job.type === "image" ? p.cfg_scale : null],
    ["精度", p.compute ? p.compute.toUpperCase() : null],
    ["LoRA", (p.loras || []).map((l) => l.name.replace(/\.safetensors$/i, "") + " ×" + l.scale).join("、") || null],
  ];
  $("detail-meta").replaceChildren(...rows.filter(([, v]) => v !== null && v !== undefined && v !== "").map(([k, v]) => {
    const box = el("div"); box.append(el("dt", "", k), el("dd", "num", String(v))); return box;
  }));
  const timings = Object.entries(r.timings || {}).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  $("timings-wrap").hidden = !timings.length;
  const max = timings.length ? timings[0][1] : 1;
  $("timings").replaceChildren(...timings.map(([k, v]) => {
    const li = el("li"), bar = el("span", "t-bar"), fill = el("span");
    li.title = k + " " + v.toFixed(3) + " 秒";
    fill.style.width = Math.max(1, 100 * v / max) + "%"; bar.append(fill);
    li.append(el("span", "t-name", TIMING_LABEL[k] || k), bar, el("span", "t-val num", v < 1 ? v.toFixed(2) + " s" : v.toFixed(1) + " s"));
    return li;
  }));
  $("detail").hidden = false;
}

// ---- 重用設定 ----
function splitTask(prompt) {
  for (const [task, [pre, post]] of Object.entries(TASK_WRAP).reverse()) {  // extract 的前綴包含 transparent 的前綴，先比對長的
    if (prompt.startsWith(pre) && prompt.endsWith(post)) return [task, prompt.slice(pre.length, prompt.length - post.length)];
  }
  return ["generate", prompt];
}

function reuseJob(job, seedOnly) {
  const p = job.parameters || {};
  if (seedOnly) {
    $("seed").value = String(job.result ? job.result.seed : p.seed ?? -1);
    updateAdvancedSummary(); updateRequestPreview();
    $("status").textContent = "已沿用種子 " + $("seed").value;
    return;
  }
  setMode(job.type);
  state.loraSelections = {};
  const notes = [];
  if (job.type === "image") {
    applyImageModel(p.model || "qwen-image-2.1");
    const [task, prompt] = splitTask(job.prompt || "");
    setRadio("image-task", task);
    $("prompt").value = prompt;
    $("cfg").value = String(p.cfg_scale ?? 1);
    $("negative").value = p.negative_prompt || "";
    $("negative-wrap").open = Boolean(p.negative_prompt);
    $("sigmas").value = (p.sigmas || []).join(", ");
    $("shift-terminal").checked = p.shift_terminal !== false;
    if (p.reference_images) notes.push("原本的 " + p.reference_images + " 張參考圖需重新加入");
  } else {
    setRadio("video-task", p.task || "first_last");
    $("prompt").value = job.prompt || "";
    $("video-shift").value = String(p.video_shift ?? 12);
    $("frames").value = String(p.num_frames || 56);
    $("duration").value = ((p.num_frames || 56) / 24).toFixed(2);
    if (p.task === "reference") { $("ref-edge").value = String(p.ref_image_short_edge || 1024); $("ref-native").checked = Boolean(p.ref_native_canvas); notes.push("參考素材需重新加入"); }
  }
  const list = sizes(), idx = list.findIndex((s) => s[1] === p.width && s[2] === p.height);
  if (!p.width) state.size = list.findIndex((s) => s[1] === 0);
  else if (idx >= 0) state.size = idx;
  else { state.size = list.findIndex((s) => s[1] === -1); $("width").value = String(p.width); $("height").value = String(p.height); }
  $("steps").value = String(p.steps || stepDefault(job.type));
  $("seed").value = String(p.seed ?? -1);
  $("compute").value = p.compute && [...$("compute").options].some((o) => o.value === p.compute) ? p.compute : "";
  (p.loras || []).forEach((l) => { state.loraSelections[l.name] = { scale: l.scale }; });
  const id = modelId(job.type);
  state.fileSelections[id] = {};
  Object.entries(p.model_files || {}).forEach(([role, path]) => { state.fileSelections[id][role] = String(path).split(/[\\/]/).pop(); });
  state.fileSignature = "";
  renderModelFiles(); renderSizes(); updateTasks(); updateRequestPreview();
  showView("studio");
  $("status").textContent = "已載入這件作品的設定" + (notes.length ? "；" + notes.join("、") : "") + "。";
  $("prompt").focus();
}

// ---- 作品庫 ----
function jobThumb(job) {
  const box = el("div", "thumb");
  if (job.status === "succeeded" && job.result && job.result.url) {
    if (job.type === "video") {
      const v = el("video"); v.src = job.result.url + "#t=0.1"; v.muted = true; v.preload = "metadata"; box.append(v);
      box.append(el("span", "badge", "影片"));
    } else { const img = el("img"); img.alt = ""; img.loading = "lazy"; img.src = job.result.url; box.append(img); }
  } else {
    const p = job.progress || {};
    box.append(el("span", "thumb-state", STATUS_TEXT[job.status] + (job.status === "running" && p.total ? " " + p.step + "/" + p.total : "")));
    box.classList.add("state-" + job.status);
  }
  return box;
}

function libraryMatch(job) {
  const f = state.libraryFilter;
  return f === "all" || f === job.type || (f === "active" && !isTerminal(job.status)) || (f === "failed" && (job.status === "failed" || job.status === "cancelled"));
}

function renderLibrary() {
  const done = state.jobs.filter((j) => j.status === "succeeded").length;
  $("library-count").textContent = done ? String(done) : "";
  if ($("library").hidden) return;
  const jobs = state.jobs.filter(libraryMatch);
  const grid = $("library-grid");
  if (!jobs.length) { grid.replaceChildren(el("p", "empty-line", state.jobs.length ? "沒有符合篩選的作品。" : "還沒有作品。到「創作」寫下提示詞，生成的圖片與影片會收在這裡。")); return; }
  grid.replaceChildren(...jobs.map((job) => {
    const card = el("article", "card"), open = el("button", "card-open"), body = el("div", "card-body"), actions = el("div", "card-actions");
    open.type = "button"; open.setAttribute("aria-label", "檢視 " + (job.prompt || job.id));
    open.append(jobThumb(job));
    open.addEventListener("click", () => { viewJob(job); showView("studio"); });
    const meta = [STATUS_TEXT[job.status], job.parameters && job.parameters.width ? job.parameters.width + "×" + job.parameters.height : "", job.result && typeof job.result.seconds === "number" ? job.result.seconds.toFixed(1) + " 秒" : ""].filter(Boolean).join(" · ");
    body.append(el("p", "card-prompt", job.prompt || "（無提示詞）"), el("span", "muted small num", meta));
    if (job.error) body.append(el("span", "error small", job.error.message));
    const reuse = el("button", "ghost small", "重用設定"); reuse.type = "button"; reuse.addEventListener("click", () => reuseJob(job));
    actions.append(reuse);
    if (!isTerminal(job.status)) { const c = el("button", "danger small", "取消"); c.type = "button"; c.addEventListener("click", () => cancelJob(job.id)); actions.append(c); }
    else if (job.result && job.result.url) { const a = el("a", "button small", "下載"); a.href = job.result.url; a.download = job.id + (job.type === "video" ? ".mp4" : ".png"); actions.append(a); }
    body.append(actions);
    card.append(open, body);
    return card;
  }));
}

function bindViews() {
  document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
  document.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => showView(b.dataset.goto)));
  $("library-filter").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    state.libraryFilter = b.dataset.filter;
    $("library-filter").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderLibrary();
  }));
  $("unload-model").addEventListener("click", () => controlModel(false));
  $("api-base").textContent = location.origin + "/v1";
  $("api-base").href = location.origin + "/v1/models";
  $("api-base").title = "查看 API 模型清單";
  $("copy-api-base").addEventListener("click", () => copyWithStatus($("api-base").textContent, "address-copy-status", "已複製 " + $("api-base").textContent));
  $("api-connection").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    rememberKey($("api-key").value.trim()); state.authRequired = false; closeSource();
    await refreshMeta(); await loadLoras();
    if (!state.authRequired) $("api-error").textContent = "";
    if (state.activeId) watch(state.activeId);
  });
  $("copy-request").addEventListener("click", async () => {
    try { await copyText(JSON.stringify(await buildRequest(), null, 2)); $("copy-status").textContent = "已複製完整 JSON，可 POST 到 /v1/jobs。"; }
    catch (e) { $("copy-status").textContent = e.message || "複製失敗"; }
  });
  $("copy-curl").addEventListener("click", () => copyWithStatus(curlCommand(), "copy-status", "已複製 curl 指令；請把 JSON 存成 request.json。"));
  $("reuse").addEventListener("click", () => { const job = state.jobs.find((j) => j.id === state.viewId); if (job) reuseJob(job); });
  $("use-seed").addEventListener("click", () => { const job = state.jobs.find((j) => j.id === state.viewId); if (job) reuseJob(job, true); });
}
