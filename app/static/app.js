const rowsEl = document.getElementById("rows");
const healthEl = document.getElementById("health");
const gpuEl = document.getElementById("gpu");
const open = new Set();          // ids whose transcript is expanded
let rowCache = {};               // id -> recording summary
const rowEls = new Map();        // id -> { row, tr }  live DOM nodes for incremental updates
let lastOrder = "";              // last rendered id order, to skip needless reordering
let currentQuery = "";           // active search term
let modelInfo = { models: [], engines: [], default_model: "", default_engine: "" };
let appSettings = { default_preprocess: false, default_vad: true };

function fmtDur(s) {
  if (!s && s !== 0) return "—";
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  return `${m}:${String(sec).padStart(2, "0")}`;
}
function fmtTs(s) {
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

async function loadHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    healthEl.textContent = h.status;
    healthEl.title = `${h.engine} · ${h.model} · ${h.status}`;
  } catch { healthEl.textContent = "unreachable"; }
}

async function loadModels() {
  try { modelInfo = await (await fetch("/api/models")).json(); }
  catch { /* keep defaults */ }
}

async function deleteRecording(id, name) {
  if (!confirm(`Delete "${name}" from disk?\n\nThis permanently removes the file from the host. This cannot be undone.`))
    return;
  try {
    const res = await fetch(`/api/recordings/${id}`, { method: "DELETE" });
    if (!res.ok) { alert(`Delete failed: ${(await res.json()).detail}`); return; }
    open.delete(id);
    loadRecordings();
  } catch { alert("Delete failed."); }
}

async function loadSettings() {
  try {
    const s = await (await fetch("/api/settings")).json();
    appSettings = s;
    document.getElementById("recursive").checked = !!s.recursive;
    document.getElementById("scanDir").textContent = s.scan_dir;
  } catch { /* ignore */ }
}

async function loadStats() {
  const wrap = document.getElementById("progress");
  let s;
  try { s = await (await fetch("/api/stats")).json(); }
  catch { return; }
  if (!s.total) { wrap.hidden = true; return; }
  wrap.hidden = false;
  const pct = Math.round((s.done / s.total) * 100);
  document.getElementById("pfill").style.width = `${pct}%`;
  const queue = s.pending + s.processing;
  let info = `${s.done}/${s.total} transcribed (${pct}%)`;
  if (queue) info += ` · ${queue} in queue`;
  if (s.error) info += ` · ${s.error} errored`;
  document.getElementById("pinfo").textContent = info;
}

async function loadGpu() {
  try {
    const g = await (await fetch("/api/gpu")).json();
    const gpu = (g.gpus || [])[0];
    if (!gpu) { gpuEl.textContent = "GPU: none"; return; }
    const util = gpu.utilization_pct != null ? `${gpu.utilization_pct}%` : "—";
    const mem = gpu.memory_total_mb
      ? `${(gpu.memory_used_mb / 1024).toFixed(1)}/${(gpu.memory_total_mb / 1024).toFixed(1)}G` : "—";
    const temp = gpu.temperature_c != null ? ` · ${gpu.temperature_c}°C` : "";
    gpuEl.textContent = `${gpu.name} · ${util} · ${mem}${temp}`;
    gpuEl.title = `${g.source}: ${gpu.name}, util ${util}, mem ${mem}${temp}`;
  } catch { gpuEl.textContent = "GPU: unreachable"; }
}

async function loadRecordings() {
  let recs;
  const url = currentQuery ? `/api/recordings?q=${encodeURIComponent(currentQuery)}` : "/api/recordings";
  try { recs = await (await fetch(url)).json(); }
  catch { return; }
  document.getElementById("searchInfo").textContent =
    currentQuery ? `${recs.length} match${recs.length === 1 ? "" : "es"}` : "";

  // Incremental reconcile: reuse existing row DOM, update only what changed, and
  // leave already-rendered transcripts untouched unless their status changed.
  rowCache = {};
  const seen = new Set();
  for (const r of recs) {
    rowCache[r.id] = r;
    seen.add(r.id);
    let e = rowEls.get(r.id);
    if (!e) {
      e = { row: renderRow(r), tr: null };
      rowEls.set(r.id, e);
    } else {
      updateRow(e.row, r);
    }
    if (open.has(r.id)) {
      if (!e.tr) {
        e.tr = renderTranscriptRow(r.id);
        e.row.after(e.tr);
        loadTranscript(r.id, e.tr.querySelector(".transcript"));
        e.tr.dataset.st = r.status;
      } else if (e.tr.dataset.st !== r.status) {
        // status changed (e.g. pending -> done): refresh this transcript once
        loadTranscript(r.id, e.tr.querySelector(".transcript"));
        e.tr.dataset.st = r.status;
      }
    } else if (e.tr) {
      e.tr.remove();
      e.tr = null;
    }
  }

  // Drop rows that no longer exist (e.g. deleted, or filtered out by search).
  for (const [id, e] of rowEls) {
    if (!seen.has(id)) {
      e.row.remove();
      if (e.tr) e.tr.remove();
      rowEls.delete(id);
    }
  }

  // Reorder/insert only when the id sequence actually changed (rare: new file or
  // search change). This is the only branch that moves nodes, so steady-state
  // polling never disturbs the DOM you're reading.
  const order = recs.map(r => r.id).join(",");
  if (order !== lastOrder) {
    const frag = document.createDocumentFragment();
    for (const r of recs) {
      const e = rowEls.get(r.id);
      frag.appendChild(e.row);
      if (e.tr) frag.appendChild(e.tr);
    }
    rowsEl.appendChild(frag);
    lastOrder = order;
  }
}

function updateRow(row, r) {
  row.querySelector(".caret").classList.toggle("open", open.has(r.id));
  const badge = row.querySelector(".badge");
  if (badge.textContent !== r.status) {
    badge.className = `badge ${r.status}`;
    badge.textContent = r.status;
  }
  const dur = fmtDur(r.duration);
  const durCell = row.querySelector(".dur");
  if (durCell.textContent !== dur) durCell.textContent = dur;
}

function renderRow(r) {
  const tr = document.createElement("tr");
  tr.className = "rec";
  tr.dataset.id = r.id;
  const caretOpen = open.has(r.id) ? "open" : "";
  tr.innerHTML = `
    <td><span class="caret ${caretOpen}">▶</span></td>
    <td>${escapeHtml(r.filename)}</td>
    <td class="muted">${r.source}</td>
    <td><span class="badge ${r.status}">${r.status}</span></td>
    <td class="muted dur">${fmtDur(r.duration)}</td>
    <td><div class="actions">
      <button class="btn dl">Download</button>
      <button class="iconbtn trash" title="Delete file from disk">🗑</button>
    </div></td>`;
  tr.querySelector(".dl").addEventListener("click", (e) => {
    e.stopPropagation();
    window.location = `/api/recordings/${r.id}/download`;
  });
  tr.querySelector(".trash").addEventListener("click", (e) => {
    e.stopPropagation();
    deleteRecording(r.id, r.filename);
  });
  tr.addEventListener("click", () => toggle(r.id));
  return tr;
}

function renderTranscriptRow(id) {
  const tr = document.createElement("tr");
  tr.className = "transcript-row";
  tr.dataset.for = id;
  tr.innerHTML = `<td colspan="6"><div class="transcript">Loading…</div></td>`;
  return tr;
}

function toggle(id) {
  const e = rowEls.get(id);
  if (!e) return;
  if (open.has(id)) {
    open.delete(id);
    if (e.tr) { e.tr.remove(); e.tr = null; }
    e.row.querySelector(".caret").classList.remove("open");
  } else {
    open.add(id);
    e.tr = renderTranscriptRow(id);
    e.row.after(e.tr);
    e.row.querySelector(".caret").classList.add("open");
    loadTranscript(id, e.tr.querySelector(".transcript"));
    e.tr.dataset.st = (rowCache[id] || {}).status || "";
  }
}

async function loadTranscript(id, el) {
  let rec;
  try { rec = await (await fetch(`/api/recordings/${id}`)).json(); }
  catch { el.innerHTML = `<div class="err-msg">Failed to load.</div>`; return; }
  if (rec.status === "processing" || rec.status === "pending") {
    el.innerHTML = `<div class="empty">Not transcribed yet (${rec.status})…</div>`;
    return;
  }

  const parts = [];
  const hasSegments = rec.status !== "error" && rec.segments && rec.segments.length;
  if (rec.status === "error") {
    parts.push(`<div class="err-msg">Transcription error: ${escapeHtml(rec.error || "unknown")}</div>`);
  } else if (!hasSegments) {
    parts.push(`<div class="empty">No speech detected.</div>`);
  } else {
    parts.push(`<audio class="player" controls preload="none" src="/api/recordings/${id}/audio"></audio>`);
    parts.push(`<div class="tbar">
      <button class="btn copy">Copy text</button>
      <a class="btn" href="/api/recordings/${id}/export?fmt=txt">.txt</a>
      <a class="btn" href="/api/recordings/${id}/export?fmt=srt">.srt</a>
      <a class="btn" href="/api/recordings/${id}/export?fmt=vtt">.vtt</a>
      <button class="btn sel-all">Select all</button>
      <span class="muted seg-hint">tip: check segments to export a clip · click a timestamp to play</span>
    </div>`);
    const lowCount = rec.segments.filter(isLowConf).length;
    if (lowCount) parts.push(
      `<div class="conf-note">⚠ ${lowCount} low-confidence segment${lowCount === 1 ? "" : "s"} highlighted — try re-transcribing with cleaning on, VAD off, or a larger model.</div>`);
    parts.push(`<div class="selbar" hidden>
      <span class="selinfo"></span>
      <button class="btn sel-play">▶ Play selection</button>
      <button class="btn sel-wav">Export .wav</button>
      <button class="btn sel-copy">Copy</button>
      <button class="btn sel-txt">.txt</button>
      <button class="btn sel-clear">Clear</button>
    </div>`);
    parts.push(rec.segments.map((s, i) => {
      const low = isLowConf(s);
      const tip = low ? ` title="low confidence (logprob ${s.logprob}, no-speech ${s.nsp})"` : "";
      return `<div class="seg${low ? " lowconf" : ""}" data-i="${i}" data-start="${s.start}" data-end="${s.end}"${tip}>` +
        `<input type="checkbox" class="seg-sel">` +
        `<span class="ts" role="button">${fmtTs(s.start)}</span>` +
        `<span class="tx">${highlight(s.text)}</span></div>`;
    }).join(""));
  }
  parts.push(retranscribeControl(rec));
  el.innerHTML = parts.join("");

  const copyBtn = el.querySelector(".copy");
  if (copyBtn) copyBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(rec.text || "");
      e.target.textContent = "Copied ✓";
      setTimeout(() => (e.target.textContent = "Copy text"), 1500);
    } catch { e.target.textContent = "Copy failed"; }
  });
  if (hasSegments) wirePlayerAndSelection(el, id, rec);
  wireRetranscribe(el, id);
}

function wirePlayerAndSelection(el, id, rec) {
  const player = el.querySelector(".player");
  const segEls = [...el.querySelectorAll(".seg")];
  const selbar = el.querySelector(".selbar");
  const stem = (rec.filename || "clip").replace(/\.[^.]+$/, "");
  let clipQueue = null, clipIdx = 0;   // for "Play selection"

  const checked = () => segEls.filter(se => se.querySelector(".seg-sel").checked);
  const rangesOf = (els) => els
    .map(se => [parseFloat(se.dataset.start), parseFloat(se.dataset.end)])
    .filter(([s, e]) => e > s)
    .sort((a, b) => a[0] - b[0]);

  function updateSelbar() {
    const sel = checked();
    if (!sel.length) { selbar.hidden = true; clipQueue = null; return; }
    selbar.hidden = false;
    const total = rangesOf(sel).reduce((a, [s, e]) => a + (e - s), 0);
    el.querySelector(".selinfo").textContent =
      `${sel.length} segment${sel.length === 1 ? "" : "s"} · ${total.toFixed(1)}s`;
  }

  // Click a timestamp to play from there; checkbox toggles selection.
  segEls.forEach(se => {
    se.querySelector(".ts").addEventListener("click", () => {
      clipQueue = null;
      player.currentTime = parseFloat(se.dataset.start);
      player.play();
    });
    se.querySelector(".seg-sel").addEventListener("change", updateSelbar);
  });

  // Karaoke highlight + advance through a selection-play queue.
  player.addEventListener("timeupdate", () => {
    const t = player.currentTime;
    for (const se of segEls) {
      const s = parseFloat(se.dataset.start), e = parseFloat(se.dataset.end);
      se.classList.toggle("playing", t >= s && t < e);
    }
    if (clipQueue) {
      const [, end] = clipQueue[clipIdx];
      if (t >= end - 0.03) {
        clipIdx++;
        if (clipIdx >= clipQueue.length) { clipQueue = null; player.pause(); }
        else { player.currentTime = clipQueue[clipIdx][0]; }
      }
    }
  });

  el.querySelector(".sel-all").addEventListener("click", () => {
    const allOn = segEls.every(se => se.querySelector(".seg-sel").checked);
    segEls.forEach(se => (se.querySelector(".seg-sel").checked = !allOn));
    updateSelbar();
  });
  el.querySelector(".sel-clear").addEventListener("click", () => {
    segEls.forEach(se => (se.querySelector(".seg-sel").checked = false));
    updateSelbar();
  });
  el.querySelector(".sel-play").addEventListener("click", () => {
    const r = rangesOf(checked());
    if (!r.length) return;
    clipQueue = r; clipIdx = 0;
    player.currentTime = r[0][0];
    player.play();
  });
  el.querySelector(".sel-copy").addEventListener("click", (e) => {
    const txt = checked().map(se => rec.segments[+se.dataset.i].text).join("\n");
    navigator.clipboard.writeText(txt).then(() => {
      e.target.textContent = "Copied ✓";
      setTimeout(() => (e.target.textContent = "Copy"), 1500);
    }).catch(() => (e.target.textContent = "Copy failed"));
  });
  el.querySelector(".sel-txt").addEventListener("click", () => {
    const txt = checked().map(se => rec.segments[+se.dataset.i].text).join("\n");
    downloadBlob(new Blob([txt], { type: "text/plain" }), `${stem}_selection.txt`);
  });
  el.querySelector(".sel-wav").addEventListener("click", async (e) => {
    const r = rangesOf(checked());
    if (!r.length) return;
    e.target.disabled = true;
    e.target.textContent = "Building…";
    try {
      const res = await fetch(`/api/recordings/${id}/clip`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ranges: r }),
      });
      if (!res.ok) throw new Error();
      downloadBlob(await res.blob(), `${stem}_clip.wav`);
      e.target.textContent = "Export .wav";
    } catch {
      e.target.textContent = "Failed — retry";
    } finally {
      e.target.disabled = false;
    }
  });
}

function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function isLowConf(s) {
  return s.logprob != null && (s.logprob < -1.0 || (s.nsp != null && s.nsp > 0.6));
}

function retranscribeControl(rec) {
  const curModel = rec.model || modelInfo.default_model;
  const curEngine = rec.engine || modelInfo.default_engine;
  const modelOpts = modelInfo.models.map(m =>
    `<option value="${m}" ${m === curModel ? "selected" : ""}>${m}</option>`).join("");
  const engineOpts = modelInfo.engines.map(e =>
    `<option value="${e}" ${e === curEngine ? "selected" : ""}>${e}</option>`).join("");
  const pre = appSettings.default_preprocess ? "checked" : "";
  const vad = appSettings.default_vad ? "checked" : "";
  return `
    <div class="retro">
      <span class="muted">Re-transcribe with</span>
      <select class="rt-model">${modelOpts}</select>
      <select class="rt-engine">${engineOpts}</select>
      <label class="check" title="Pre-clean weak/noisy audio before transcribing"><input type="checkbox" class="rt-pre" ${pre}> Clean audio</label>
      <label class="check" title="Voice activity detection: trims silence. Turn off to recover quiet/marginal speech."><input type="checkbox" class="rt-vad" ${vad}> VAD</label>
      <button class="btn rt-go">Re-transcribe</button>
    </div>`;
}

function wireRetranscribe(el, id) {
  const btn = el.querySelector(".rt-go");
  if (!btn) return;
  btn.addEventListener("click", async (e) => {
    e.stopPropagation();
    const model = el.querySelector(".rt-model").value;
    const engine = el.querySelector(".rt-engine").value;
    const preprocess = el.querySelector(".rt-pre").checked;
    const vad = el.querySelector(".rt-vad").checked;
    btn.disabled = true;
    btn.textContent = "Queued…";
    try {
      await fetch(`/api/recordings/${id}/retranscribe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, engine, preprocess, vad }),
      });
      loadRecordings();
    } catch {
      btn.disabled = false;
      btn.textContent = "Failed — retry";
    }
  });
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function highlight(s) {
  const esc = escapeHtml(s);
  if (!currentQuery) return esc;
  const q = currentQuery.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return esc.replace(new RegExp(q, "gi"), m => `<mark>${m}</mark>`);
}

// ---- Uploads ----
const dz = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");

fileInput.addEventListener("change", () => uploadFiles(fileInput.files));
["dragenter", "dragover"].forEach(ev =>
  dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach(ev =>
  dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", e => uploadFiles(e.dataTransfer.files));

async function uploadFiles(files) {
  for (const f of files) {
    uploadStatus.textContent = `Uploading ${f.name}…`;
    const fd = new FormData();
    fd.append("file", f);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      uploadStatus.textContent = res.ok ? `Queued ${f.name}` : `Failed ${f.name}: ${(await res.json()).detail}`;
    } catch { uploadStatus.textContent = `Failed ${f.name}`; }
  }
  loadRecordings();
}

// ---- Search ----
const searchEl = document.getElementById("search");
let searchTimer = null;
searchEl.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    currentQuery = searchEl.value.trim();
    loadRecordings();
  }, 250);
});

// ---- Free models ----
const freeBtn = document.getElementById("freeBtn");
freeBtn.addEventListener("click", async () => {
  freeBtn.disabled = true;
  freeBtn.textContent = "Freeing…";
  try {
    const r = await (await fetch("/api/models/free", { method: "POST" })).json();
    freeBtn.textContent = `Freed ${r.freed.length}`;
    loadGpu();
  } catch { freeBtn.textContent = "Failed"; }
  setTimeout(() => { freeBtn.textContent = "Free models"; freeBtn.disabled = false; }, 1500);
});

// ---- Recursive setting ----
document.getElementById("recursive").addEventListener("change", async (e) => {
  try {
    await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recursive: e.target.checked }),
    });
  } catch { e.target.checked = !e.target.checked; }  // revert on failure
});

// ---- Log panel ----
const logToggle = document.getElementById("logToggle");
const logLevel = document.getElementById("logLevel");
const logOutput = document.getElementById("logOutput");
const logInfo = document.getElementById("logInfo");
let logsOpen = false;
let logTimer = null;

logToggle.addEventListener("click", () => {
  logsOpen = !logsOpen;
  logToggle.textContent = logsOpen ? "▼ Logs" : "▶ Logs";
  logLevel.hidden = !logsOpen;
  logOutput.hidden = !logsOpen;
  if (logsOpen) { loadLogs(); logTimer = setInterval(loadLogs, 3000); }
  else { clearInterval(logTimer); }
});
logLevel.addEventListener("change", loadLogs);

async function loadLogs() {
  const lvl = logLevel.value;
  const url = `/api/logs?limit=500${lvl ? `&level=${lvl}` : ""}`;
  let data;
  try { data = await (await fetch(url)).json(); }
  catch { logInfo.textContent = "unreachable"; return; }
  const lines = data.lines || [];
  const stick = logOutput.scrollTop + logOutput.clientHeight >= logOutput.scrollHeight - 20;
  logOutput.innerHTML = lines.map(l => {
    const t = new Date(l.ts * 1000).toLocaleTimeString();
    return `<span class="log-${l.level}">${t} ${l.level.padEnd(7)} ${escapeHtml(l.name)}: ${escapeHtml(l.msg)}</span>`;
  }).join("\n");
  logInfo.textContent = `${lines.length} lines`;
  if (stick) logOutput.scrollTop = logOutput.scrollHeight;  // follow tail
}

loadModels();
loadSettings();
loadHealth();
loadGpu();
loadRecordings();
loadStats();
setInterval(loadRecordings, 4000);   // live status refresh
setInterval(loadStats, 4000);        // backlog progress
setInterval(loadHealth, 15000);
setInterval(loadGpu, 3000);          // GPU telemetry
