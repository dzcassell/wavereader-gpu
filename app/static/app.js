const rowsEl = document.getElementById("rows");
const healthEl = document.getElementById("health");
const gpuEl = document.getElementById("gpu");
const open = new Set();          // ids whose transcript is expanded
let rowCache = {};               // id -> recording summary
const rowEls = new Map();        // id -> { row, tr }  live DOM nodes for incremental updates
let lastOrder = "";              // last rendered id order, to skip needless reordering
let currentQuery = "";           // active search term
let alertsOnly = false;          // filter: only recordings that hit a watch term
const pendingSeek = {};          // id -> seconds to seek once its player is wired
let modelInfo = { models: [], engines: [], default_model: "", default_engine: "" };
let appSettings = { default_preprocess: false, default_vad: true };
let speakers = [];               // [{id, name, prints, tags}]

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
  const params = new URLSearchParams();
  if (currentQuery) params.set("q", currentQuery);
  if (alertsOnly) params.set("alerts_only", "true");
  const qs = params.toString();
  try { recs = await (await fetch("/api/recordings" + (qs ? "?" + qs : ""))).json(); }
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

function alertMark(r) {
  return (r.alerts && r.alerts.length) ? "🔔" : "";
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
  const ab = row.querySelector(".alert-badge");
  const mark = alertMark(r);
  if (ab.textContent !== mark) ab.textContent = mark;
}

function renderRow(r) {
  const tr = document.createElement("tr");
  tr.className = "rec";
  tr.dataset.id = r.id;
  const caretOpen = open.has(r.id) ? "open" : "";
  tr.innerHTML = `
    <td><span class="caret ${caretOpen}">▶</span></td>
    <td>${escapeHtml(r.filename)}<span class="alert-badge" title="matched a watch term">${alertMark(r)}</span></td>
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
    if (rec.entities && rec.entities.length) parts.push(entityChips(rec.entities));
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
      <select class="tag-spk"></select>
      <button class="btn sel-tag">Tag as…</button>
      <button class="btn sel-play">▶ Play selection</button>
      <button class="btn sel-wav">Export .wav</button>
      <button class="btn sel-copy">Copy</button>
      <button class="btn sel-txt">.txt</button>
      <button class="btn sel-clear">Clear</button>
    </div>`);
    parts.push(rec.segments.map((s, i) => {
      const low = isLowConf(s);
      const tip = low ? ` title="low confidence (logprob ${s.logprob}, no-speech ${s.nsp})"` : "";
      const spk = s.speaker ? speakerChip(s, i) : "";
      return `<div class="seg${low ? " lowconf" : ""}" data-i="${i}" data-start="${s.start}" data-end="${s.end}"${tip}>` +
        `<input type="checkbox" class="seg-sel">` +
        `<span class="ts" role="button">${fmtTs(s.start)}</span>` +
        spk +
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

function entityChips(entities) {
  const byVal = new Map();   // dedupe by value, keep earliest occurrence
  for (const e of entities) {
    const cur = byVal.get(e.value);
    if (!cur || e.start < cur.start) byVal.set(e.value, e);
  }
  const chips = [...byVal.values()]
    .sort((a, b) => a.start - b.start)
    .map(e => `<button class="chip ent-${e.type}" data-start="${e.start}" title="${e.type} @ ${fmtTs(e.start)}">${escapeHtml(e.value)}</button>`)
    .join("");
  return `<div class="entities">${chips}</div>`;
}

function speakerChip(s, i) {
  const auto = s.speaker_source === "auto";
  const conf = (auto && s.speaker_conf != null) ? ` ${Math.round(s.speaker_conf * 100)}%` : "";
  const title = auto ? `auto-identified${conf}` : "tagged";
  return `<span class="spk${auto ? " auto" : ""}" title="${title}">${escapeHtml(s.speaker)}${conf}` +
    `<button class="spk-x" data-i="${i}" title="remove tag">×</button></span>`;
}

function speakerOptions(selectedId) {
  const opts = speakers.map(sp =>
    `<option value="${sp.id}" ${sp.id === selectedId ? "selected" : ""}>${escapeHtml(sp.name)}</option>`).join("");
  return opts + `<option value="__new__">+ new speaker…</option>`;
}

function seekTo(player, s) {
  const go = () => { try { player.currentTime = s; } catch (_) {} player.play(); };
  if (player.readyState >= 1) go();
  else { player.addEventListener("loadedmetadata", go, { once: true }); player.load(); }
}

function wirePlayerAndSelection(el, id, rec) {
  const player = el.querySelector(".player");
  const segEls = [...el.querySelectorAll(".seg")];
  const selbar = el.querySelector(".selbar");
  const stem = (rec.filename || "clip").replace(/\.[^.]+$/, "");
  let clipQueue = null, clipIdx = 0;   // for "Play selection"

  el.querySelectorAll(".chip").forEach(c =>
    c.addEventListener("click", () => { clipQueue = null; seekTo(player, parseFloat(c.dataset.start)); }));

  // Speaker tagging
  const tagSel = el.querySelector(".tag-spk");
  if (tagSel) tagSel.innerHTML = speakerOptions();
  el.querySelectorAll(".spk-x").forEach(b => b.addEventListener("click", async (ev) => {
    ev.stopPropagation();
    await fetch(`/api/recordings/${id}/untag`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ segments: [+b.dataset.i] }),
    });
    loadTranscript(id, el);
  }));

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
      seekTo(player, parseFloat(se.dataset.start));
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
  el.querySelector(".sel-tag").addEventListener("click", async (ev) => {
    const ids = checked().map(se => +se.dataset.i);
    if (!ids.length) return;
    const v = el.querySelector(".tag-spk").value;
    let body;
    if (v === "__new__") {
      const name = prompt("Name this voice:");
      if (!name || !name.trim()) return;
      body = { segments: ids, name: name.trim() };
    } else {
      body = { segments: ids, speaker_id: +v };
    }
    ev.target.disabled = true;
    ev.target.textContent = "Tagging…";
    try {
      const res = await fetch(`/api/recordings/${id}/tag`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error();
      const r = await res.json();
      await loadSpeakers();          // refresh roster (may have a new name)
      loadTranscript(id, el);        // re-render with labels
      if (r.skipped_short) {
        // brief heads-up that some clips were too short to enroll a voiceprint
        console.info(`${r.skipped_short} segment(s) too short to enroll`);
      }
    } catch {
      ev.target.disabled = false;
      ev.target.textContent = "Tag failed — retry";
    }
  });
  el.querySelector(".sel-play").addEventListener("click", () => {
    const r = rangesOf(checked());
    if (!r.length) return;
    clipQueue = r; clipIdx = 0;
    seekTo(player, r[0][0]);
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

  // If we were asked to jump here (from the Heard index or search results), do it now.
  if (pendingSeek[id] != null) {
    const s = pendingSeek[id];
    delete pendingSeek[id];
    seekTo(player, s);
  }
}

async function openAndPlay(id, start) {
  if (!rowEls.has(id)) {
    // Row may be hidden by an active filter — clear filters so it exists, then retry.
    currentQuery = ""; alertsOnly = false;
    const search = document.getElementById("search");
    if (search) search.value = "";
    document.getElementById("alertsOnly").checked = false;
    await loadRecordings();
  }
  const e = rowEls.get(id);
  if (!e) return;
  e.row.scrollIntoView({ behavior: "smooth", block: "center" });
  if (!open.has(id)) {
    pendingSeek[id] = start;
    toggle(id);                      // opens + async-loads transcript, which consumes pendingSeek
  } else if (e.tr) {
    const p = e.tr.querySelector(".player");
    if (p) seekTo(p, start); else pendingSeek[id] = start;
  }
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
const searchResults = document.getElementById("searchResults");
let searchTimer = null;
searchEl.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    currentQuery = searchEl.value.trim();
    loadRecordings();
    loadSearchResults();
  }, 250);
});

async function loadSearchResults() {
  if (!currentQuery) { searchResults.hidden = true; searchResults.innerHTML = ""; return; }
  let data;
  try { data = await (await fetch(`/api/search?q=${encodeURIComponent(currentQuery)}&limit=100`)).json(); }
  catch { return; }
  const hits = data.hits || [];
  if (!hits.length) { searchResults.hidden = true; searchResults.innerHTML = ""; return; }
  searchResults.hidden = false;
  searchResults.innerHTML = hits.map(h =>
    `<div class="sr-hit" data-id="${h.id}" data-start="${h.start}">` +
    `<span class="sr-time">${fmtTs(h.start)}</span>` +
    `<span class="sr-file">${escapeHtml(h.filename)}</span>` +
    `<span class="sr-text">${highlight(h.text)}</span></div>`).join("");
  searchResults.querySelectorAll(".sr-hit").forEach(el =>
    el.addEventListener("click", () => openAndPlay(+el.dataset.id, parseFloat(el.dataset.start))));
}

// ---- Alerts-only filter ----
document.getElementById("alertsOnly").addEventListener("change", (e) => {
  alertsOnly = e.target.checked;
  loadRecordings();
});

// ---- Heard & alerts panel ----
const heardBtn = document.getElementById("heardBtn");
const heardPanel = document.getElementById("heardpanel");
const entType = document.getElementById("entType");
const entList = document.getElementById("entList");
const watchBox = document.getElementById("watchTerms");
const watchInfo = document.getElementById("watchInfo");

heardBtn.addEventListener("click", () => {
  heardPanel.hidden = !heardPanel.hidden;
  if (!heardPanel.hidden) { loadEntities(); loadWatch(); }
});
entType.addEventListener("change", loadEntities);
document.getElementById("entRefresh").addEventListener("click", loadEntities);

async function loadEntities() {
  entList.innerHTML = `<span class="muted">loading…</span>`;
  let data;
  try { data = await (await fetch(`/api/entities?type=${entType.value}&limit=400`)).json(); }
  catch { entList.innerHTML = `<span class="muted">unreachable</span>`; return; }
  const ents = data.entities || [];
  if (!ents.length) { entList.innerHTML = `<span class="muted">nothing detected yet — try "Re-index all"</span>`; return; }
  entList.innerHTML = ents.map(e => {
    const occ = e.occurrences[0];
    return `<div class="ent-row">` +
      `<span class="ent-val chip ent-${e.type}" data-id="${occ.id}" data-start="${occ.start}">${escapeHtml(e.value)}</span>` +
      `<span class="ent-count">×${e.count}</span>` +
      `<span class="ent-occ" data-id="${occ.id}" data-start="${occ.start}">play first ▶</span></div>`;
  }).join("");
  entList.querySelectorAll("[data-id]").forEach(el =>
    el.addEventListener("click", () => openAndPlay(+el.dataset.id, parseFloat(el.dataset.start))));
}

async function loadWatch() {
  try {
    const d = await (await fetch("/api/watch")).json();
    watchBox.value = (d.terms || []).join("\n");
  } catch { /* ignore */ }
}

document.getElementById("watchSave").addEventListener("click", async () => {
  const terms = watchBox.value.split("\n").map(t => t.trim()).filter(Boolean);
  watchInfo.textContent = "saving…";
  try {
    const d = await (await fetch("/api/watch", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ terms }),
    })).json();
    watchInfo.textContent = `${d.terms.length} term(s) saved — Re-index to apply to existing files`;
  } catch { watchInfo.textContent = "save failed"; }
});

document.getElementById("reindexBtn").addEventListener("click", async (e) => {
  e.target.disabled = true;
  watchInfo.textContent = "re-indexing all transcripts…";
  try {
    const d = await (await fetch("/api/reindex", { method: "POST" })).json();
    watchInfo.textContent = `re-indexed ${d.reindexed} recording(s)`;
    loadEntities();
    loadRecordings();
  } catch { watchInfo.textContent = "re-index failed"; }
  finally { e.target.disabled = false; }
});

// ---- Speakers panel ----
const speakersBtn = document.getElementById("speakersBtn");
const speakersPanel = document.getElementById("speakerspanel");
const speakerList = document.getElementById("speakerList");

speakersBtn.addEventListener("click", () => {
  speakersPanel.hidden = !speakersPanel.hidden;
  if (!speakersPanel.hidden) loadSpeakers();
});

async function loadSpeakers() {
  try { speakers = (await (await fetch("/api/speakers")).json()).speakers || []; }
  catch { return; }
  if (speakersPanel.hidden) return;
  if (!speakers.length) {
    speakerList.innerHTML = `<span class="muted">No speakers yet. In a transcript, check segments where you recognize a voice and use "Tag as… → + new speaker".</span>`;
    return;
  }
  speakerList.innerHTML = speakers.map(sp =>
    `<div class="sp-row"><span class="sp-name">${escapeHtml(sp.name)}</span>` +
    `<span class="sp-meta">${sp.prints} voiceprint${sp.prints === 1 ? "" : "s"} · ${sp.tags} tagged</span>` +
    `<button class="chipbtn sp-del" data-id="${sp.id}">Delete</button></div>`).join("");
  speakerList.querySelectorAll(".sp-del").forEach(b => b.addEventListener("click", async () => {
    if (!confirm("Delete this speaker and all its voiceprints/tags?")) return;
    await fetch(`/api/speakers/${b.dataset.id}`, { method: "DELETE" });
    loadSpeakers();
  }));
}

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
loadSpeakers();
loadHealth();
loadGpu();
loadRecordings();
loadStats();
setInterval(loadRecordings, 4000);   // live status refresh
setInterval(loadStats, 4000);        // backlog progress
setInterval(loadHealth, 15000);
setInterval(loadGpu, 3000);          // GPU telemetry
