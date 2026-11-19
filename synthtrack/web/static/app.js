// SynthTrack web GUI – vanilla JS, no build step.
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const CONF = { high: [3, "vysoká"], medium: [2, "střední"], low: [1, "nízká"], none: [0, "žádná"] };
const confRank = (c) => (CONF[c.confidence] || [0])[0];
const state = { view: "home", jobId: null, tab: "cands", filter: "unreviewed", sort: "snr", focus: 0, mlMin: 0, confMin: 0, snrMin: 0,
  groupStars: true, starGroupsOpen: new Set(), cands: [], jobs: [], status: null, instruments: [], history: [] };

async function api(path, opts = {}) {
  const o = { headers: { "Content-Type": "application/json" }, ...opts };
  if (o.body && typeof o.body !== "string") o.body = JSON.stringify(o.body);
  const r = await fetch(path, o);
  const ct = r.headers.get("content-type") || "";
  const data = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error((data && data.detail) || r.statusText);
  return data;
}
// Desktop window (pywebview): native dialogs via window.pywebview.api; null in the browser.
const desk = () => (window.pywebview && window.pywebview.api) || null;
function toast(msg, err = false) {
  const t = $("#toast"); t.textContent = msg; t.className = "toast" + (err ? " err" : "");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), 4500);
}
const fmtDate = (ts) => ts ? new Date(ts * 1000).toLocaleString("cs-CZ") : "";
const fmtDur = (a, b) => { if (!a) return ""; const s = Math.round((b || Date.now() / 1000) - a); return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`; };
function raStr(ra) { if (ra == null) return "–"; let h = ra / 15; const H = Math.floor(h); let m = (h - H) * 60; const M = Math.floor(m); const S = (m - M) * 60; return `${String(H).padStart(2, "0")} ${String(M).padStart(2, "0")} ${S.toFixed(2).padStart(5, "0")}`; }
function decStr(d) { if (d == null) return "–"; const sg = d < 0 ? "-" : "+"; d = Math.abs(d); const D = Math.floor(d); let m = (d - D) * 60; const M = Math.floor(m); const S = (m - M) * 60; return `${sg}${String(D).padStart(2, "0")} ${String(M).padStart(2, "0")} ${S.toFixed(1).padStart(4, "0")}`; }

// ------------------------------------------------------------------ header / sidebar
async function refreshStatus() {
  try { state.status = await api("/api/status"); } catch { return; }
  const s = state.status;
  $("#ver").textContent = "v" + s.version;
  const chips = [];
  if (s.gpus.length) {
    for (const g of s.gpus) chips.push(`<span class="chip"><b>${esc(g.name.replace(/^(NVIDIA GeForce |NVIDIA |AMD )/, ""))}</b> · ${g.util_pct ?? "?"}% · ${((g.total_mb - g.free_mb) / 1024).toFixed(1)}/${(g.total_mb / 1024).toFixed(0)} GB</span>`);
  }
  chips.push(`<span class="chip ${s.backend.backend === "numpy" ? "warn" : "ok"}" title="${esc(s.backend.note)}">backend <b>${s.backend.backend === "numpy" ? "CPU" : (s.backend.label || "GPU")}</b>${s.backend.backend !== "numpy" ? ` · limit ${Math.round(s.gpu_memory_fraction * 100)}%` : ""}</span>`);
  for (const [k, lbl] of [["mpcorb", "MPCORB"], ["comets", "Komety"]]) {
    const c = s.catalogs[k];
    const txt = c.present ? `${c.age_days < 1 ? (c.age_days * 24).toFixed(0) + " h" : c.age_days.toFixed(0) + " d"}` : "chybí";
    chips.push(`<span class="chip click ${!c.present || c.stale ? "warn" : ""}" data-view="data">${lbl} <b>${txt}</b></span>`);
  }
  chips.push(`<span class="chip" title="${esc(s.ephemeris)}">efemerida <b>${esc((s.ephemeris || "").split(" ").slice(0, 2).join(" ").replace("astropy builtin", "ERFA"))}</b></span>`);
  chips.push(`<span class="chip ${s.astap ? "" : "warn"}" title="${esc(s.astap || "ASTAP nenalezen")}">ASTAP <b>${s.astap ? "OK" : "–"}</b></span>`);
  if (s.watch) chips.push(`<span class="chip click ${s.watch.running ? "ok" : ""}" data-view="watch">auto <b>${s.watch.running ? "běží" : "vyp."}</b></span>`);
  $("#chips").innerHTML = chips.join("");
}

async function refreshJobs() {
  try { state.jobs = await api("/api/jobs"); } catch { return; }
  const fin = state.jobs.filter((j) => ["done", "failed", "cancelled"].includes(j.status)).length;
  $("#del-finished").textContent = `Smazat hotové úlohy (${fin})`;
  $("#joblist").innerHTML = state.jobs.map((j) => `
    <div class="jobitem ${j.id === state.jobId && state.view === "job" ? "active" : ""}" data-job="${j.id}">
      <div class="t">${esc(j.name)}</div>
      <div class="m"><span class="pill ${j.status}">${statusCs(j.status)}</span>
        ${j.source === "auto" ? '<span class="pill auto">auto</span>' : ""}
        <span>${j.status === "running" ? Math.round(j.progress * 100) + " %" : fmtDur(j.started, j.finished)}</span>
        <span style="margin-left:auto">${fmtDate(j.created)}</span></div>
    </div>`).join("") || '<div class="empty">Zatím žádné úlohy</div>';
  if (state.view === "job") {
    const j = state.jobs.find((x) => x.id === state.jobId);
    if (j && (j.status === "running" || j.status === "queued")) renderJobHead(j);
    if (j && j.status === "done" && state._lastStatus !== "done") { state._lastStatus = "done"; openJob(j.id, true); }
    if (j) state._lastStatus = j.status;
  }
}
const statusCs = (s) => ({ queued: "ve frontě", running: "běží", done: "hotovo", failed: "chyba", cancelled: "zrušeno" }[s] || s);

// ------------------------------------------------------------------ routing
function go(view, arg) {
  state.view = view;
  if (view === "job") return openJob(arg);
  ({ home: viewHome, new: viewNew, watch: viewWatch, instruments: viewInstruments, data: viewData }[view] || viewHome)();
  refreshJobs();
}
document.addEventListener("click", (e) => {
  const v = e.target.closest("[data-view]");
  if (v) { e.preventDefault(); go(v.dataset.view); return; }
  const j = e.target.closest("[data-job]");
  if (j) { go("job", j.dataset.job); }
});
$("#del-finished").onclick = async (e) => {
  e.stopPropagation();
  if (!confirm("Smazat všechny dokončené, chybné a zrušené úlohy?")) return;
  const r = await api("/api/jobs/delete-finished", { method: "POST" });
  toast(`Smazáno ${r.deleted} úloh`); if (state.view === "job") go("home"); refreshJobs();
};

function viewHome() {
  $("#main").innerHTML = `<div class="panel"><h2>SynthTrack</h2>
  <p class="sub">Synthetic tracking planetek a komet. Začněte <a href="#" data-view="instruments">profilem přístroje</a>
  (ohnisko, pixel, adresář s master darky a flaty), potom vytvořte <a href="#" data-view="new">novou úlohu</a>
  nebo zapněte <a href="#" data-view="watch">automatický režim</a>.</p></div>`;
}

// ------------------------------------------------------------------ job detail
async function openJob(id, keepTab = false) {
  state.view = "job"; state.jobId = id; if (!keepTab) { state.tab = "cands"; state.focus = 0; }
  let job; try { job = await api(`/api/jobs/${id}`); } catch (e) { toast(e.message, true); return go("home"); }
  state._lastStatus = job.status;
  $("#main").innerHTML = `<div class="panel" id="jobhead"></div>
    <div class="tabs">${[["cands", "Kandidáti (pohyblivé)"], ["static", "Zdroje navíc (nepohyblivé)"], ["known", "Známé objekty"], ["summary", "Souhrn"], ["export", "Export"], ["log", "Log"]]
      .map(([k, l]) => `<div class="tab ${state.tab === k ? "active" : ""}" data-tab="${k}">${l}</div>`).join("")}</div>
    <div id="tabbody"></div>`;
  renderJobHead(job);
  document.querySelectorAll(".tab").forEach((t) => t.onclick = () => { state.tab = t.dataset.tab; openJob(id, true); });
  renderTab(job);
  refreshJobs();
}

function renderJobHead(job) {
  const h = $("#jobhead"); if (!h) return;
  const busy = job.status === "running" || job.status === "queued";
  h.innerHTML = `<div class="jobhead"><div>
      <h2>${esc(job.name)} <span class="pill ${job.status}">${statusCs(job.status)}</span></h2>
      ${job.field && job.field.label ? `<div class="sub">Zorné pole${job.field.total > 1 ? ` ${job.field.index}/${job.field.total}` : ""}: <b>${esc(job.field.label)}</b>${job.field.ra_deg != null ? ` · RA ${raStr(job.field.ra_deg)} · Dec ${decStr(job.field.dec_deg)}` : ""}</div>` : ""}
      <div class="sub">${job.n_frames || "?"} snímků · přístroj ${esc(job.instrument_id)} · ${esc(job.lights_dir)} · vytvořeno ${fmtDate(job.created)}</div>
      ${busy ? `<div class="sub">${esc(job.stage)}</div><div class="progress"><div style="width:${Math.round(job.progress * 100)}%"></div></div>` : ""}
      ${job.error ? `<div class="warnbox" style="color:var(--bad);border-color:#6b2b2b">${esc(job.error)}</div>` : ""}
      ${(job.warnings || []).map((w) => `<div class="warnbox">${esc(w)}</div>`).join("")}
      ${calibNote(job.params)}
      <div id="jcalib"></div>
    </div><div style="display:flex;gap:8px;align-items:flex-start">
      ${busy ? `<button class="btn" id="jcancel">Zrušit</button>` : `<button class="btn" id="jcalbtn">Kalibrace…</button><button class="btn" id="jrerun">Spustit znovu</button>`}
      <button class="btn danger" id="jdel" ${busy ? "disabled" : ""}>Smazat</button></div></div>`;
  const cb = $("#jcalbtn"); if (cb) cb.onclick = () => toggleJobCalib(job);
  if (job.status === "failed" && /^Master (dark|flat)/.test(job.error || "")) toggleJobCalib(job, true);
  const c = $("#jcancel"); if (c) c.onclick = async () => { await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); toast("Rušení…"); };
  const r = $("#jrerun"); if (r) r.onclick = async () => {
    const thr = prompt("Práh SNR (prázdné = beze změny):", job.params?.snr_threshold ?? "");
    const params = thr ? { snr_threshold: parseFloat(thr) } : {};
    const nj = await api(`/api/jobs/${job.id}/rerun`, { method: "POST", body: { params } });
    toast("Úloha zařazena"); go("job", nj.id);
  };
  const d = $("#jdel"); if (d) d.onclick = async () => { if (!confirm("Smazat úlohu včetně výsledků?")) return; try { await api(`/api/jobs/${job.id}`, { method: "DELETE" }); go("home"); } catch (e) { toast("Smazání selhalo: " + e.message); } };
}

// ------------------------------------------------------------- calibration choice
// job params calib_dark / calib_flat: "auto" | "none" | path to a master FITS
const CAL_NAMES = { dark: ["Master dark", "Bez darku (jen bias, je-li)"], flat: ["Master flat", "Bez master flatu (jen umělý flat)"] };
const baseName = (p) => String(p || "").split(/[\\/]/).pop();
function calibNote(params) {
  const parts = ["dark", "flat"].map((k) => {
    const v = params?.[`calib_${k}`];
    if (!v || v === "auto") return "";
    return `${k}: ${v === "none" ? "bez" : esc(baseName(v)) + " (ručně)"}`;
  }).filter(Boolean);
  return parts.length ? `<div class="sub">Kalibrace: ${parts.join(" · ")}</div>` : "";
}
function calibHtml(masters, params, pfx) {
  return `<div class="form">` + ["dark", "flat"].map((k) => {
    const cur = params?.[`calib_${k}`] || "auto";
    const list = masters.filter((m) => m.type === k);
    const other = masters.filter((m) => m.type === "unknown");
    const lab = (m) => `${baseName(m.path)} · ${m.exptime != null ? m.exptime + " s" : "bez EXPTIME"}${k === "flat" || m.filter ? " · " + (m.filter || "?") : ""}${m.temperature != null ? " · " + m.temperature + " °C" : ""} · ${m.width}×${m.height} bin ${m.xbin}`;
    const known = ["auto", "none", ...list.map((m) => m.path), ...other.map((m) => m.path)];
    const isOther = !known.includes(cur);
    const opt = (v, t) => `<option value="${esc(v)}" ${(!isOther && cur === v) || (isOther && v === "__file") ? "selected" : ""}>${esc(t)}</option>`;
    return `<label>${CAL_NAMES[k][0]}</label><div>
      <select id="${pfx}-${k}">${opt("auto", "Automaticky (podle hlaviček)")}${opt("none", CAL_NAMES[k][1])}
        ${list.length ? `<optgroup label="Mastery přístroje">${list.map((m) => opt(m.path, lab(m))).join("")}</optgroup>` : ""}
        ${other.length ? `<optgroup label="Nerozpoznaný typ">${other.map((m) => opt(m.path, lab(m))).join("")}</optgroup>` : ""}
        ${opt("__file", "Jiný soubor…")}</select>
      <div class="row" id="${pfx}-${k}-filerow" style="margin-top:6px;${isOther ? "" : "display:none"}"><input id="${pfx}-${k}-file" spellcheck="false" placeholder="cesta k FITS" value="${isOther ? esc(cur) : ""}"></div>
      ${k === "dark" ? `<div class="hint">Ručně zvolený dark se použije i při jiné nebo chybějící délce expozice (jen s varováním). Musí mít stejný rozměr a binning jako snímky.</div>` : ""}
    </div>`;
  }).join("") + `</div>`;
}
function bindCalib(pfx) {
  ["dark", "flat"].forEach((k) => {
    const sel = $(`#${pfx}-${k}`); if (!sel) return;
    sel.onchange = () => { $(`#${pfx}-${k}-filerow`).style.display = sel.value === "__file" ? "" : "none"; };
  });
}
function readCalib(pfx) {
  const out = {};
  for (const k of ["dark", "flat"]) {
    const sel = $(`#${pfx}-${k}`); if (!sel) continue;
    let v = sel.value;
    if (v === "__file") { v = $(`#${pfx}-${k}-file`).value.trim().replace(/^"(.*)"$/, "$1"); if (!v) throw new Error(`Zadejte cestu k souboru (${k})`); }
    out[`calib_${k}`] = v;
  }
  return out;
}
async function toggleJobCalib(job, forceOpen = false) {
  const box = $("#jcalib"); if (!box) return;
  if (box.innerHTML && !forceOpen) { box.innerHTML = ""; return; }
  let masters = [];
  try { masters = await api(`/api/instruments/${encodeURIComponent(job.instrument_id)}/masters`); } catch (e) { toast(e.message, true); }
  box.innerHTML = `<div class="sel" style="margin-top:10px"><b>Kalibrace pro nové spuštění</b>
      <div class="sub">Zvolené mastery se uloží do parametrů úlohy a použijí se i při dalším „Spustit znovu“.</div>
      ${calibHtml(masters, job.params, "jc")}
      <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
        <button class="btn primary" id="jc-run">Spustit znovu s touto kalibrací</button>
        <button class="btn" id="jc-raw">Spustit bez kalibrace</button>
        <button class="btn" id="jc-close">Zavřít</button></div></div>`;
  bindCalib("jc");
  const rerun = async (params) => {
    try { const nj = await api(`/api/jobs/${job.id}/rerun`, { method: "POST", body: { params } }); toast("Úloha zařazena"); go("job", nj.id); }
    catch (e) { toast(e.message, true); }
  };
  $("#jc-run").onclick = () => { try { rerun(readCalib("jc")); } catch (e) { toast(e.message, true); } };
  $("#jc-raw").onclick = () => rerun({ calib_dark: "none", calib_flat: "none" });
  $("#jc-close").onclick = () => { box.innerHTML = ""; };
}

async function renderTab(job) {
  const body = $("#tabbody");
  if (state.tab === "log") { body.innerHTML = `<pre class="log">${esc(await api(`/api/jobs/${job.id}/log`))}</pre>`; return; }
  if (job.status !== "done") { body.innerHTML = `<div class="empty">Výsledky budou k dispozici po dokončení úlohy.</div>`; return; }
  if (state.tab === "cands" || state.tab === "static") return renderCands(job);
  if (state.tab === "known") return renderKnown(job);
  if (state.tab === "summary") return renderSummary(job);
  if (state.tab === "export") return renderExport(job);
}

function candCounts() {
  const counts = { unreviewed: 0, accepted: 0, unsure: 0, rejected: 0, injected: 0, all: state.cands.filter((c) => c.status !== "injected").length };
  state.cands.forEach((c) => {
    counts[c.status] = (counts[c.status] || 0) + 1;
    if (c.status === "rejected") { const k = c.reviewed ? "rej_manual" : "rej_auto"; counts[k] = (counts[k] || 0) + 1;
      if (!c.reviewed && c.known) counts.rej_known = (counts.rej_known || 0) + 1; }
    if (isBorderline(c)) counts.borderline = (counts.borderline || 0) + 1;
    if (isSecondChance(c)) counts.second = (counts.second || 0) + 1;
  });
  return counts;
}
const FILTER_CZ = { unreviewed: "neprohlédnuté", accepted: "přijaté", unsure: "nejisté", rej_manual: "zamítnuté mnou", rej_auto: "zamítnuté automaticky", all: "vše", injected: "injekce", borderline: "hraniční skóre", rej_known: "zamítnuté známé", second: "druhá šance" };
function updateCounts() {
  tabLabels();
  const counts = candCounts();
  document.querySelectorAll("[data-f]").forEach((b) => { b.textContent = `${FILTER_CZ[b.dataset.f]} (${counts[b.dataset.f] || 0})`; });
  const u = $("#undo"); if (u) u.disabled = !undoable(state.jobs.find((j) => j.id === state.jobId) || { id: state.jobId });
}
// moving candidates (synthetic tracking) and static sources ("zdroj navíc": catalogue comparison,
// difference image, glows, supernovae) are reviewed in separate tabs – the static ones of a dense
// field must not bury the movers
const isStatic = (c) => c.source === "twilight";
function tabCands() { return (state.allCands || []).filter((c) => isStatic(c) === (state.tab === "static")); }
function tabLabels() {
  const n = (st) => (state.allCands || []).filter((c) => isStatic(c) === st && c.status === "unreviewed").length;
  document.querySelectorAll(".tab").forEach((t) => {
    if (t.dataset.tab === "cands") t.textContent = `Kandidáti (pohyblivé) · ${n(false)}`;
    if (t.dataset.tab === "static") t.textContent = `Zdroje navíc (nepohyblivé) · ${n(true)}`;
  });
}
async function renderCands(job) {
  state.allCands = await api(`/api/jobs/${job.id}/candidates`);
  // a job of only "zdroj navíc" (2–6 frames) opens on that tab
  if (state.tab === "cands" && !state.allCands.some((c) => !isStatic(c)) && state.allCands.some(isStatic)) {
    state.tab = "static";
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === "static"));
  }
  state.cands = tabCands();
  state.orientSky = skyTransform(state.allCands);
  tabLabels();
  renderCandsView(job);
}
function renderCandsView(job) {
  const counts = candCounts();
  const filters = [["unreviewed", "neprohlédnuté"], ["accepted", "přijaté"], ["unsure", "nejisté"], ["rej_manual", "zamítnuté mnou"], ["rej_auto", "zamítnuté automaticky"], ["all", "vše"]];
  if (counts.injected) filters.push(["injected", "injekce"]);
  if (counts.borderline) filters.push(["borderline", "hraniční skóre"]);
  // identified known objects the vetting rejected – the same rules lose unknown ones (no need to
  // page through 1000+ automatic rejections to find them)
  if (counts.rej_known) filters.push(["rej_known", "zamítnuté známé"]);
  if (counts.second) filters.push(["second", "druhá šance"]);
  const hasMl = state.cands.some((c) => c.ml_score != null);
  $("#tabbody").innerHTML = `<div class="sticky-tools"><div class="toolbar">
      <div class="seg">${filters.map(([k, l]) => `<button data-f="${k}" class="${state.filter === k ? "active" : ""}">${l} (${counts[k] || 0})</button>`).join("")}</div>
      <select id="sort"><option value="snr">řadit: SNR</option><option value="conf">řadit: spolehlivost</option><option value="ml">řadit: skóre klasifikátoru</option><option value="rate">řadit: rychlost</option><option value="known">řadit: neznámé první</option></select>
      ${hasMl ? `<label class="sl" title="Skrýt kandidáty, které klasifikátor považuje za falešné (nic se nemaže)">ML ≥ <select id="mlmin">${[0, 0.02, 0.05, 0.1, 0.2, 0.5].map((v) => `<option value="${v}" ${state.mlMin === v ? "selected" : ""}>${v ? Math.round(v * 100) + " %" : "vše"}</option>`).join("")}</select></label>` : ""}
      <label class="sl" title="Skrýt kandidáty pod tímto SNR (nic se nemaže). V hustých polích je většina falešných těsně nad prahem.">SNR ≥ <select id="snrmin">${[0, 6.5, 7, 7.5, 8, 10].map((v) => `<option value="${v}" ${state.snrMin === v ? "selected" : ""}>${v || "vše"}</option>`).join("")}</select></label>
      <label class="sl" title="Skrýt kandidáty s nižší spolehlivostí (nic se nemaže)">spolehlivost ≥ <select id="confmin">${[[0, "vše"], [1, "nízká"], [2, "střední"], [3, "vysoká"]].map(([v, l]) => `<option value="${v}" ${state.confMin === v ? "selected" : ""}>${l}</option>`).join("")}</select></label>
      ${state.filter === "rej_auto" ? `<label class="sl" title="Opakování téže statické hvězdy nebo jednoho plošného artefaktu zobrazit jako rozbalitelnou skupinu. Žádná detekce se nemaže."><input type="checkbox" id="group-stars" ${state.groupStars ? "checked" : ""}> seskupit opakované artefakty</label>` : ""}
      <button class="btn small" id="ml-train" title="Naučit klasifikátor na všech označených kandidátech (Přijato/Zamítnuto), známých objektech a injekcích">Naučit klasifikátor</button>
      <button class="btn small" id="undo" title="Vrátit poslední označení (Z nebo Ctrl+Z)" ${undoable(job) ? "" : "disabled"}>↶ Zpět</button>
      <span class="keys"><kbd>J</kbd>/<kbd>K</kbd> pohyb · <kbd>A</kbd> přijmout · <kbd>R</kbd> zamítnout · <kbd>U</kbd> nejisté · <kbd>Z</kbd> zpět · <kbd>P</kbd> pauza · <kbd>X</kbd> kříž · <kbd>D</kbd> DSS · <kbd>E</kbd> efemerida · <kbd>←</kbd>/<kbd>→</kbd> snímek</span>
    </div>${animToolbar()}</div><div id="cards" class="${state.anim.cross ? "" : "nocross"}"></div>`;
  bindAnimToolbar();
  applyOrient();
  $("#sort").value = state.sort;
  $("#sort").onchange = (e) => { state.sort = e.target.value; drawCards(job); };
  $("#snrmin").onchange = (e) => { state.snrMin = +e.target.value; state.focus = 0; drawCards(job); };
  $("#confmin").onchange = (e) => { state.confMin = +e.target.value; state.focus = 0; drawCards(job); };
  if ($("#mlmin")) $("#mlmin").onchange = (e) => { state.mlMin = +e.target.value; state.focus = 0; drawCards(job); };
  if ($("#group-stars")) $("#group-stars").onchange = (e) => { state.groupStars = e.target.checked; state.focus = 0; drawCards(job); };
  $("#ml-train").onclick = async () => {
    const b = $("#ml-train"); b.disabled = true; b.textContent = "Učím…";
    try { const r = await api("/api/classifier/train", { method: "POST" }); alert(r.report); renderCands(job); }
    catch (e) { alert(e.message); b.disabled = false; b.textContent = "Naučit klasifikátor"; }
  };
  document.querySelectorAll("[data-f]").forEach((b) => b.onclick = () => { state.filter = b.dataset.f; state.focus = 0; renderCandsView(job); });
  $("#undo").onclick = () => undoStatus(job);
  drawCards(job);
}

// set aside by the injection ranking between the main and the looser cut, not yet looked at
// "Druhá šance": automatically rejected ONLY for a nearby star / lower throughput, but strong and
// consistent like a real mover. CTA-S1 320 (dense fields, 50 % of the found injections lost in
// vetting, mostly "u hvězdy"): 11–26 per field, about half of them injections – a short list
// instead of 1000+ rejections, so that a discovery beside a star is not lost.
const STAR_ONLY_PREFIX = ["u hvězdy", "reziduum hvězdy", "pomalý u hvězdy", "snížená propustnost", "hvězda z katalogu UCAC4", "také jako zdroj navíc"];
function isSecondChance(c) {
  if (c.status !== "rejected" || c.reviewed || c.source === "twilight" || c.injected != null) return false;
  const fl = c.flags || [];
  if (fl.includes("těsně pod prahem")) return fl.length === 1;   // passed every check, SNR just below the threshold
  if (!fl.length || !fl.every((f) => STAR_ONLY_PREFIX.some((p) => f.startsWith(p)))) return false;
  const seg = c.seg_snr || [];
  const e = seg.length ? c.snr / Math.sqrt(seg.length) : 0;
  return c.snr >= 8 && e > 0 && Math.min(...seg) / e >= 0.2 && (c.median_ratio || 0) >= 0.6 && (c.chi2 ?? 9) <= 2.5;
}
function isBorderline(c) {
  return c.status === "rejected" && !c.reviewed && (c.flags || []).includes("hraniční skóre podle injekcí");
}

// colour of a card: only a human decision is "zamítnuto" (red); automatic set-asides stay neutral,
// the borderline band (between the two injection cuts) is amber – it waits for a decision
function cardState(c) {
  if (c.status === "rejected" && !c.reviewed) return isBorderline(c) ? "border" : "auto";
  return c.status;
}
const human = (c, k) => c.status === k && (k !== "rejected" || !!c.reviewed);

function visibleCands() {
  const fOk = (c) => state.filter === "all" ? c.status !== "injected"
    : state.filter === "borderline" ? isBorderline(c)
    : state.filter === "rej_auto" ? c.status === "rejected" && !c.reviewed
    : state.filter === "rej_known" ? c.status === "rejected" && !c.reviewed && !!c.known
    : state.filter === "second" ? isSecondChance(c)
    : state.filter === "rej_manual" ? c.status === "rejected" && !!c.reviewed
    : state.filter === "rejected" ? c.status === "rejected" : c.status === state.filter;
  let l = state.cands.filter((c) => fOk(c)
    && (!state.mlMin || c.ml_score == null || c.ml_score >= state.mlMin || c.status === "accepted")
    && (!state.confMin || confRank(c) >= state.confMin || c.status === "accepted")
    && (!state.snrMin || c.snr >= state.snrMin || c.status === "accepted"));
  if (state.sort === "snr") l.sort((a, b) => b.snr - a.snr);
  if (state.sort === "conf") l.sort((a, b) => confRank(b) - confRank(a) || (b.ml_score ?? 0) - (a.ml_score ?? 0) || b.snr - a.snr);
  if (state.sort === "ml") l.sort((a, b) => (b.ml_score ?? -1) - (a.ml_score ?? -1) || b.snr - a.snr);
  if (state.sort === "rate") l.sort((a, b) => (b.rate_arcsec_min || 0) - (a.rate_arcsec_min || 0));
  if (state.sort === "known") l.sort((a, b) => (!!a.known - !!b.known) || b.snr - a.snr);
  l.forEach((c) => delete c._starGroup);
  if (state.filter === "rej_auto" && state.groupStars) {
    const groups = new Map(), plain = [];
    for (const c of l) {
      const flags = c.flags || [];
      const star = c.static_star_x != null && c.static_star_y != null && flags.some((f) => f.includes("hvězd"));
      const transient = flags.includes("jednosnímkový artefakt") && flags.includes("nekonzistentní snímky")
        && (flags.includes("nekonzistentní segmenty") || flags.includes("bez směru pohybu"));
      if (!star && !transient) { plain.push(c); continue; }
      const key = star ? `star:${Math.round(c.static_star_x)}:${Math.round(c.static_star_y)}`
        : `transient:${Math.floor(c.x / 256)}:${Math.floor(c.y / 256)}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(c);
    }
    l = plain;
    for (const [key, members] of groups) {
      if (members.length < 2 || state.starGroupsOpen.has(key)) {
        members.forEach((c) => { c._starGroup = { key, count: members.length, open: true }; l.push(c); });
      } else {
        const c = members.reduce((a, b) => b.snr > a.snr ? b : a);
        c._starGroup = { key, count: members.length, open: false }; l.push(c);
      }
    }
    if (state.sort === "snr") l.sort((a, b) => b.snr - a.snr);
  }
  return l;
}

const CROSS = `<svg class="cross" viewBox="0 0 210 210"><g stroke="rgba(255,255,255,.6)" stroke-width="1.5">
  <line x1="105" y1="70" x2="105" y2="92"/><line x1="105" y1="118" x2="105" y2="140"/>
  <line x1="70" y1="105" x2="92" y2="105"/><line x1="118" y1="105" x2="140" y2="105"/></g></svg>`;

// a known variable right at a static source: a small survey map with the source in the centre (yellow
// ring) and the VSX stars as pink crosses. The map (north up, east left) is turned with the
// candidate's WCS matrix into exactly the orientation and scale of the cards' own cut-outs
// (screen = T · pixel; pixel = M⁻¹ · sky) – the same sky spot sits at the same place in all four.
function vsxThumb(job, c) {
  const scale = +(job.summary?.pixel_scale || 0);
  const S = c.images?.frames?.size || 64;
  const fov = c.vsx_map_fov || Math.max(90, Math.min(900, S * (scale || 6) * 1.45)) / 3600;   // degrees
  const src = c.images?.vsx_map ? `/api/jobs/${job.id}/img/${c.images.vsx_map}`   // saved by the pipeline
    : `/api/jobs/${job.id}/candidates/${c.id}/sky?survey=DSS2r&fov=${fov.toFixed(5)}`;
  const cap = vsxCaption(c, c.vsx_near || []);
  return `<div class="thumb vsxmap" data-vsxcid="${c.id}"><div class="vm-clip"><img src="${src}" loading="lazy" onerror="this.replaceWith(Object.assign(document.createElement('div'), {className: 'empty', textContent: 'mapa z DSS nedostupná (offline?)'}))"></div>
    <svg viewBox="0 0 190 190"></svg>
    <div class="cap" title="DSS2 červeně, natočeno podle WCS jako ostatní náhledy. Žlutý kroužek = zdroj, růžový křížek = proměnná VSX (kliknutím do VSX).">${cap}</div></div>`;
}
function vsxCaption(c, list) {
  const v0 = list[0];
  if (!v0) return `<span class="sub">DSS2 · bez známé proměnné</span>`;
  const sep = v0.sep_arcsec ?? Math.round(Math.hypot(v0.de, v0.dn) * 36000) / 10;
  return `<a href="${esc(v0.url)}" target="_blank" rel="noopener">${esc(v0.name)}</a>`
    + `<br><span class="vtype">${esc([v0.type, v0.range].filter(Boolean).join(" · ") || "typ neuveden")} · ${sep}″</span>`
    + (list.length > 1 ? ` <span class="sub">(+${list.length - 1})</span>` : "");
}
function vsxThumbOrient() {
  document.querySelectorAll(".thumb.vsxmap[data-vsxcid]").forEach((el) => {
    const c = (state.cands || []).find((x) => x.id === +el.dataset.vsxcid);
    if (!c) return;
    const job = state.jobs.find((j) => j.id === state.jobId) || {};
    const scale = +(job.summary?.pixel_scale || 0);
    const S = c.images?.frames?.size || 64;
    const fov = c.vsx_map_fov || Math.max(90, Math.min(900, S * (scale || 6) * 1.45)) / 3600;
    const T = currentT();
    const M = c.sky_matrix;
    let toScr;                                        // (east, north) in degrees → card pixels (0–190)
    const img = el.querySelector("img");
    const det = M ? M[0][0] * M[1][1] - M[0][1] * M[1][0] : 0;
    if (M && Math.abs(det) > 0) {
      const Mi = [[M[1][1] / det, -M[0][1] / det], [-M[1][0] / det, M[0][0] / det]];       // px per degree
      const TM = [[T[0][0] * Mi[0][0] + T[0][1] * Mi[1][0], T[0][0] * Mi[0][1] + T[0][1] * Mi[1][1]],
                  [T[1][0] * Mi[0][0] + T[1][1] * Mi[1][0], T[1][0] * Mi[0][1] + T[1][1] * Mi[1][1]]];
      const k = -fov / S;                             // map pixel (q, east left / north up) → card pixel
      if (img) img.style.transform = `matrix(${k * TM[0][0]},${k * TM[1][0]},${k * TM[0][1]},${k * TM[1][1]},0,0)`;
      toScr = (de, dn) => [95 + 190 / S * (TM[0][0] * de + TM[0][1] * dn), 95 + 190 / S * (TM[1][0] * de + TM[1][1] * dn)];
    } else {                                          // no WCS matrix: the map as it is, north up
      if (img) img.style.transform = "none";
      toScr = (de, dn) => [95 - de / fov * 190, 95 - dn / fov * 190];
    }
    // the variables around: all VSX stars of the cut-out (fetched once per card, cached by the server),
    // crosses for the ≤ 5 nearest the source that lie inside the card – also when not right on it
    const all = el._vsx || c.vsx_near || [];
    const near = all.map((v) => [v, toScr(v.de, v.dn), Math.hypot(v.de, v.dn)])
      .filter(([, [x, y]]) => x >= 4 && x <= 186 && y >= 4 && y <= 186)
      .sort((a, b) => a[2] - b[2]).slice(0, 5);
    const svg = el.querySelector("svg");
    svg.innerHTML = `<circle cx="95" cy="95" r="7" fill="none" stroke="rgba(255,196,40,.95)" stroke-width="1.5"/>`
      + near.map(([v, [x, y], d]) => {
        const t = `${v.name}${v.type ? " · " + v.type : ""}${v.range ? " · " + v.range : ""}${v.period ? " · P " + v.period + " d" : ""} · ${(d * 3600).toFixed(1)}″`;
        return `<a href="${esc(v.url)}" target="_blank" rel="noopener"><title>${esc(t)}</title><circle cx="${x}" cy="${y}" r="6" fill="transparent"/>
          <path d="M${x - 8} ${y}h4.5M${x + 3.5} ${y}h4.5M${x} ${y - 8}v4.5M${x} ${y + 3.5}v4.5" stroke="#ff5ad2" stroke-width="1.6"/></a>`;
      }).join("");
    const cap = el.querySelector(".cap");
    if (cap && (el._vsx || c.vsx_near)) cap.innerHTML = vsxCaption(c, near.map(([v]) => v));
    if (!el._vsx && !el._vsxLoading && img) {
      const go = () => {
        if (el._vsxLoading) return;
        el._vsxLoading = true;
        fetch(`/api/jobs/${state.jobId}/candidates/${c.id}/vsx?fov=${fov.toFixed(5)}`)
          .then((r) => r.ok ? r.json() : null).then((d) => { if (d) { el._vsx = d.stars || []; vsxThumbOrient(); } })
          .catch(() => { /* VSX / ASTAP unavailable: the card keeps what the pipeline found */ });
      };
      if (img.complete && img.naturalWidth) go(); else img.addEventListener("load", go, { once: true });
    }
  });
}

function drawCards(job) {
  const list = visibleCands();
  if (!list.length) { $("#cards").innerHTML = `<div class="empty">Žádní kandidáti v tomto filtru.</div>`; return; }
  state.focus = Math.min(state.focus, list.length - 1);
  const img = (c, k) => c.images?.[k] ? `/api/jobs/${job.id}/img/${c.images[k]}` : "";
  $("#cards").innerHTML = list.map((c, i) => {
    const badge = c.known ? `<span class="badge">${esc(c.known.name)} · O−C ${c.known.o_c_arcsec}″</span>`
      : (c.ra != null && c.auto_status === "candidate" ? `<span class="badge new">neznámý objekt</span>` : "");
    const flags = (c.flags || []).map((f) => `<span class="badge flag">${esc(f)}</span>`).join(" ");
    const starGroup = c._starGroup && c._starGroup.count > 1 ? `<button class="btn small star-group" data-star-group="${esc(c._starGroup.key)}" title="Všechny detekce zůstávají uložené">${c._starGroup.open ? "sbalit" : `opakovaný artefakt ×${c._starGroup.count} · rozbalit`}</button>` : "";
    const autoBadge = cardState(c) === "border" ? `<span class="badge auto border" title="Mezi hlavním a volnějším řezem podle injekcí. Program ho jen odložil – čeká na vaše rozhodnutí.">odloženo automaticky – rozhodněte</span> `
      : cardState(c) === "auto" ? `<span class="badge auto" title="Zamítl program, ne vy. Tlačítkem Přijmout / Zamítnout rozhodnete sami.">zamítnuto automaticky</span> ` : "";
    return `<div class="card st-${cardState(c)} ${i === state.focus ? "focus" : ""}" data-cid="${c.id}" data-i="${i}">
      <div class="card-head"><span class="id">#${c.id}</span><span>${autoBadge}${badge} ${flags} ${starGroup}${c.injected != null ? ` <span class="badge">injekce #${c.injected}</span>` : ""}</span><span class="snr">${c.confidence ? `<span class="badge conf conf-${c.confidence}" title="${esc((c.confidence_reasons || []).length ? "Proč ne vyšší: " + c.confidence_reasons.join("; ") : "Splňuje všechna kritéria")}">spolehlivost: ${CONF[c.confidence][1]}</span> ` : ""}${c.ml_score != null ? `<span class="badge ml" title="Pravděpodobnost skutečného objektu podle klasifikátoru">ML ${Math.round(c.ml_score * 100)} %</span> ` : ""}SNR ${c.snr.toFixed(1)}</span></div>
      <div class="card-body">
        ${c.preview_skipped
          ? `<div class="thumb"><div class="empty">${esc(c.preview_skipped)}</div><div class="cap">číselný záznam zůstal uložen</div></div>`
          : c.images?.frames
          ? `<div class="thumb"><canvas class="anim" width="190" height="190" data-cid="${c.id}"></canvas><span class="lbl">pohyb</span><span class="cnt"></span><div class="cap">animace, hvězdy v klidu</div></div>`
          : c.images?.motion
          ? `<div class="thumb"><img src="${img(c, "motion")}" loading="lazy"><span class="lbl">pohyb</span><div class="cap">animace, hvězdy v klidu</div></div>`
          : `<div class="thumb"><img src="${img(c, "gif")}" loading="lazy"><span class="lbl">podstacky</span>${CROSS}<div class="cap">animace (${c.seg_snr.length} části)</div></div>`}
        ${c.preview_skipped ? "" : `<div class="thumb"><img src="${img(c, "mean")}" loading="lazy">${CROSS}<div class="cap">průměrný stack</div></div>
        <div class="thumb"><img src="${img(c, "median")}" loading="lazy">${CROSS}<div class="cap">mediánový stack</div></div>`}
        ${isStatic(c) && c.ra != null && !c.preview_skipped ? vsxThumb(job, c) : ""}
        ${c.images?.abv ? `<div class="thumb"><img src="${img(c, "abv")}" loading="lazy">${CROSS}<div class="cap" title="Snímky minus statická obloha (medián v čase), posunuté podle pohybu: hvězdy se odečtou, zůstane jen to, co se pohybuje se stopou. U velmi pomalých objektů (&lt; 2 FWHM za sérii) se objekt částečně odečte sám.">pohybová vrstva (bez hvězd)</div></div>` : ""}
        ${c.images?.snr ? `<div class="thumb"><img src="${img(c, "snr")}" loading="lazy">${CROSS}<div class="cap">mapa SNR podél stopy</div></div>` : ""}
        <div class="facts">
          <span class="k">Rychlost</span><span class="v">${c.rate_arcsec_min != null ? c.rate_arcsec_min.toFixed(3) + "″/min" : "–"}</span>
          <span class="k">PA</span><span class="v">${c.pa_deg != null ? c.pa_deg.toFixed(1) + "°" : "–"}</span>
          <span class="k">RA</span><span class="v">${raStr(c.ra)}</span>
          <span class="k">Dec</span><span class="v">${decStr(c.dec)}</span>
          <span class="k">Mag</span><span class="v" title="Přibližná jasnost: nulový bod z hvězd UCAC4 (APASS), jinak z identifikovaných planetek pole (předpovědi MPC, škála V)">${c.mag != null ? "≈" + c.mag.toFixed(1) + " (změřeno)" + (c.known?.pred_mag != null ? ", předp. " + c.known.pred_mag : "") : (c.known?.pred_mag != null ? "≈" + c.known.pred_mag + " (předpověď)" : "–")}</span>
          <span class="k">x, y</span><span class="v">${c.x.toFixed(1)}, ${c.y.toFixed(1)}</span>
          <span class="k">Snímky</span><span class="v">${c.frames_used}/${job.n_frames}</span>
          <span class="k">Seg SNR</span><span class="v">${c.seg_snr.map((v) => v.toFixed(1)).join(" / ")}</span>
          <span class="k">Medián/průměr</span><span class="v">${c.median_ratio.toFixed(2)}</span>
          <span class="k">χ² snímků</span><span class="v">${(c.chi2 ?? 0).toFixed(2)}</span>
          <span class="k">SNR pro −v</span><span class="v">${(c.snr_reverse ?? 0).toFixed(1)}</span>
          <span class="k">FWHM hvězd</span><span class="v">${c.fwhm != null ? c.fwhm.toFixed(2) + " px" : "–"}</span>
          ${c.obj_fwhm_scale != null ? `<span class="k">FWHM objektu</span><span class="v" title="ze dvou měřítek (σ a 2σ profilu hvězdy), vůči okolním hvězdám – vidí i komu; momenty v malém okně: ${c.obj_fwhm != null ? c.obj_fwhm.toFixed(2) + " px" : "?"}">${c.obj_fwhm_scale >= 50 ? "rozlehlý" : c.obj_fwhm_scale.toFixed(2) + " px"} (${c.fwhm && c.obj_fwhm_scale < 50 ? Math.round(100 * c.obj_fwhm_scale / c.fwhm) : "?"} %)</span>`
            : c.obj_fwhm != null ? `<span class="k">FWHM objektu</span><span class="v" title="Δχ² profil hvězdy vs. nejlepší šířka: ${c.sharp_dchi2}">${c.obj_fwhm.toFixed(2)} px (${c.fwhm ? Math.round(100 * c.obj_fwhm / c.fwhm) : "?"} %)</span>` : ""}
          ${c.throughput_min != null ? `<span class="k">Propustnost min</span><span class="v">${Math.round(c.throughput_min * 100)} %</span>` : ""}
        </div>
      </div>
      <div class="card-actions">
        <button class="btn small ${human(c, "accepted") ? "ok" : ""}" data-set="accepted">Přijmout</button>
        <button class="btn small ${human(c, "unsure") ? "warnb" : ""}" data-set="unsure">Nejisté</button>
        <button class="btn small ${human(c, "rejected") ? "badb" : ""}" data-set="rejected">Zamítnout</button>
        <span class="act-gap" aria-hidden="true"></span>
        <button class="btn small" data-sky title="Porovnat s přehlídkou DSS / 2MASS (D)">DSS</button>
        <button class="btn small" data-eph title="Přibližná poloha na 1–48 h dopředu (lineární aproximace pohybu) (E)">Efemerida</button>
        ${c.status !== "unreviewed" && (c.status !== "rejected" || c.reviewed) ? `<button class="btn small" data-set="unreviewed" title="Zrušit označení (zpět mezi neprohlédnuté)">Zrušit označení</button>` : ""}
        <input placeholder="trkSub" value="${esc(c.trksub || "")}" maxlength="7" data-trk>
      </div></div>`;
  }).join("");
  vsxThumbOrient();
  document.querySelectorAll(".card").forEach((card) => {
    card.onclick = () => { state.focus = +card.dataset.i; markFocus(); };
    card.querySelectorAll("[data-set]").forEach((b) => b.onclick = (e) => { e.stopPropagation(); setStatus(job, +card.dataset.cid, b.dataset.set); });
    const sg = card.querySelector("[data-star-group]");
    if (sg) sg.onclick = (e) => { e.stopPropagation(); const k = sg.dataset.starGroup; state.starGroupsOpen.has(k) ? state.starGroupsOpen.delete(k) : state.starGroupsOpen.add(k); drawCards(job); };
    const sk = card.querySelector("[data-sky]");
    if (sk) sk.onclick = (e) => { e.stopPropagation(); openSky(job, state.cands.find((x) => x.id === +card.dataset.cid)); };
    const ep = card.querySelector("[data-eph]");
    if (ep) ep.onclick = (e) => { e.stopPropagation(); openEphem(job, state.cands.find((x) => x.id === +card.dataset.cid)); };
    const t = card.querySelector("[data-trk]");
    t.onchange = () => api(`/api/jobs/${job.id}/candidates/${card.dataset.cid}`, { method: "PATCH", body: { trksub: t.value } });
  });
  animAttach(job);
}

// ------------------------------------------------------------------ animation viewer
// Frames come as one 8-bit strip per candidate (q = C + B·asinh(x/A), x in σ);
// sub-stacks, black/white level and gamma are computed here, so they can be
// changed instantly for all candidates.
const ANIM_DEF = { k: 3, black: -1.5, white: 8, gamma: 1.0, speed: 350, play: true, align: true, cross: true,
  orient: "sky", fix: true, skipObs: true, stack: "mean" };
state.anim = Object.assign({}, ANIM_DEF, (() => { try { return JSON.parse(localStorage.getItem("st.anim") || "{}"); } catch { return {}; } })());
state.anim.play = true;
state.anim.k = ANIM_DEF.k;     // every start begins with 3 images; other settings are remembered
const saveAnim = () => { try { localStorage.setItem("st.anim", JSON.stringify(state.anim)); } catch { /* private window */ } };
const animCache = new Map();
const animLive = new Set();
let animObs = null, animTimer = null, animTick = 0;

function orientSelect() {
  const A = state.anim, sky = !!state.orientSky;
  const o = [["sky", "sever nahoře, východ vlevo"], ["fits", "jako FITS (DS9)"], ["raw", "surově (řádky shora)"]];
  return `<label class="sl" title="Orientace všech náhledů. „Sever nahoře“ potřebuje WCS (souřadnice kandidátů).">orientace <select id="an-orient">${o.map(([k, l]) =>
    `<option value="${k}" ${A.orient === k ? "selected" : ""} ${k === "sky" && !sky ? "disabled" : ""}>${l}</option>`).join("")}</select></label>`;
}

function animToolbar() {
  const A = state.anim;
  if (!state.cands.some((c) => c.images?.frames)) {
    return `<div class="toolbar anim-tools">${orientSelect()}<label class="sl"><input type="checkbox" id="an-cross" ${A.cross ? "checked" : ""}> zaměřovací kříž</label><span class="sub">Animace: tato úloha byla zpracována starší verzí – pro nastavitelnou animaci (počet obrázků, jas, kontrast) ji spusťte znovu.</span></div>`;
  }
  const ks = [2, 3, 4, 5, 6, 0];
  const sl = (id, lbl, min, max, step, v, unit, tip) => `<label class="sl" title="${tip}">${lbl} <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${v}"><span id="${id}-v" class="mono">${v}${unit}</span></label>`;
  return `<div class="toolbar anim-tools"><span class="sub">Animace:</span>
    <div class="seg" id="an-k" title="Počet obrázků animace; každý je průměrem části snímků (vše = jednotlivé snímky)">${ks.map((k) => `<button data-k="${k}" class="${A.k === k ? "active" : ""}">${k || "vše"}</button>`).join("")}</div>
    <div class="seg" id="an-stack" title="Jak se skládají snímky jednoho obrázku animace: průměr (nejmenší šum) nebo medián (potlačí kosmiku, stopy družic a jednosnímkové artefakty; potřebuje aspoň 3 snímky na obrázek)">${[["mean", "průměr"], ["median", "medián"]].map(([k, l]) => `<button data-st="${k}" class="${(A.stack || "mean") === k ? "active" : ""}">${l}</button>`).join("")}</div>
    ${sl("an-black", "černá", -5, 3, 0.1, A.black, " σ", "Úroveň černé v násobcích šumu jednoho obrázku animace (0 = pozadí)")}
    ${sl("an-white", "bílá", 2, 40, 0.5, A.white, " σ", "Úroveň bílé v násobcích šumu – vyšší = méně vytažené pozadí")}
    ${sl("an-gamma", "gama", 0.4, 3, 0.05, A.gamma, "", "1 = lineárně, >1 zvýrazní slabé objekty (vytáhne i šum)")}
    ${sl("an-speed", "krok", 80, 1500, 20, A.speed, " ms", "Doba zobrazení jednoho obrázku")}
    <label class="sl" title="Snímky v každém obrázku se složí na pohyb objektu: objekt je ostrý, hvězdy mírně protažené, ale stojí na místě a objekt se mezi obrázky posouvá"><input type="checkbox" id="an-align" ${A.align ? "checked" : ""}> skládat na pohyb</label>
    <label class="sl" title="Zaměřovací kříž v animaci i ve stacích (X)"><input type="checkbox" id="an-cross" ${A.cross ? "checked" : ""}> kříž</label>
    <label class="sl" title="Lokálně dorovná hvězdy každého snímku (globální sesazení může v rohu pole nebo u částečně zakrytého snímku ujet o 1–2 px – animace pak poskakuje)"><input type="checkbox" id="an-fix" ${A.fix ? "checked" : ""}> dorovnat hvězdy</label>
    <label class="sl" title="Vynechá snímky, kde je místo kandidáta zakryté nebo silně zeslabené (propustnost pod 50 %)"><input type="checkbox" id="an-skip" ${A.skipObs ? "checked" : ""}> bez zakrytých</label>
    ${orientSelect()}
    <button class="btn small" id="an-play" title="Pauza / přehrát (P)">${A.play ? "❚❚" : "▶"}</button>
    <button class="btn small" id="an-reset" title="Výchozí nastavení animace">výchozí</button></div>`;
}

// ------------------------------------------------------------ orientation
// Dihedral transform T (screen ← image) chosen from the candidates' RA/Dec so that
// north is up and east left; "fits" = rows bottom-up (as DS9), "raw" = rows top-down.
function skyTransform(cands) {
  const p = cands.filter((c) => c.ra != null && c.dec != null);
  if (p.length < 3) return null;
  const dec0 = p.reduce((s, c) => s + c.dec, 0) / p.length * Math.PI / 180, ra0 = p[0].ra;
  const rows = p.map((c) => { let dr = c.ra - ra0; if (dr > 180) dr -= 360; if (dr < -180) dr += 360;
    return [c.x, c.y, dr * Math.cos(dec0), c.dec]; });
  const fit = (k) => {               // least squares q = a + b x + c y
    let n = 0, sx = 0, sy = 0, sq = 0, sxx = 0, syy = 0, sxy = 0, sxq = 0, syq = 0;
    for (const r of rows) { n++; sx += r[0]; sy += r[1]; sq += r[k]; sxx += r[0] * r[0]; syy += r[1] * r[1]; sxy += r[0] * r[1]; sxq += r[0] * r[k]; syq += r[1] * r[k]; }
    const mx = sx / n, my = sy / n, mq = sq / n;
    const cxx = sxx / n - mx * mx, cyy = syy / n - my * my, cxy = sxy / n - mx * my, cxq = sxq / n - mx * mq, cyq = syq / n - my * mq;
    const det = cxx * cyy - cxy * cxy; if (Math.abs(det) < 1e-9) return null;
    return [(cxq * cyy - cyq * cxy) / det, (cyq * cxx - cxq * cxy) / det];
  };
  const xi = fit(2), eta = fit(3); if (!xi || !eta) return null;
  const M = [[xi[0], xi[1]], [eta[0], eta[1]]];        // (ξ east, η north) per (x, y)
  let best = null, bs = -Infinity;
  for (const sw of [false, true]) for (const a of [1, -1]) for (const b of [1, -1]) {
    const T = sw ? [[0, a], [b, 0]] : [[a, 0], [0, b]];
    // screen u right, v down: want ξ ∝ −u, η ∝ −v  →  maximise −Σ M∘T
    const sc = -(M[0][0] * T[0][0] + M[0][1] * T[0][1] + M[1][0] * T[1][0] + M[1][1] * T[1][1]);
    if (sc > bs) { bs = sc; best = T; }
  }
  return best;
}
function currentT() {
  const o = state.anim.orient;
  if (o === "sky" && state.orientSky) return state.orientSky;
  if (o === "raw") return [[1, 0], [0, 1]];
  return [[1, 0], [0, -1]];                            // FITS / DS9: row 0 at the bottom
}
const cssT = (T) => `matrix(${T[0][0]},${T[1][0]},${T[0][1]},${T[1][1]},0,0)`;
function applyOrient() {
  const cards = $("#cards"); if (cards) cards.style.setProperty("--orient", cssT(currentT()));
  vsxThumbOrient();
}

function animSetCross(on) {
  state.anim.cross = on; saveAnim();
  const cb = $("#an-cross"); if (cb) cb.checked = on;
  const cards = $("#cards"); if (cards) cards.classList.toggle("nocross", !on);
  animRedrawAll();
}

function bindAnimToolbar() {
  const A = state.anim;
  const cb = $("#an-cross"); if (cb) cb.onchange = (e) => animSetCross(e.target.checked);
  const os = $("#an-orient"); if (os) os.onchange = (e) => { A.orient = e.target.value; saveAnim(); applyOrient(); };
  if (!$("#an-play")) return;                      // job from an older version: only the cross toggle
  document.querySelectorAll("#an-k [data-k]").forEach((b) => b.onclick = () => {
    A.k = +b.dataset.k; animTick = 0; saveAnim();
    document.querySelectorAll("#an-k [data-k]").forEach((x) => x.classList.toggle("active", +x.dataset.k === A.k));
    animRedrawAll();
  });
  const unit = { "an-black": " σ", "an-white": " σ", "an-gamma": "", "an-speed": " ms" };
  const key = { "an-black": "black", "an-white": "white", "an-gamma": "gamma", "an-speed": "speed" };
  Object.keys(key).forEach((id) => {
    const el = $("#" + id); if (!el) return;
    el.oninput = () => {
      A[key[id]] = +el.value; $("#" + id + "-v").textContent = el.value + unit[id]; saveAnim();
      if (id === "an-speed") animStart(); else animRedrawAll();
    };
  });
  document.querySelectorAll("#an-stack button").forEach((b) => b.onclick = () => {
    A.stack = b.dataset.st; saveAnim();
    document.querySelectorAll("#an-stack button").forEach((x) => x.classList.toggle("active", x === b));
    animRedrawAll();
  });
  $("#an-play").onclick = () => animTogglePlay();
  $("#an-align").onchange = (e) => { A.align = e.target.checked; saveAnim(); animRedrawAll(); };
  $("#an-fix").onchange = (e) => { A.fix = e.target.checked; saveAnim(); animRedrawAll(); };
  $("#an-skip").onchange = (e) => { A.skipObs = e.target.checked; animTick = 0; saveAnim(); animRedrawAll(); };
  $("#an-reset").onclick = () => {
    Object.assign(A, ANIM_DEF); saveAnim();
    const tb = document.querySelector(".anim-tools"); tb.outerHTML = animToolbar(); bindAnimToolbar();
    animStart(); animRedrawAll();
  };
}

function animTogglePlay() {
  state.anim.play = !state.anim.play;
  const b = $("#an-play"); if (b) b.textContent = state.anim.play ? "❚❚" : "▶";
}
function animStep(d) { state.anim.play = false; const b = $("#an-play"); if (b) b.textContent = "▶"; animTick = Math.max(0, animTick + d); animRedrawAll(); }

function animLoad(job, c) {
  const key = job.id + "/" + c.id;
  if (animCache.has(key)) return animCache.get(key);
  const meta = c.images.frames;
  const p = new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => {
      const cv = document.createElement("canvas"); cv.width = im.width; cv.height = im.height;
      const cx = cv.getContext("2d", { willReadFrequently: true }); cx.drawImage(im, 0, 0);
      const px = cx.getImageData(0, 0, im.width, im.height).data;
      const [a, b, c0] = meta.enc; const lut = new Float32Array(256);
      for (let q = 0; q < 256; q++) lut[q] = a * Math.sinh((q - c0) / b);
      const S = meta.size, frames = [];
      for (let i = 0; i < meta.n; i++) {
        const f = new Float32Array(S * S), o = i * S * S;
        for (let j = 0; j < S * S; j++) f[j] = lut[px[(o + j) * 4]];
        frames.push(f);
      }
      resolve({ meta, frames, subs: {}, fwhm: c.fwhm || 2.5 });
    };
    im.onerror = reject;
    im.src = `/api/jobs/${job.id}/img/${meta.file}`;
  });
  animCache.set(key, p);
  if (animCache.size > 400) animCache.delete(animCache.keys().next().value);
  return p;
}

function animShiftAdd(m, f, S, dx, dy) {
  // m[y,x] += f[y+dy, x+dx] (bilinear, zero outside)
  const ix = Math.floor(dx), iy = Math.floor(dy), wx = dx - ix, wy = dy - iy;
  for (let y = 0; y < S; y++) {
    const y0 = y + iy, y1 = y0 + 1;
    for (let x = 0; x < S; x++) {
      const x0 = x + ix, x1 = x0 + 1;
      let v = 0;
      if (y0 >= 0 && y0 < S) {
        if (x0 >= 0 && x0 < S) v += f[y0 * S + x0] * (1 - wx) * (1 - wy);
        if (x1 >= 0 && x1 < S) v += f[y0 * S + x1] * wx * (1 - wy);
      }
      if (y1 >= 0 && y1 < S) {
        if (x0 >= 0 && x0 < S) v += f[y1 * S + x0] * (1 - wx) * wy;
        if (x1 >= 0 && x1 < S) v += f[y1 * S + x1] * wx * wy;
      }
      m[y * S + x] += v;
    }
  }
}

function animFrameList(d) {
  const N = d.meta.n, tp = d.meta.tp;
  let idx = [...Array(N).keys()];
  if (state.anim.skipObs && tp) {
    const ok = idx.filter((i) => tp[i] >= 0.5);
    if (ok.length >= 2) idx = ok;
  }
  return idx;
}

// One image from frames ``grp``: shifted onto the object (align) or with the stars still,
// combined by mean or per-pixel median (median needs >= 3 frames).
function animGroup(d, grp, align, mode, fix = true) {
  const S = d.meta.size, sh = (fix && d.meta.shift) || null, cnt = grp.length;
  let px = 0, py = 0;
  for (const i of grp) { px += d.meta.track[i][0]; py += d.meta.track[i][1]; }
  px /= cnt; py /= cnt;
  const useMed = mode === "median" && cnt >= 3;
  const layers = useMed ? [] : null;
  const m = new Float32Array(S * S);
  for (const i of grp) {
    const f = d.frames[i];
    // local star shift of frame i (stars and object of that frame are displaced alike)
    let dx = sh ? sh[i][0] : 0, dy = sh ? sh[i][1] : 0;
    if (align && cnt > 1) {
      // stack on the object: its position in frame i lands on the group's mean position;
      // stars get slightly trailed but stay put
      dx += d.meta.track[i][0] - px; dy += d.meta.track[i][1] - py;
    }
    const tgt = useMed ? new Float32Array(S * S) : m;
    if (dx || dy) animShiftAdd(tgt, f, S, dx, dy);
    else for (let j = 0; j < tgt.length; j++) tgt[j] += f[j];
    if (useMed) layers.push(tgt);
  }
  if (useMed) {
    const v = new Float32Array(cnt), h = cnt >> 1;
    for (let j = 0; j < m.length; j++) {
      for (let q = 0; q < cnt; q++) v[q] = layers[q][j];
      v.sort();
      m[j] = cnt & 1 ? v[h] : 0.5 * (v[h - 1] + v[h]);
    }
  } else {
    for (let j = 0; j < m.length; j++) m[j] /= cnt;
  }
  const smp = []; for (let j = 0; j < m.length; j += 3) smp.push(m[j]);
  smp.sort((a, b) => a - b);
  const med = smp[smp.length >> 1];
  const dev = smp.map((v) => Math.abs(v - med)).sort((a, b) => a - b);
  return { m, x: px, y: py, med, sig: 1.4826 * dev[dev.length >> 1] || 1 };
}

function animSubs(d, k) {
  const A = state.anim, list = animFrameList(d), N = list.length;
  k = (!k || k >= N) ? N : k;
  const mode = A.stack || "mean";
  const key = `${k}${A.align ? "a" : "s"}${A.fix ? "f" : ""}${mode}${N}`;
  if (d.subs[key]) return d.subs[key];
  const out = [], skipped = d.meta.n - N;
  let start = 0;
  for (let g = 0; g < k; g++) {
    const cnt = Math.floor(N / k) + (g < N % k ? 1 : 0);
    const o = animGroup(d, list.slice(start, start + cnt), A.align, mode, A.fix);
    o.skipped = skipped;
    out.push(o);
    start += cnt;
  }
  // one noise level for all images of the animation (no flicker)
  const sig = out.map((o) => o.sig).sort((a, b) => a - b)[out.length >> 1];
  out.forEach((o) => o.sig = sig);
  d.subs[key] = out;
  return out;
}

function drawImageG(cv, g, S, fwhm, lo, hi, gamma, cross) {
  const inv = 1 / (hi - lo), ig = 1 / gamma;
  let tmp = cv._tmp;
  if (!tmp || tmp.width !== S) { tmp = cv._tmp = document.createElement("canvas"); tmp.width = tmp.height = S; }
  const tx = tmp.getContext("2d"), id = tx.createImageData(S, S), px = id.data;
  for (let j = 0; j < S * S; j++) {
    let t = ((g.m[j] - g.med) / g.sig - lo) * inv;
    t = t <= 0 ? 0 : t >= 1 ? 1 : (ig === 1 ? t : Math.pow(t, ig));
    const v = (t * 255) | 0, o = j * 4;
    px[o] = px[o + 1] = px[o + 2] = v; px[o + 3] = 255;
  }
  tx.putImageData(id, 0, 0);
  const ctx = cv.getContext("2d"), W = cv.width, sc = W / S;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, W, W);
  const cx = (g.x + 0.5) * sc, cy = (g.y + 0.5) * sc;
  const gap = Math.max(2 * fwhm, 5) * sc * 0.6, arm = Math.max(2.5 * fwhm, 6) * sc * 0.6;
  if (cross) {
    ctx.strokeStyle = "rgb(255,196,40)"; ctx.lineWidth = 2; ctx.beginPath();
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      ctx.moveTo(cx + dx * gap, cy + dy * gap); ctx.lineTo(cx + dx * (gap + arm), cy + dy * (gap + arm));
    }
    ctx.stroke();
  }
}

function animDraw(cv, d) {
  const A = state.anim, subs = animSubs(d, A.k), idx = animTick % subs.length, g = subs[idx], S = d.meta.size;
  drawImageG(cv, g, S, d.fwhm, A.black, Math.max(A.white, A.black + 0.1), A.gamma, A.cross);
  const cnt = cv.parentNode && cv.parentNode.querySelector(".cnt");
  if (cnt) cnt.textContent = `${idx + 1}/${subs.length}${g.skipped ? ` · bez ${g.skipped}` : ""}${(A.stack === "median" && subs.length < d.meta.n && (d.meta.n / subs.length) >= 3) ? " · medián" : ""}`;
}

function animRedrawAll() {
  for (const cv of animLive) {
    if (!cv.isConnected) { animLive.delete(cv); continue; }
    if (cv._d) animDraw(cv, cv._d);
  }
}
function animStart() {
  clearInterval(animTimer);
  animTimer = setInterval(() => {
    if (!state.anim.play || state.view !== "job" || (state.tab !== "cands" && state.tab !== "static" && !document.querySelector(".known-modal"))) return;
    animTick++; animRedrawAll();
  }, Math.max(50, state.anim.speed));
}
function animAttach(job) {
  if (animObs) animObs.disconnect();
  animLive.clear();
  if (!("IntersectionObserver" in window)) return;
  animObs = new IntersectionObserver((ents) => {
    for (const e of ents) {
      const cv = e.target;
      if (!e.isIntersecting) { animLive.delete(cv); continue; }
      animLive.add(cv);
      if (!cv._d && !cv._loading) {
        const c = state.cands.find((x) => x.id === +cv.dataset.cid);
        if (!c) continue;
        cv._loading = true;
        animLoad(job, c).then((d) => { cv._d = d; animDraw(cv, d); }).catch(() => { cv._loading = false; });
      }
    }
  }, { rootMargin: "300px" });
  document.querySelectorAll("canvas.anim").forEach((cv) => animObs.observe(cv));
  animStart();
}
function markFocus() {
  document.querySelectorAll(".card").forEach((c) => c.classList.toggle("focus", +c.dataset.i === state.focus));
  const f = document.querySelector(".card.focus"); if (f) f.scrollIntoView({ block: "nearest", behavior: "smooth" });   // scroll-margin-top keeps it below the sticky bar
}
const STATUS_CZ = { accepted: "přijato", rejected: "zamítnuto", unsure: "nejisté", unreviewed: "neprohlédnuto" };
function undoable(job) { return state.history.some((h) => h.jobId === job.id); }
// A decision is shown at once and saved in the background (candidates.json of a dense field has
// thousands of entries – waiting for the server and re-drawing every card took seconds).
function applyLocal(cid, status, reviewed) {
  const c = state.cands.find((x) => x.id === cid);
  if (!c) return;
  c.status = status; c.reviewed = reviewed;
}
function refreshCard(job, cid) {
  const card = document.querySelector(`.card[data-cid="${cid}"]`);
  const visible = visibleCands();
  const still = visible.some((x) => x.id === cid);
  if (!card || (state.filter === "rej_auto" && state.groupStars)) { drawCards(job); updateCounts(); return; }
  if (still) {               // same filter (e.g. "vše"): only the card's colour and buttons change
    const c = state.cands.find((x) => x.id === cid);
    card.className = card.className.replace(/\bst-\S+/, `st-${cardState(c)}`);
    card.querySelectorAll("[data-set]").forEach((b) => {
      const k = b.dataset.set;
      b.classList.toggle("ok", k === "accepted" && human(c, k));
      b.classList.toggle("warnb", k === "unsure" && human(c, k));
      b.classList.toggle("badb", k === "rejected" && human(c, k));
    });
    card.querySelectorAll(".badge.auto").forEach((b) => b.remove());
  } else {
    card.remove();
    const cards = document.querySelectorAll("#cards .card");
    cards.forEach((el, i) => { el.dataset.i = i; });
    if (!cards.length) $("#cards").innerHTML = `<div class="empty">Žádní kandidáti v tomto filtru.</div>`;
    state.focus = Math.min(state.focus, Math.max(0, cards.length - 1));
    markFocus();
  }
  updateCounts();
}
let saveQueue = Promise.resolve();        // decisions reach the server in click order
function saveStatus(job, cid, body, rollback) {
  saveQueue = saveQueue.then(() => saveOne(job, cid, body, rollback));
}
async function saveOne(job, cid, body, rollback) {
  try {
    const c = await api(`/api/jobs/${job.id}/candidates/${cid}`, { method: "PATCH", body });
    const i = state.cands.findIndex((x) => x.id === cid);
    if (i >= 0 && state.cands[i].status === c.status) {
      state.cands[i] = c;   // not overtaken by a newer click
      const ia = (state.allCands || []).findIndex((x) => x.id === cid);
      if (ia >= 0) state.allCands[ia] = c;
    }
  } catch (e) {
    toast(`#${cid}: označení se nepodařilo uložit (${e.message})`, true);
    rollback(); if (state.jobId === job.id) refreshCard(job, cid);
  }
}
function setStatus(job, cid, status) {
  const old = state.cands.find((x) => x.id === cid);
  if (!old || (old.status === status && (status !== "rejected" || old.reviewed))) return;
  const prev = old.status, prevReviewed = !!old.reviewed;
  const body = { status };
  if (status === "unreviewed") body.reviewed = false;
  applyLocal(cid, status, status !== "unreviewed");
  state.history.push({ jobId: job.id, cid, prev, prevReviewed });
  if (state.history.length > 200) state.history.shift();
  toast(`#${cid} → ${STATUS_CZ[status] || status}   ·   Z = zpět`);
  refreshCard(job, cid);
  saveStatus(job, cid, body, () => applyLocal(cid, prev, prevReviewed));
}
function undoStatus(job) {
  let k = -1;
  for (let j = state.history.length - 1; j >= 0; j--) if (state.history[j].jobId === job.id) { k = j; break; }
  if (k < 0) { toast("Není co vracet."); return; }
  const h = state.history.splice(k, 1)[0];
  const cur = state.cands.find((x) => x.id === h.cid);
  const now = cur ? [cur.status, !!cur.reviewed] : null;
  applyLocal(h.cid, h.prev, h.prevReviewed);
  drawCards(job); updateCounts();
  const idx = visibleCands().findIndex((x) => x.id === h.cid);
  if (idx >= 0) { state.focus = idx; markFocus(); }
  toast(`Vráceno: #${h.cid} je zase ${STATUS_CZ[h.prev] || h.prev}.`);
  saveStatus(job, h.cid, { status: h.prev, reviewed: h.prevReviewed }, () => { if (now) applyLocal(h.cid, now[0], now[1]); });
}
document.addEventListener("keydown", (e) => {
  if (state.view !== "job" || (state.tab !== "cands" && state.tab !== "static") || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const list = visibleCands(); if (!list.length) return;
  const job = state.jobs.find((j) => j.id === state.jobId);
  const k = e.key.toLowerCase();
  if (k === "z") { e.preventDefault(); undoStatus(job); return; }
  if (k === "j") { state.focus = Math.min(list.length - 1, state.focus + 1); markFocus(); }
  else if (k === "k") { state.focus = Math.max(0, state.focus - 1); markFocus(); }
  else if (["a", "r", "u"].includes(k)) setStatus(job, list[state.focus].id, { a: "accepted", r: "rejected", u: "unsure" }[k]);
  else if (k === "p") animTogglePlay();
  else if (k === "x") animSetCross(!state.anim.cross);
  else if (k === "d") openSky(job, list[state.focus]);
  else if (k === "e") openEphem(job, list[state.focus]);
  else if (e.key === "ArrowRight") { e.preventDefault(); animStep(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); animStep(-1); }
});

async function renderKnown(job) {
  const known = await api(`/api/jobs/${job.id}/known`);
  // comets first, then the detected objects, then the rest by predicted brightness (a field near the
  // ecliptic lists ~1000 asteroids; 88P/Howell 18.3 mag sat on row 204)
  const rank = (k) => (k.kind === "comet" ? 0 : (k.matched_candidate != null || k.saturated) ? 1 : 2);
  known.sort((a, b) => rank(a) - rank(b) || (a.mag ?? 99) - (b.mag ?? 99));
  if (!known.length) { $("#tabbody").innerHTML = `<div class="empty">Žádné známé objekty v poli (nebo chybí WCS / MPCORB).</div>`; return; }
  $("#tabbody").innerHTML = `<table class="tbl"><tr><th>Objekt</th><th>Typ</th><th>Mag (předp.)</th><th title="Přibližná jasnost z PSF fitu, nulový bod z identifikovaných planetek pole">Mag (změř.)</th><th>Rychlost ″/min</th><th>PA</th><th>RA</th><th>Dec</th><th>x, y</th><th>Detekován</th><th title="SNR změřené přímo na předpovězené stopě">Vynucené SNR</th><th title="Snímky podél předpovězené stopy: průměrný a mediánový stack složený na pohyb a animace (vytvoří se až po kliknutí)">Náhled</th></tr>
    ${known.map((k, i) => `<tr><td>${esc(k.name)}</td><td>${k.kind === "comet" ? "kometa" : "planetka"}</td><td>${k.mag != null ? k.mag.toFixed(1) : "–"}</td><td>${k.mag_measured != null ? k.mag_measured.toFixed(1) : "–"}</td>
      <td>${k.rate_arcsec_min.toFixed(3)}</td><td>${k.pa_deg.toFixed(1)}</td><td class="mono">${raStr(k.ra)}</td><td class="mono">${decStr(k.dec)}</td>
      <td class="mono">${k.x?.toFixed(0)}, ${k.y?.toFixed(0)}</td><td>${k.matched_candidate != null ? "✓ #" + k.matched_candidate : (k.saturated ? `<span title="Přesycený objekt: jádro je ve všech snímcích maskované, poloha z díry v datech, která se pohybuje s ním">✓ přesycený</span>` : "–")}</td>
      <td>${k.forced_snr != null ? k.forced_snr.toFixed(1) + (k.matched_candidate == null && !k.saturated && k.forced_snr < (+(job.params?.snr_threshold) || 6) ? ' <span class="sub">(pod prahem)</span>' : "") : "–"}${k.forced_note ? ` <span class="badge flag" title="Hodnota může být zkreslená">${esc(k.forced_note)}</span>` : ""}</td>
      <td>${k.images?.frames ? `<button class="btn small" data-kprev="${i}">náhled</button>` : "–"}</td></tr>`).join("")}</table>
    <p class="sub">Vynucené SNR = signifikance změřená přímo na předpovězené stopě, dělená skutečným šumem téhož součtu na náhodných stopách stejné rychlosti (pomalé stopy mají šum vyšší, než udává model). Ukazuje, zda byl objekt v datech nad prahem (${esc(job.params?.snr_threshold ?? "výchozí")}), i když ho hledání nenašlo. Náhled ukazuje snímky podél předpovězené stopy (kříž = předpověď).</p>`;
  document.querySelectorAll("[data-kprev]").forEach((b) => b.onclick = () => openKnownPreview(job, known[+b.dataset.kprev], +b.dataset.kprev));
}

// Preview of a known object along its predicted track: mean and median stack on the
// motion and the animation – everything is computed here from the stored frame strip.
async function openKnownPreview(job, k, i) {
  document.querySelector(".known-modal")?.remove();
  const A = state.anim, fwhm = +(job.summary?.fwhm_px || 2.5);
  const pseudo = { id: "k" + (k.name || i), images: k.images, fwhm };
  const wrap = document.createElement("div");
  wrap.className = "modal known-modal";
  wrap.innerHTML = `<div class="modal-box known-box"><div class="modal-head"><b>${esc(k.name)}</b>
      <span class="sub">${k.mag != null ? k.mag.toFixed(1) + " mag · " : ""}${k.rate_arcsec_min.toFixed(2)}″/min · PA ${k.pa_deg.toFixed(0)}° · vynucené SNR ${k.forced_snr != null ? k.forced_snr.toFixed(1) : "–"}${k.matched_candidate != null ? " · kandidát #" + k.matched_candidate : (k.saturated ? " · přesycený (detekován podle masky)" : (k.forced_snr != null && k.forced_snr >= 4.5 ? " · nalezen jen na předpovězené dráze (pod prahem hledání)" : (k.forced_snr != null && k.forced_snr >= 3.5 ? " · slabě na předpovězené dráze (na hranici šumu)" : " · nedetekován")))}</span>
      <button class="btn small" data-close>✕</button></div>
    <div class="known-panels">
      <figure><canvas id="kp-mean" width="230" height="230"></canvas><figcaption>průměrný stack (na pohyb)</figcaption></figure>
      <figure><canvas id="kp-med" width="230" height="230"></canvas><figcaption>mediánový stack (na pohyb)</figcaption></figure>
      <figure><canvas id="kp-anim" class="anim" width="230" height="230"></canvas><span class="cnt"></span><figcaption>animace (${A.stack === "median" ? "medián" : "průměr"}, nastavení jako u kandidátů)</figcaption></figure>
    </div>
    <div class="modal-foot"><span class="sub">Kříž = předpovězená poloha. Jas a kontrast podle lišty animace na kartě Kandidáti.</span></div></div>`;
  document.body.appendChild(wrap);
  const cvA = $("#kp-anim");
  const close = () => { animLive.delete(cvA); wrap.remove(); document.removeEventListener("keydown", esc_, true); };
  const esc_ = (e) => { if (e.key === "Escape") { e.stopImmediatePropagation(); e.preventDefault(); close(); } };
  document.addEventListener("keydown", esc_, true);
  wrap.onclick = (e) => { if (e.target === wrap || e.target.closest("[data-close]")) close(); };
  try {
    const d = await animLoad(job, pseudo);
    const all = animFrameList(d), S = d.meta.size;
    for (const [id, mode] of [["#kp-mean", "mean"], ["#kp-med", "median"]]) {
      const g = animGroup(d, all, true, mode, A.fix);
      drawImageG($(id), g, S, fwhm, -1.5, 12, 2.0, A.cross);
    }
    cvA._d = d; animLive.add(cvA); animDraw(cvA, d); animStart();
  } catch (e) {
    wrap.querySelector(".known-panels").innerHTML = `<div class="empty">Náhled nelze načíst.</div>`;
  }
}

// ------------------------------------------------------------ where real objects get lost
// Injections are real objects with a known position: whatever rule rejects them in the end would
// reject an unknown asteroid in the same field. Final state after ALL vetting (rules chain: (372)
// Palma passed one rule and was rejected by the next). Overview only – changes no decision.
const INJ_SOFT = new Set(["snížená propustnost", "spíš proměnná hvězda", "užší než hvězdy?", "pomalý u hvězdy?",
  "také jako zdroj navíc", "hvězda vedle stopy", "hraniční skóre podle injekcí"]);
const injReason = (f) => f.replace(/\s*\(.*$/, "").replace(/\s+–.*$/, "").trim();
async function injLossPanel(job) {
  const box = $("#inj-loss"); if (!box) return;
  let cands;
  try { cands = await api(`/api/jobs/${job.id}/candidates`); } catch { return; }
  const inj = cands.filter((c) => c.injected != null && c.source !== "twilight");
  if (!inj.length) return;
  const byObj = new Map();
  for (const c of inj) { const l = byObj.get(c.injected) || []; l.push(c); byObj.set(c.injected, l); }
  const found = byObj.size;
  let passed = 0; const reasons = new Map(), chains = new Map();
  for (const l of byObj.values()) {
    if (l.some((c) => c.auto_status === "candidate")) { passed++; continue; }
    const best = l.reduce((a, b) => (b.snr > a.snr ? b : a));
    const hard = [...new Set((best.flags || []).map(injReason).filter((f) => f && !INJ_SOFT.has(f)))];
    const key = hard.length ? hard : ["(bez důvodu)"];
    key.forEach((f) => reasons.set(f, (reasons.get(f) || 0) + 1));
    const ch = key.join(" → "); chains.set(ch, (chains.get(ch) || 0) + 1);
  }
  const lost = found - passed;
  const total = job.summary?.injection?.n;
  const rs = [...reasons.entries()].sort((a, b) => b[1] - a[1]);
  const cs = [...chains.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
  const maxR = rs.length ? rs[0][1] : 1;
  const warn = rs.filter(([, n]) => n > 0.2 * found);
  const pct = (a, b) => b ? Math.round(100 * a / b) : 0;
  box.innerHTML = `<div class="panel inj-loss"><h3 style="margin-top:0">Injekce – kde se ztrácejí skutečné objekty</h3>
    <div class="inj-stats">${total != null ? `<div><span class="sub">injektováno</span><b>${total}</b></div>` : ""}
      <div><span class="sub">nalezeno</span><b>${found}</b></div>
      <div><span class="sub">prošlo celým prověřením</span><b>${passed}</b> <span class="sub">(${pct(passed, found)} %)</span></div>
      <div><span class="sub">ztraceno v prověřování</span><b>${lost}</b></div></div>
    ${warn.map(([f, n]) => `<div class="inj-warn">„${esc(f)}“ zamítlo ${n} z ${found} nalezených injekcí (${pct(n, found)} %) – v tomto poli zahazuje i skutečné objekty. Zkontrolujte filtr „druhá šance“.</div>`).join("")}
    ${rs.length ? `<div class="sub" style="margin:10px 0 4px">Konečný důvod zamítnutí ztracených injekcí (jedna injekce může mít víc důvodů)</div>
    <div class="inj-bars">${rs.map(([f, n]) => `<span>${esc(f)}</span><div class="bar"><div style="width:${Math.round(100 * n / maxR)}%"></div></div><span class="n">${n}</span>`).join("")}</div>
    <div class="sub" style="margin:10px 0 4px">Nejčastější řetězy</div>
    ${cs.map(([k, n]) => `<div>${esc(k)} <span class="sub">– ${n}×</span></div>`).join("")}` : `<div class="sub" style="margin-top:8px">Žádná nalezená injekce nebyla zamítnuta.</div>`}
    <div class="sub" style="margin-top:10px">Jen přehled – na rozhodování o kandidátech nemá vliv.</div></div>`;
}

function renderSummary(job) {
  const s = job.summary || {};
  const m = s.masters || {};
  const sel = (x) => x ? `<div class="sel ${x.path ? "ok" : "bad"}"><b>${x.kind}</b>: ${esc(x.path || "—")}<br><span class="sub">${esc(x.reason)}</span>${(x.warnings || []).map((w) => `<br><span class="w">${esc(w)}</span>`).join("")}</div>` : "";
  const rows = [["Snímků", s.n_frames], ["Expozice", s.exptime + " s"], ["Filtr", s.filter], ["Délka sekvence", s.span_min + " min"],
    ["Měřítko", s.pixel_scale + " ″/px"], ["FWHM", s.fwhm_px + " px"],
    ["WCS", ({ header: "z hlavičky (řešení)", astap: "ASTAP", pointing: "jen poloha montáže – nepřesné", "pointing+known": "poloha montáže, posun podle známých objektů", "header+UCAC4": "z hlavičky, zpřesněno podle UCAC4", "astap+UCAC4": "ASTAP, zpřesněno podle UCAC4", "pointing+UCAC4": "poloha montáže, zpřesněno podle UCAC4" }[s.wcs_source] || (s.wcs ? "ano" : "–"))
      + (s.wcs_shift ? ` (posun ${s.wcs_shift.arcsec}″ podle ${s.wcs_shift.objects.join(", ")})` : "")], ["Hypotéz rychlosti", s.grid_size], ["Backend", s.backend],
    ["Detekcí", s.detections], ["Po automatickém prověření", s.candidates], ["Známých v poli", s.known_in_field],
    ["Známých spárováno s kandidáty", s.known_detected],
    ["Známých s nuceným SNR nad prahem", s.known_forced_above_threshold ?? "–"],
    ["Kandidátů na neznámé", s.unknown_candidates], ["WCS", s.wcs ? "ano" : "ne"]];
  if (s.known_rejected) rows.push(["Známé objekty zamítnuté prověřením", s.known_rejected.length
    ? s.known_rejected.map((k) => `${k.name} (${k.flags.join(", ") || "–"})`).join("; ")
    : "žádné – pravidla nezahodila žádnou identifikovanou planetku"]);
  if (s.search_mode) rows.push(["Režim hledání", `${{ tracking: "synthetic tracking", twilight: "zdroj navíc (porovnání s UCAC4)", both: "synthetic tracking + zdroj navíc" }[s.search_mode] || s.search_mode}${s.search_mode_reason ? " – " + s.search_mode_reason : ""}`]);
  if (s.twilight) { const t = s.twilight; rows.push(["Zdroj navíc", `${t.detections} zdrojů v mediánu ${t.n_frames} snímků, ${t.matched} spárováno s UCAC4 (z ${t.catalog_in_field} v poli) · astrometrie ${t.astrometry_rms_px} px, nulový bod ±${t.zp_scatter_mag} mag, mez ~${t.limit_mag_detected} mag · kandidátů ${t.candidates}, zamítnuto ${t.rejected}${t.planets && t.planets.length ? " · v poli: " + t.planets.join(", ") : ""}`]); }
  if (s.throughput) rows.push(["Propustnost (mapa)", `rozdíly v poli až ${Math.round(s.throughput.spread * 100)} %, nejhorší snímek ${Math.round(Math.min(...s.throughput.frame_min) * 100)} %`]);
  if (s.noise_threshold) { const t = s.noise_threshold; rows.push(["Práh šumu po dlaždicích", `SNR ${t.min.toFixed(1)}–${t.max.toFixed(1)} (medián ${t.median.toFixed(1)}, základ ${t.base}) · odhad falešných kandidátů: ${t.expected_false_base} při pevném prahu → ${t.expected_false} s prahem dlaždic`]); }
  if (s.injection_rank) { const r = s.injection_rank; rows.push(["Kalibrace injekcemi", `ponecháno ${Math.round(r.keep * 100)} % injekcí (AUC ${r.cv_auc}) · neznámých ${r.others} → ${r.remaining} ke kontrole, ${r.set_aside} odloženo (${r.borderline ? `${r.borderline} neznámých kandidátů s hraničním skóre – filtr „hraniční skóre“, ` : ""}ostatní „nízké skóre podle injekcí“)${r.feature_set === "safe" ? " · bezpečná sada příznaků (s plnou by pod hranicí byly známé)" : ""}` + (r.known_below_cut && r.known_below_cut.length ? ` · POZOR: pod hranicí i známé ${r.known_below_cut.join(", ")}${r.cut_lowered_for_known ? " (hranice snížena)" : ""}` : "")]); }
  if (s.injection && s.injection.static) {
    const K = { point: "bodové (nova, kometa v outburstu)", comet: "komety s komou" };
    rows.push(["Injekce (zdroj navíc)", `${s.injection.detected}/${s.injection.n} statických zdrojů nalezeno, ${s.injection.passed} prošlo · `
      + Object.entries(s.injection.kinds || {}).map(([k, v]) => `${K[k] || k} ${v.passed}/${v.n}`
        + (v.faintest_passed_mag != null ? ` (nejslabší ≈ ${(+v.faintest_passed_mag).toFixed(1)} mag)` : "")
        + " [" + v.bins.map((b) => `SNR ${b.snr_from}–${b.snr_to ?? "∞"}: ${b.passed}/${b.n}`).join(", ") + "]").join(" · ")
      + (s.injection.rejected_by && Object.keys(s.injection.rejected_by).length
        ? " · zamítnuté nalezené: " + Object.entries(s.injection.rejected_by).slice(0, 4).map(([f, n]) => `${f} ${n}×`).join(", ") : "")]);
  } else if (s.injection) rows.push(["Injekce", `${s.injection.detected}/${s.injection.n} nalezeno, ${s.injection.passed} prošlo prověřením · `
    + (s.injection.snr_basis === "expected" ? "očekávané " : "")
    + s.injection.bins.map((b) => `SNR ${b.snr_from}–${b.snr_to ?? "∞"}: ${b.passed}/${b.n}`).join(", ")
    + (s.injection.comets ? ` · komety ${s.injection.comets.passed}/${s.injection.comets.n} (difúzní ≥ 30″: ${s.injection.comets.diffuse_passed}/${s.injection.comets.diffuse_n})` : "")]);
  if (s.photometry) { const p = s.photometry; rows.push(["Fotometrie (přibližná)", p.source === "ucac4"
    ? `nulový bod ${p.zp} z ${p.n} hvězd UCAC4 (APASS ${p.band}), rozptyl ±${p.scatter} mag` + (p.known_offset != null ? ` · známé planetky: změřeno − předpověď (V) = ${p.known_offset > 0 ? "+" : ""}${p.known_offset} mag (${p.known_n} objektů)` : "")
    : `nulový bod ${p.zp} z ${p.n} planetek pole (předpovědi MPC, V), rozptyl ±${p.scatter} mag`]); }
  if (s.known_saturated && s.known_saturated.length) rows.push(["Přesycené známé objekty", s.known_saturated.join(", ") + " – nalezeny podle masky přesycení, která se pohybuje s nimi"]);
  $("#tabbody").innerHTML = `<div class="grid2"><div class="panel"><table class="tbl">${rows.map(([k, v]) => `<tr><td class="sub">${k}</td><td>${esc(v)}</td></tr>`).join("")}</table></div>
    <div class="panel"><h3 style="margin-top:0">Kalibrace</h3>${sel(m.dark)}${sel(m.flat)}
    <h3>Časy (s)</h3><table class="tbl">${Object.entries(s.timings_s || {}).map(([k, v]) => `<tr><td class="sub">${k}</td><td>${v}</td></tr>`).join("")}</table></div></div>
    <div id="inj-loss"></div>`;
  injLossPanel(job);
}

function astroNote(s) {
  const src = s.wcs_source || "", r = s.wcs_refined || {};
  if (src.includes("UCAC4") && r.refined_px) {
    const worst = Math.max(...Object.values(r.refined_px)), med = r.refined_median_px, sc = +(s.pixel_scale || 0);
    return `Astrometrie: WCS zpřesněno podle ${r.pairs} hvězd UCAC4 (SIP 5. řádu), odchylka hvězd medián ${med} px${sc ? ` = ${(med * sc).toFixed(2)}″` : ""}, nejhorší zóna ${worst} px. V hlavičce MPC: NET UCAC-4.`;
  }
  if (src.startsWith("pointing")) return "Astrometrie: WCS jen z polohy montáže – pro MPC nevhodné.";
  return `Astrometrie: WCS ${src === "astap" ? "z ASTAP" : "z hlavičky"} bez zpřesnění podle katalogu (chybí UCAC4 nebo zpřesnění neprošlo).`;
}

function renderExport(job) {
  $("#tabbody").innerHTML = `<div class="panel">
    <p>Export obsahuje jen <b>přijaté</b> kandidáty (MPC) nebo všechny (CSV).</p>
    <div class="warnbox">${astroNote(job.summary || {})} Před odesláním na MPC každé měření zkontrolujte.</div>
    <div class="toolbar"><a class="btn" href="/api/jobs/${job.id}/export/mpc80" download="${job.id}_mpc.txt">MPC 80 sloupců</a>
    <a class="btn" href="/api/jobs/${job.id}/export/csv" download="${job.id}.csv">CSV</a></div>
    <pre class="log" id="mpcprev"></pre></div>`;
  api(`/api/jobs/${job.id}/export/mpc80`).then((t) => $("#mpcprev").textContent = t);
  if (desk()) {           // the desktop window has no download manager: native "Save as" dialog
    $("#tabbody").querySelectorAll("a[download]").forEach((a) => a.onclick = async (ev) => {
      ev.preventDefault();
      try {
        const text = await api(a.getAttribute("href"));
        const p = await desk().save_text(a.getAttribute("download"), text);
        if (p) toast(`Uloženo: ${p}`);
      } catch (e) { toast(e.message, true); }
    });
  }
}

// ------------------------------------------------------------------ directory browser
async function browseDir(start) {
  if (desk()) {
    try { return await desk().pick_folder(start || ""); } catch (e) { toast(e.message, true); }
  }
  return browseDirWeb(start);
}
function browseDirWeb(start) {
  return new Promise((resolve) => {
    const m = $("#browser"); m.classList.remove("hidden");
    let cur = start || "";
    const load = async (p) => {
      try {
        const r = await api(`/api/browse?path=${encodeURIComponent(p)}`);
        cur = r.path; $("#br-path").value = r.path;
        $("#br-info").textContent = r.path ? `${r.fits} FITS v tomto adresáři` : "";
        $("#br-list").innerHTML = (r.parent !== null && r.path ? `<div class="br-item" data-p="${esc(r.parent)}">⬑ ..</div>` : "")
          + r.dirs.map((d) => `<div class="br-item" data-p="${esc(d.path)}">📁 ${esc(d.name)}</div>`).join("");
        $("#br-list").querySelectorAll(".br-item").forEach((it) => it.onclick = () => load(it.dataset.p));
      } catch (e) { toast(e.message, true); }
    };
    const close = (v) => { m.classList.add("hidden"); resolve(v); };
    $("#br-close").onclick = () => close(null);
    $("#br-ok").onclick = () => close(cur || null);
    $("#br-go").onclick = () => load($("#br-path").value);
    load(cur);
  });
}
function bindBrowse(root = document) {
  root.querySelectorAll("[data-browse]").forEach((b) => b.onclick = async () => {
    const inp = document.getElementById(b.dataset.browse);
    const p = await browseDir(inp.value); if (p) { inp.value = p; inp.dispatchEvent(new Event("change")); }
  });
}

// ------------------------------------------------------------------ new job
async function viewNew() {
  state.instruments = await api("/api/instruments");
  const st = await api("/api/settings");
  if (!state.instruments.length) {
    $("#main").innerHTML = `<div class="panel"><h2>Nová úloha</h2><p>Nejdřív vytvořte <a href="#" data-view="instruments">profil přístroje</a>.</p></div>`; return;
  }
  const s = st.search;
  $("#main").innerHTML = `<div class="panel"><h2>Nová úloha</h2><p class="sub">Ruční zpracování adresáře se snímky. Může obsahovat více zorných polí, nebo podadresáře s předem roztříděnými poli.</p>
    <div class="form">
      <label>Přístroj</label><div><select id="nj-inst">${state.instruments.map((i) => `<option value="${esc(i.id)}">${esc(i.name || i.id)} (${i.pixel_scale_arcsec}″/px)</option>`).join("")}</select><div id="nj-ident" class="hint">Po výběru adresáře se přístroj rozpozná podle hlavičky FITS.</div></div>
      <label>Adresář se snímky</label><div class="row"><input id="nj-dir" spellcheck="false"><button class="btn" data-browse="nj-dir">Procházet…</button></div>
      <label>Název úlohy</label><input id="nj-name" placeholder="automaticky z adresáře">
      <label>Režim hledání</label><div><select id="nj-mode">${[["auto", "automaticky"], ["tracking", "synthetic tracking"], ["twilight", "zdroj navíc (porovnání s UCAC4)"], ["both", "obojí"]].map(([k, l]) => `<option value="${k}" ${(s.search_mode || "auto") === k ? "selected" : ""}>${l}</option>`).join("")}</select><div class="hint">Automaticky: pole se 3–${s.twilight_max_frames ?? 6} snímky → zdroj navíc (bez syntetického trackingu); delší série se snímky za soumraku (Slunce nad −18°) → obojí; jinak synthetic tracking. Zdroj navíc hledá objekty, které nejsou v UCAC4 (do ~16 mag), i mlhavé, a kontroluje NGC/IC/PGC, proměnné VSX, odlesky (dithering) a známé planetky.</div></div>
      <label>Rychlost min / max (″/min)</label><div class="row"><input id="nj-rmin" type="number" step="0.05" value="${s.rate_min_arcsec_min}"><input id="nj-rmax" type="number" step="0.5" value="${s.rate_max_arcsec_min}"></div>
      <label>Práh SNR</label><input id="nj-snr" type="number" step="0.5" value="${s.snr_threshold}">
      <label>Max. odchylka segmentu (σ)</label><input id="nj-seg" type="number" step="0.5" value="${s.segment_z_max}">
      <label>Syntetické objekty (injekce)</label><div><input id="nj-inj" type="number" min="0" step="10" value="${s.inject_n ?? 0}"><div class="hint">Počet umělých planetek vložených do snímků (např. 50). Změří úplnost a dodá příklady pro klasifikátor; v seznamu kandidátů se neukazují.</div></div>
    </div>
    <div id="nj-fields"></div>
    <h3>Kalibrace, kterou program použije</h3><div id="nj-prev" class="sub">Vyberte adresář se snímky…</div>
    <div id="nj-cal"></div>
    <div style="margin-top:14px"><button class="btn primary" id="nj-go">Spustit</button></div></div>`;
  bindBrowse();
  const prev = async () => {
    const dir = $("#nj-dir").value; if (!dir) return;
    try {
      const r = await api(`/api/instruments/${$("#nj-inst").value}/preview`, { method: "POST", body: { lights_dir: dir } });
      if (!r.frames) { $("#nj-fields").innerHTML = ""; $("#nj-prev").innerHTML = `<div class="sel bad">V adresáři nejsou FITS snímky.</div>`; return; }
      const m = r.meta;
      $("#nj-fields").innerHTML = (r.dirs ? dirsHtml(r.dirs) : fieldsHtml(r.fields)) + groupsHtml(r.groups);
      $("#nj-prev").innerHTML = `<div class="sub">${r.dirs ? `celkem ${r.frames} snímků v ${r.dirs.length} adresářích; mastery níže pro první adresář (${esc(r.dirs[0].rel)}) – ostatní viz sloupec Mastery<br>` : ""}${r.dirs ? r.dirs[0].frames : r.frames} snímků · expozice ${m.exptime} s · filtr ${esc(m.filter)} (${esc(m.raw_filter)}) · ${m.width}×${m.height} · bin ${m.xbin}x${m.ybin} · T ${m.temperature ?? "?"} °C · gain ${m.gain ?? "?"}</div>`
        + Object.values(r.selection).map((x) => `<div class="sel ${x.path ? "ok" : "bad"}"><b>${x.kind}</b>: ${esc(x.path || "nenalezen")}<br><span class="sub">${esc(x.reason)}</span>${(x.warnings || []).map((w) => `<br><span class="w">${esc(w)}</span>`).join("")}</div>`).join("");
    } catch (e) { $("#nj-prev").innerHTML = `<div class="sel bad">${esc(e.message)}</div>`; }
  };
  const calUi = async () => {
    let ms = []; try { ms = await api(`/api/instruments/${encodeURIComponent($("#nj-inst").value)}/masters`); } catch { /* shown in preview */ }
    $("#nj-cal").innerHTML = `<div class="sub" style="margin-top:8px">Automatický výběr lze přepsat – konkrétní soubor, nebo zpracování bez darku / flatu:</div>` + calibHtml(ms, {}, "njc");
    bindCalib("njc");
  };
  calUi();
  let instManual = false;           // the user picked the instrument himself -> only report
  const ident = async () => {
    const dir = $("#nj-dir").value, h = $("#nj-ident"); if (!dir) return;
    try {
      const r = await api("/api/identify", { method: "POST", body: { lights_dir: dir } });
      if (r.instrument_id) {
        const differs = $("#nj-inst").value !== r.instrument_id;
        if (differs && !instManual) { $("#nj-inst").value = r.instrument_id; calUi(); }
        h.className = "hint"; h.style.color = instManual && differs ? "var(--warn, #c80)" : "";
        h.textContent = `Rozpoznáno: ${r.reason}` + (instManual && differs ? " – ručně je vybrán jiný přístroj" : "");
      } else {
        h.className = "hint"; h.style.color = "var(--warn, #c80)";
        h.textContent = `Přístroj nerozpoznán: ${r.reason}. Vyberte ho ručně, nebo doplňte pravidla v Přístroje.`;
      }
    } catch (e) { h.textContent = ""; }
  };
  $("#nj-dir").onchange = async () => { await ident(); prev(); };
  $("#nj-inst").onchange = () => { instManual = true; ident(); prev(); calUi(); };
  $("#nj-go").onclick = async () => {
    try {
      const boxes = [...document.querySelectorAll("[data-field]")];
      const dboxes = [...document.querySelectorAll("[data-dir]")];
      const body = {
        instrument_id: $("#nj-inst").value, lights_dir: $("#nj-dir").value, name: $("#nj-name").value,
        params: { rate_min_arcsec_min: +$("#nj-rmin").value, rate_max_arcsec_min: +$("#nj-rmax").value,
          snr_threshold: +$("#nj-snr").value, segment_z_max: +$("#nj-seg").value, inject_n: +$("#nj-inj").value || 0,
          search_mode: $("#nj-mode").value, ...readCalib("njc") } };
      if ($("#nj-group")) body.params.frame_group = $("#nj-group").value;
      if (dboxes.length) {
        body.dirs = dboxes.filter((b) => b.checked).map((b) => b.dataset.dir);
        if (!body.dirs.length) { toast("Vyberte aspoň jeden adresář", true); return; }
      } else if (boxes.length) {
        body.fields = boxes.filter((b) => b.checked).map((b) => +b.dataset.field);
        if (!body.fields.length) { toast("Vyberte aspoň jedno zorné pole", true); return; }
      }
      const r = await api("/api/jobs", { method: "POST", body });
      const n = r.jobs.length;
      toast(n > 1 ? `${n} úloh (${dboxes.length ? "adresářů / " : ""}zorných polí) zařazeno do fronty – zpracují se postupně` : "Úloha zařazena do fronty");
      (r.skipped || []).forEach((s) => toast(s, true));
      go("job", r.jobs[0].id);
    } catch (e) { toast(e.message, true); }
  };
}

// Frames with different exposure / filter / binning are never stacked together: one group per job.
function groupsHtml(groups) {
  if (!groups || groups.length < 2) return "";
  return `<div class="warnbox">Snímky nejsou stejné (${groups.length} skupin podle expozice, filtru a binningu) – zpracuje se jen jedna skupina.
    <div class="row" style="margin-top:6px"><label for="nj-group" style="margin:0 8px 0 0">Skupina snímků</label>
    <select id="nj-group">${groups.map((g, i) => `<option value="${esc(g.key)}" ${i === 0 ? "selected" : ""}>${esc(g.label)}${i === 0 ? " (největší)" : ""}</option>`).join("")}</select></div></div>`;
}

function fieldsHtml(fs) {
  if (!fs || !fs.fields || fs.fields.length < 2 && !(fs.unassigned || []).length) return fs && fs.notes && fs.notes.length ? `<div class="sub">${fs.notes.map(esc).join("<br>")}</div>` : "";
  const tol = fs.tolerance_deg != null ? (fs.tolerance_deg * 60).toFixed(1) + "′" : "?";
  return `<h3>Zorná pole v adresáři: ${fs.fields.length}</h3>
    <div class="sub">Každé pole se zpracuje jako samostatná úloha, postupně jedno po druhém. Snímky s rozdílem pointace pod ${tol} patří k jednomu poli.</div>
    <table class="tbl"><tr><th></th><th>Pole</th><th>RA</th><th>Dec</th><th>Snímků</th><th>Dithering</th><th>Čas</th></tr>
    ${fs.fields.map((f) => `<tr><td><input type="checkbox" data-field="${f.index}" ${f.n >= 2 ? "checked" : "disabled title='jen 1 snímek'"}></td>
      <td>${esc(f.label)}</td><td class="mono">${f.ra_deg == null ? "–" : raStr(f.ra_deg)}</td><td class="mono">${f.dec_deg == null ? "–" : decStr(f.dec_deg)}</td>
      <td>${f.n}${f.n < 2 ? ' <span class="w">(málo)</span>' : f.n < 5 ? ' <span class="w" title="Synthetic tracking potřebuje aspoň 5 snímků; pole se zpracuje jen porovnáním s katalogem (zdroj navíc)">(jen zdroj navíc)</span>' : ""}</td><td>${f.spread_arcmin.toFixed(1)}′ · ${Math.round(f.spread_px)} px</td>
      <td>${f.mjd_first ? fmtMjd(f.mjd_first) + "–" + fmtMjd(f.mjd_last) : "–"}</td></tr>`).join("")}</table>
    ${(fs.notes || []).filter((n) => !n.startsWith("Nalezeno")).map((n) => `<div class="warnbox">${esc(n)}</div>`).join("")}`;
}
function dirsHtml(dirs) {
  const fld = (d) => {
    const fs = d.fields; if (!fs || !fs.fields || !fs.fields.length) return "–";
    if (fs.fields.length > 1) return `<span class="w">${fs.fields.length} pole → ${fs.fields.length} úloh</span>`;
    const f = fs.fields[0];
    return f.ra_deg == null ? "bez souřadnic" : `<span class="mono">${raStr(f.ra_deg)} ${decStr(f.dec_deg)}</span>`;
  };
  return `<h3>Adresáře se snímky: ${dirs.length}</h3>
    <div class="sub">Každý adresář se zpracuje jako samostatná úloha (pole), postupně jedno po druhém. Obsahuje-li adresář přesto více polí, rozdělí se ještě podle souřadnic.</div>
    <table class="tbl"><tr><th></th><th>Adresář</th><th>Snímků</th><th>Expozice / filtr</th><th>Střed pole</th><th>Mastery</th></tr>
    ${dirs.map((d) => `<tr><td><input type="checkbox" data-dir="${esc(d.path)}" ${d.frames >= 2 ? "checked" : "disabled title='jen 1 snímek'"}></td>
      <td class="mono">${esc(d.rel)}</td><td>${d.frames}${d.frames < 2 ? ' <span class="w">(málo)</span>' : d.frames < 5 ? ' <span class="w" title="Synthetic tracking potřebuje aspoň 5 snímků; jen porovnání s katalogem (zdroj navíc)">(jen zdroj navíc)</span>' : ""}</td>
      <td>${d.exptime ?? "?"} s · ${esc(d.filter)}</td><td>${fld(d)}</td>
      <td>${d.masters_ok ? "✓" : `<span class="w" title="${esc(d.masters_note)}">chybí</span>`}</td></tr>`).join("")}</table>
    ${dirs.flatMap((d) => (d.fields?.notes || []).filter((n) => !n.startsWith("Nalezeno")).map((n) => `<div class="warnbox">${esc(d.rel)}: ${esc(n)}</div>`)).join("")}`;
}
const fmtMjd = (mjd) => { const d = new Date((mjd - 40587) * 86400000); return d.toISOString().slice(11, 16); };

// ------------------------------------------------------------------ auto mode
async function viewWatch() {
  const [w, insts] = await Promise.all([api("/api/watch"), api("/api/instruments")]);
  const c = w.config || {};
  $("#main").innerHTML = `<div class="panel"><h2>Automatický režim</h2>
    <p class="sub">Program sleduje kořenový adresář i všechny podadresáře (typicky jeden podadresář = jedno pole z jednoho dalekohledu; dalekohled se pozná z hlavičky). Kořenový adresář lze přejmenovat nebo přesunout – stačí tu nastavit novou cestu, hotové sady se znovu nespustí. Jakmile adresář obsahuje nastavený počet snímků
    a po dobu ustálení se nic nemění, sám spustí úlohu. Výsledkem je seznam kandidátů v úlohách vlevo.</p>
    <div class="form">
      <label>Kořenový adresář</label><div class="row"><input id="w-root" value="${esc(c.root)}" spellcheck="false"><button class="btn" data-browse="w-root">Procházet…</button></div>
      <label>Přístroj</label><select id="w-inst"><option value="">podle hlavičky (pravidla v profilu)</option>${insts.map((i) => `<option value="${esc(i.id)}" ${i.id === c.instrument ? "selected" : ""}>${esc(i.name || i.id)}</option>`).join("")}</select>
      <label>Snímků na sadu</label><input id="w-n" type="number" value="${c.frames_per_set}">
      <div class="hint">Počet snímků v adresáři celkem. Obsahuje-li adresář více zorných polí, rozdělí se podle souřadnic v hlavičkách na samostatné úlohy (Nastavení → Zorná pole).</div>
      <label>Ustálení (s)</label><input id="w-settle" type="number" value="${c.settle_seconds}">
      <div class="hint">Jak dlouho se v adresáři nesmí nic změnit, než se sada považuje za úplnou.</div>
      <label>Interval kontroly (s)</label><input id="w-poll" type="number" value="${c.poll_seconds}">
      <label>Zapnuto</label><label class="switch"><input type="checkbox" id="w-en" ${c.enabled ? "checked" : ""}> sledovat (i po restartu programu)</label>
    </div>
    <div style="margin-top:14px;display:flex;gap:8px"><button class="btn primary" id="w-save">Uložit</button><button class="btn" id="w-scan">Zkontrolovat teď</button></div></div>
    <div class="panel"><h3 style="margin-top:0">Stav: ${w.running ? "běží" : "vypnuto"} · zpracovaných adresářů ${w.processed}</h3>
    ${Object.keys(w.pending || {}).length ? `<table class="tbl"><tr><th>Adresář</th><th>Přístroj</th><th>Snímků</th><th>Poslední změna</th></tr>${Object.entries(w.pending).map(([d, p]) => `<tr><td class="mono">${esc(d)}</td><td>${p.instrument ? esc(p.instrument) : `<span style="color:var(--warn,#c80)" title="${esc(p.error || "")}">nerozpoznán</span>`}${p.error ? `<div class="sub">${esc(p.error)}</div>` : ""}</td><td>${p.count} / ${c.frames_per_set}</td><td>před ${p.age_s} s</td></tr>`).join("")}</table>` : `<div class="sub">Žádné rozpracované sady.</div>`}
    <h3>Události</h3><pre class="log">${esc((w.messages || []).join("\n")) || "–"}</pre></div>`;
  bindBrowse();
  $("#w-save").onclick = async () => {
    await api("/api/watch", { method: "POST", body: { root: $("#w-root").value, instrument: $("#w-inst").value,
      frames_per_set: +$("#w-n").value, settle_seconds: +$("#w-settle").value, poll_seconds: +$("#w-poll").value, enabled: $("#w-en").checked } });
    toast("Uloženo"); viewWatch(); refreshStatus();
  };
  $("#w-scan").onclick = async () => { const r = await api("/api/watch/scan", { method: "POST" }); toast(r.started.length ? `Spuštěno ${r.started.length} úloh` : "Nic nového"); viewWatch(); refreshJobs(); };
}

// ------------------------------------------------------------------ instruments
const kvToText = (o) => Object.entries(o || {}).map(([k, v]) => `${k} = ${Array.isArray(v) ? v.join(", ") : v}`).join("\n");
function textToKv(t, list = false) {
  const o = {};
  for (const line of t.split("\n")) {
    const i = line.indexOf("="); if (i < 0) continue;
    const k = line.slice(0, i).trim(); let v = line.slice(i + 1).trim(); if (!k) continue;
    if (list) o[k] = v.split(",").map((s) => s.trim()).filter(Boolean);
    else o[k] = v !== "" && !isNaN(+v) ? +v : v;
  }
  return o;
}
async function viewInstruments(editId) {
  const insts = await api("/api/instruments");
  const cur = insts.find((i) => i.id === editId) || null;
  $("#main").innerHTML = `<div class="panel"><div class="jobhead"><div><h2>Přístroje</h2><p class="sub">Každý přístroj (dalekohled/objektiv + kamera) má vlastní adresář s master darky, flaty a biasy.
    Program vybírá mastery automaticky podle expozice, teploty, binningu, gainu a filtru.</p></div>
    <button class="btn primary" id="in-new">+ Nový přístroj</button></div>
    <table class="tbl"><tr><th>Název</th><th>Ohnisko</th><th>Pixel</th><th>Měřítko</th><th>Zorné pole</th><th>Mastery</th><th></th></tr>
    ${insts.map((i) => `<tr><td>${esc(i.name || i.id)}</td><td>${i.focal_length_mm} mm</td><td>${i.pixel_size_um} µm ×${i.binning}</td><td>${i.pixel_scale_arcsec}″/px</td>
      <td>${i.fov_deg ? i.fov_deg[0].toFixed(2) + "° × " + i.fov_deg[1].toFixed(2) + "°" : "–"}</td><td class="mono">${esc(i.masters_dir)}</td>
      <td><button class="btn small" data-edit="${esc(i.id)}">Upravit</button></td></tr>`).join("") || `<tr><td colspan="7" class="sub">Zatím žádný přístroj.</td></tr>`}</table></div>
    <div id="in-form"></div>`;
  $("#in-new").onclick = () => editInstrument(null);
  document.querySelectorAll("[data-edit]").forEach((b) => b.onclick = () => editInstrument(insts.find((i) => i.id === b.dataset.edit)));
  if (cur) editInstrument(cur);
}
function editInstrument(i) {
  const d = i || { id: "", name: "", focal_length_mm: 1000, pixel_size_um: 3.76, binning: 1, width_px: 0, height_px: 0, saturation_adu: 60000,
    gain_e_adu: 1, read_noise_e: 3, masters_dir: "", software_binning: 1, dark_exptime_tolerance_s: 0.5, dark_temp_tolerance_c: 2,
    allow_dark_scaling: false, require_flat: true, date_obs_reference: "start", time_offset_s: 0, observatory_code: "",
    observer_lat_deg: null, observer_lon_deg: null, observer_alt_m: null, site_from_header: true,
    keymap: {}, filter_aliases: {}, match: {}, notes: "" };
  const num = (k, lbl, step = "any", hint = "") => `<label>${lbl}</label><input id="i-${k}" type="number" step="${step}" value="${d[k] ?? ""}">${hint ? `<div class="hint">${hint}</div>` : ""}`;
  $("#in-form").innerHTML = `<div class="panel"><h2>${i ? "Upravit: " + esc(i.name || i.id) : "Nový přístroj"}</h2><div class="form">
    <label>Název</label><input id="i-name" value="${esc(d.name)}" placeholder="např. C14 + ASI6200 / 135mm + ASI2600">
    ${num("focal_length_mm", "Ohnisková vzdálenost (mm)")}
    ${num("pixel_size_um", "Velikost pixelu (µm)")}
    ${num("binning", "Binning kamery", 1)}
    <label>Měřítko</label><div id="i-scale" class="sub"></div>
    ${num("software_binning", "Softwarový binning", 1, "2 = rychlejší hledání u převzorkovaných dat (dlouhé ohnisko).")}
    ${num("saturation_adu", "Saturace (ADU)")}
    ${num("gain_e_adu", "Gain (e⁻/ADU)")}
    ${num("read_noise_e", "Šum čtení (e⁻)")}
    <label>Adresář master snímků</label><div class="row"><input id="i-masters_dir" value="${esc(d.masters_dir)}" spellcheck="false"><button class="btn" data-browse="i-masters_dir">Procházet…</button></div>
    ${num("dark_exptime_tolerance_s", "Tolerance expozice darku (s)")}
    ${num("dark_temp_tolerance_c", "Tolerance teploty darku (°C)")}
    <label>Škálování darku</label><label class="switch"><input type="checkbox" id="i-allow_dark_scaling" ${d.allow_dark_scaling ? "checked" : ""}> povolit přes master bias, když chybí dark se stejnou expozicí</label>
    <div class="hint">U CMOS s amp glow nedoporučeno.</div>
    <label>Flat povinný</label><label class="switch"><input type="checkbox" id="i-require_flat" ${d.require_flat ? "checked" : ""}> bez odpovídajícího flatu úlohu nespouštět</label>
    <label>DATE-OBS znamená</label><select id="i-date_obs_reference"><option value="start">začátek expozice</option><option value="mid">střed expozice</option></select>
    ${num("time_offset_s", "Korekce času (s)", "any", "Známé zpoždění závěrky/hodin, přičte se k času snímku.")}
    <label>Kód observatoře MPC</label><input id="i-observatory_code" value="${esc(d.observatory_code || "")}" maxlength="3" placeholder="např. 247">
    ${num("observer_lat_deg", "Zeměpisná šířka stanoviště (°)")}
    ${num("observer_lon_deg", "Zeměpisná délka stanoviště (°)")}
    ${num("observer_alt_m", "Nadmořská výška stanoviště (m)")}
    <label>Stanoviště z FITS</label><label class="switch"><input type="checkbox" id="i-site_from_header" ${d.site_from_header ? "checked" : ""}> souřadnice z hlavičky mají přednost před profilem</label>
    <div class="hint">MPC kód je vždy z profilu. Souřadnice z FITS se použijí jen při úplné šířce a délce.</div>
    <label>Pravidla pro automatický režim</label><textarea id="i-match" rows="3" placeholder="INSTRUME = ASI6200&#10;FOCALLEN = 2000">${esc(kvToText(d.match))}</textarea>
    <div class="hint">Podle těchto klíčů hlavičky se v automatickém režimu pozná, ke kterému přístroji snímky patří.</div>
    <label>Aliasy filtrů</label><textarea id="i-filter_aliases" rows="3" placeholder="Lum = L&#10;Clear = L">${esc(kvToText(d.filter_aliases))}</textarea>
    <label>Vlastní FITS klíče</label><textarea id="i-keymap" rows="3" placeholder="exptime = EXPOSURE&#10;temperature = CCD-TEMP">${esc(kvToText(d.keymap))}</textarea>
    <div class="hint">Jen když software kamery používá nestandardní názvy (výchozí: EXPTIME/EXPOSURE, FILTER, CCD-TEMP, GAIN, OFFSET, XBINNING, DATE-OBS…).</div>
    <label>Poznámka</label><input id="i-notes" value="${esc(d.notes)}">
  </div>
  <div style="margin-top:14px;display:flex;gap:8px"><button class="btn primary" id="i-save">Uložit</button>
    ${i ? `<button class="btn" id="i-masters">Zobrazit mastery</button><button class="btn danger" id="i-del">Smazat</button>` : ""}</div>
  <div id="i-mlist"></div></div>`;
  $("#i-date_obs_reference").value = d.date_obs_reference;
  bindBrowse($("#in-form"));
  const upd = () => { const s = 206.264806 * +$("#i-pixel_size_um").value * +$("#i-binning").value / +$("#i-focal_length_mm").value;
    $("#i-scale").textContent = `${s.toFixed(3)}″/px nativně, ${(s * +$("#i-software_binning").value).toFixed(3)}″/px při hledání`; };
  ["focal_length_mm", "pixel_size_um", "binning", "software_binning"].forEach((k) => $("#i-" + k).oninput = upd); upd();
  $("#i-save").onclick = async () => {
    const o = { ...d };
    for (const k of ["focal_length_mm", "pixel_size_um", "binning", "saturation_adu", "gain_e_adu", "read_noise_e", "software_binning",
      "dark_exptime_tolerance_s", "dark_temp_tolerance_c", "time_offset_s"]) o[k] = +$("#i-" + k).value;
    for (const k of ["observer_lat_deg", "observer_lon_deg", "observer_alt_m"]) {
      const v = $("#i-" + k).value.trim(); o[k] = v === "" ? null : +v;
    }
    o.binning = Math.round(o.binning); o.software_binning = Math.max(1, Math.round(o.software_binning));
    o.name = $("#i-name").value; o.masters_dir = $("#i-masters_dir").value; o.notes = $("#i-notes").value;
    o.allow_dark_scaling = $("#i-allow_dark_scaling").checked; o.require_flat = $("#i-require_flat").checked;
    o.site_from_header = $("#i-site_from_header").checked; o.observatory_code = $("#i-observatory_code").value.trim().toUpperCase();
    o.date_obs_reference = $("#i-date_obs_reference").value;
    o.match = textToKv($("#i-match").value); o.filter_aliases = textToKv($("#i-filter_aliases").value);
    for (const k in o.filter_aliases) o.filter_aliases[k] = String(o.filter_aliases[k]);
    o.keymap = textToKv($("#i-keymap").value, true);
    if (!o.id) o.id = o.name;
    if (!o.id) return toast("Zadejte název", true);
    try { const r = await api("/api/instruments", { method: "POST", body: o }); toast("Uloženo"); viewInstruments(r.id); } catch (e) { toast(e.message, true); }
  };
  if (i) {
    $("#i-del").onclick = async () => { if (!confirm("Smazat profil přístroje? (soubory masterů zůstanou)")) return; await api(`/api/instruments/${i.id}`, { method: "DELETE" }); viewInstruments(); };
    $("#i-masters").onclick = async () => {
      $("#i-mlist").innerHTML = `<div class="sub">Skenuji…</div>`;
      try {
        const l = await api(`/api/instruments/${i.id}/masters`);
        $("#i-mlist").innerHTML = `<h3>Nalezené master snímky (${l.length})</h3><table class="tbl"><tr><th>Typ</th><th>Expozice</th><th>Filtr</th><th>Teplota</th><th>Bin</th><th>Gain/Offset</th><th>Datum</th><th>Soubor</th></tr>
          ${l.map((m) => `<tr><td>${m.type}</td><td>${m.exptime ?? "–"}</td><td>${esc(m.filter)}</td><td>${m.temperature ?? "–"}</td><td>${m.xbin}x${m.ybin}</td><td>${m.gain ?? "–"}/${m.offset ?? "–"}</td><td>${esc((m.date_obs || "").slice(0, 10))}</td><td class="mono">${esc(m.path.split(/[\\/]/).pop())}</td></tr>`).join("")}</table>`;
      } catch (e) { $("#i-mlist").innerHTML = `<div class="sel bad">${esc(e.message)}</div>`; }
    };
  }
}

// ------------------------------------------------------------------ catalogs & settings
async function viewData() {
  const [st, s] = await Promise.all([api("/api/status"), api("/api/settings")]);
  const cat = (k, lbl) => { const c = st.catalogs[k]; return `<div class="sel ${c.present && !c.stale ? "ok" : "bad"}"><b>${lbl}</b>:
    ${c.present ? `${c.count.toLocaleString("cs-CZ")} objektů · staženo ${fmtDate(c.downloaded_at)} (${c.age_days.toFixed(1)} d)` : "nestaženo"}
    <div style="margin-top:6px;display:flex;gap:6px"><button class="btn small" data-upd="${k}" ${s.online.mpc_download ? "" : "disabled title='Povolte stahování z MPC níže'"}>Aktualizovat z MPC</button>
    <button class="btn small" data-imp="${k}">Importovat ze souboru…</button></div></div>`; };
  const t = st.catalog_task;
  $("#main").innerHTML = `<div class="panel"><h2>Katalogy známých objektů</h2>
    <p class="sub">Aktualizace jen ručně. Při chybě zůstane předchozí verze. Import ze souboru funguje i úplně offline
    (MPCORB.DAT, MPCORB.DAT.gz, CometEls.txt).</p>
    ${cat("mpcorb", "MPCORB (planetky)")}${cat("comets", "CometEls (komety)")}
    ${t.running ? `<div class="sub">${esc(t.kind)}: ${esc(t.message)} ${t.done ? (t.done / 1048576).toFixed(1) + " MB" : ""}${t.total ? " / " + (t.total / 1048576).toFixed(1) + " MB" : ""}</div>` : t.error ? `<div class="sel bad">${esc(t.error)}</div>` : ""}
  </div>
  <div class="panel"><h2>Nastavení</h2><div class="form">
    <label>Efemerida planet</label><input id="s-eph" value="${esc(s.ephemeris)}" placeholder="auto">
    <div class="hint">auto = JPL *.bsp (např. de440s.bsp) v adresáři katalogů, jinak přibalená JPL DE421 2020–2035, jinak ERFA. Aktuálně: ${esc(st.ephemeris)}</div>
    <label>Varovat při stáří MPCORB (dní)</label><input id="s-age" type="number" value="${s.mpcorb_max_age_days}">
    <label>ASTAP program</label><div class="row"><input id="s-astap" value="${esc(s.astap.executable)}" placeholder="prázdné = hledat v PATH / Program Files"></div>
    <label>ASTAP databáze</label><div class="row"><input id="s-astapdb" value="${esc(s.astap.database_dir)}" placeholder="prázdné = výchozí"><button class="btn" data-browse="s-astapdb">Procházet…</button></div>
    <label>UCAC4 (fotometrie)</label><div class="row"><input id="s-ucac4" value="${esc(s.ucac4_dir || "")}" placeholder="prázdné = bez katalogu (nulový bod ze známých planetek)"><button class="btn" data-browse="s-ucac4">Procházet…</button></div>
    <div class="hint">Složka s lokálním UCAC4 (obsahuje u4b\z001…z900, nebo přímo u4b). Nulový bod přibližné fotometrie se pak určí z hvězd pole s jasností APASS (pásmo podle filtru: R/clear → r, V → V, B → B). ${st.ucac4?.set ? (st.ucac4.ok ? `Nalezeno: ${esc(st.ucac4.path)} (${st.ucac4.zones} zón)` : `<span class="w">Ve složce nejsou zónové soubory z001…</span>`) : ""}</div>
    <label>GPU backend</label><select id="s-gpu"><option value="auto">automaticky (GPU NVIDIA/AMD, jinak CPU)</option><option value="cuda">NVIDIA (CUDA)</option><option value="rocm">AMD (ROCm)</option><option value="numpy">jen CPU</option></select>
    <label>Zorná pole</label><label class="switch"><input type="checkbox" id="s-fsplit" ${s.field_split ? "checked" : ""}> rozdělit adresář s více poli na samostatné úlohy (podle souřadnic v hlavičce)</label>
    <label>Tolerance pole</label><input id="s-ftol" type="number" step="0.05" min="0.05" max="0.6" value="${s.field_tolerance_fov}">
    <div class="hint">Snímky patří k jednomu poli, pokud se jejich pointace liší méně než tento zlomek menší strany zorného pole (0,25 = čtvrtina). Dithering musí být menší, sousední pole mozaiky vzdálenější.</div>
    <label>Limit paměti GPU</label><input id="s-gmem" type="number" step="0.05" min="0.1" max="1" value="${s.gpu.memory_fraction}">
    <label>Online: stahování MPC</label><label class="switch"><input type="checkbox" id="s-on-mpc" ${s.online.mpc_download ? "checked" : ""}> povolit tlačítko aktualizace</label>
    <label>Online: SkyBoT</label><label class="switch"><input type="checkbox" id="s-on-sky" ${s.online.skybot ? "checked" : ""}> křížová kontrola identifikací (připravuje se)</label>
    <label>Online: náhledy DSS</label><label class="switch"><input type="checkbox" id="s-on-dss" ${s.online.sky_images ? "checked" : ""}> výřez z přehlídky DSS/2MASS u kandidáta na vyžádání (CDS, záložně STScI a NASA SkyView)</label>
    <label>Online: astrometry.net</label><label class="switch"><input type="checkbox" id="s-on-ast" ${s.online.astrometry_net ? "checked" : ""}> záložní řešení pole (připravuje se)</label>
  </div><div style="margin-top:14px"><button class="btn primary" id="s-save">Uložit nastavení</button></div></div>`;
  $("#s-gpu").value = s.gpu.backend === "cupy" ? "auto" : s.gpu.backend;
  bindBrowse();
  document.querySelectorAll("[data-upd]").forEach((b) => b.onclick = async () => {
    try { await api("/api/catalogs/update", { method: "POST", body: { kind: b.dataset.upd } }); toast("Stahování spuštěno"); pollCatalog(); } catch (e) { toast(e.message, true); }
  });
  document.querySelectorAll("[data-imp]").forEach((b) => b.onclick = async () => {
    const p = desk() ? await desk().pick_file("") : prompt("Cesta k souboru (MPCORB.DAT[.gz] / CometEls.txt):");
    if (!p) return;
    try { await api("/api/catalogs/import", { method: "POST", body: { kind: b.dataset.imp, path: p } }); toast("Import spuštěn"); pollCatalog(); } catch (e) { toast(e.message, true); }
  });
  $("#s-save").onclick = async () => {
    await api("/api/settings", { method: "PUT", body: {
      field_split: $("#s-fsplit").checked, field_tolerance_fov: +$("#s-ftol").value,
      mpcorb_max_age_days: +$("#s-age").value, ephemeris: $("#s-eph").value || "auto", ucac4_dir: $("#s-ucac4").value.trim(),
      astap: { executable: $("#s-astap").value, database_dir: $("#s-astapdb").value },
      gpu: { backend: $("#s-gpu").value, memory_fraction: +$("#s-gmem").value },
      online: { mpc_download: $("#s-on-mpc").checked, skybot: $("#s-on-sky").checked, astrometry_net: $("#s-on-ast").checked, sky_images: $("#s-on-dss").checked } } });
    toast("Nastavení uloženo"); viewData(); refreshStatus();
  };
}
function pollCatalog() {
  const t = setInterval(async () => {
    const st = await api("/api/status");
    if (state.view === "data") viewData();
    refreshStatus();
    if (!st.catalog_task.running) { clearInterval(t); if (st.catalog_task.error) toast(st.catalog_task.error, true); else toast("Katalog aktualizován"); }
  }, 1500);
}

// ------------------------------------------------------------------ boot
refreshStatus(); refreshJobs(); go("home");
setInterval(refreshJobs, 2000);
setInterval(refreshStatus, 5000);

// ------------------------------------------------------------------ comparison with DSS
// Pop-up: our frames (mean of all frames = stars sharp, object as a short trail) next to
// a survey cut-out of the same field, both north up / east left.
let skyState = { survey: "DSS2c", vsx: (() => { try { return localStorage.getItem("st-vsx") !== "0"; } catch { return true; } })() };
async function openSky(job, c) {
  if (!c) return;
  document.querySelector(".sky-modal")?.remove();
  const scale = +(job.summary?.pixel_scale || 0);
  const frames = c.images?.frames;
  const size = frames?.size || 64;
  // "zdroj navíc" of older jobs stored the field centre's matrix – near a pole that is rotated
  // Since 13. 11. every "zdroj navíc" has its own matrix (twilight.local_sky_matrix); older jobs
  // stored one matrix of the field centre for all of them – recognisable as identical everywhere.
  // A matrix fitted from neighbouring candidates is only the fallback: near the pole (FRAM, Dec +84°)
  // the neighbours span degrees and the linear fit came out sheared and stretched
  const twOwn = c.source === "twilight" && c.sky_matrix
    && (state.allCands || state.cands).some((o) => o.source === "twilight" && o.sky_matrix && JSON.stringify(o.sky_matrix) !== JSON.stringify(c.sky_matrix));
  const W = (c.source === "twilight" && !twOwn ? localSkyMatrix(state.allCands || state.cands, c) : null) || c.sky_matrix || localSkyMatrix(state.allCands || state.cands, c);
  // A rotated source square only covers a smaller north-up/east-left square.
  // Use the largest fully covered field (45° rotation -> about 1/sqrt(2)).
  const plate = scale / 3600;
  const cover = W && plate ? Math.max((Math.abs(W[0][0]) + Math.abs(W[0][1])) / plate,
                                      (Math.abs(W[1][0]) + Math.abs(W[1][1])) / plate) : 1;
  const refPixel = frames ? trackAtReference(frames) : [size / 2, size / 2];
  const usable = W ? 2 * Math.min(refPixel[0], refPixel[1], size - refPixel[0], size - refPixel[1]) : size;
  const fovDeg = usable * scale / 3600 / Math.max(1, cover);
  const T = state.orientSky;
  const wrap = document.createElement("div");
  wrap.className = "modal sky-modal";
  const surveys = [["DSS2c", "DSS2 barevně"], ["DSS2r", "DSS2 červeně"], ["DSS2b", "DSS2 modře"], ["2MASSJ", "2MASS J"]];
  // Transient Name Server (supernovae, novae, other transients): its own search page with the
  // position filled in – no account or API needed; radius ≥ 15″ and 3 pixels
  const tnsR = Math.round(Math.max(15, 3 * scale));
  const tns = c.ra != null ? `https://www.wis-tns.org/search?ra=${c.ra.toFixed(6)}&decl=${c.dec.toFixed(6)}&radius=${tnsR}&coords_unit=arcsec` : null;
  const aladin = c.ra != null ? `https://aladin.cds.unistra.fr/AladinLite/?target=${encodeURIComponent(c.ra.toFixed(5) + " " + c.dec.toFixed(5))}&fov=${Math.max(fovDeg, 0.05).toFixed(3)}&survey=P%2FDSS2%2Fcolor` : null;
  wrap.innerHTML = `<div class="modal-box sky-box"><div class="modal-head"><b>#${c.id} · porovnání s přehlídkou</b>
      <span class="sub">${c.ra != null ? raStr(c.ra) + " " + decStr(c.dec) : ""} · pole ${(fovDeg * 60).toFixed(1)}′ · ${W ? "WCS: sever nahoře, východ vlevo" : "orientace přibližná"}</span>
      <button class="btn small" data-close>✕</button></div>
    <div class="sky-panels">
      <figure><div class="sky-img"><canvas id="sky-ours" width="360" height="360"></canvas>${SKYMARK}</div><figcaption>naše snímky – průměr všech (hvězdy ostré, objekt jako krátká stopa)</figcaption></figure>
      <figure><div class="sky-img" id="sky-ref"><div class="empty">načítám…</div></div><figcaption id="sky-src">přehlídka</figcaption></figure>
    </div>
    <div class="modal-foot"><div class="seg" id="sky-surv">${surveys.map(([k, l]) => `<button data-s="${k}" class="${skyState.survey === k ? "active" : ""}">${l}</button>`).join("")}</div>
      <label class="sl" title="Křížky proměnných hvězd v obrázku přehlídky – AAVSO VSX online, záložně VizieR, bez sítě katalog ASTAP; kliknutím se otevře hvězda ve VSX"><input type="checkbox" id="sky-vsx" ${skyState.vsx ? "checked" : ""}> proměnné VSX</label>
      ${tns ? `<a href="${tns}" target="_blank" rel="noopener" class="btn small" title="Transient Name Server: supernovy, novy a jiné transienty do ${tnsR}″ od kandidáta">TNS ↗</a>` : ""}
      ${aladin ? `<a href="${aladin}" target="_blank" rel="noopener" class="btn small">Aladin Lite ↗</a>` : ""}</div><div class="sky-vsx-list" id="sky-vsx-list"></div></div>`;
  document.body.appendChild(wrap);
  const close = () => { wrap.remove(); document.removeEventListener("keydown", esc_, true); };
  const esc_ = (e) => { if (e.key === "Escape" || e.key.toLowerCase() === "d") { e.stopImmediatePropagation(); e.preventDefault(); close(); } };
  document.addEventListener("keydown", esc_, true);
  wrap.onclick = (e) => { if (e.target === wrap || e.target.closest("[data-close]")) close(); };
  if (!T) $("#sky-src").textContent = "bez WCS – orientaci nelze určit";
  // our image
  const cv = $("#sky-ours");
  if (frames) {
    try {
      const d = await animLoad(job, c);
      const S = d.meta.size, m = new Float32Array(S * S), list = [...Array(d.meta.n).keys()];
      for (const i of list) {
        const sh = d.meta.shift ? d.meta.shift[i] : [0, 0];
        if (sh[0] || sh[1]) animShiftAdd(m, d.frames[i], S, sh[0], sh[1]); else for (let j = 0; j < m.length; j++) m[j] += d.frames[i][j];
      }
      for (let j = 0; j < m.length; j++) m[j] /= list.length;
      const smp = []; for (let j = 0; j < m.length; j += 3) smp.push(m[j]); smp.sort((a, b) => a - b);
      const med = smp[smp.length >> 1], dev = smp.map((v) => Math.abs(v - med)).sort((a, b) => a - b), sg = 1.4826 * dev[dev.length >> 1] || 1;
      const tmp = document.createElement("canvas"); tmp.width = tmp.height = S;
      const id = tmp.getContext("2d").createImageData(S, S);
      for (let j = 0; j < S * S; j++) {
        let t = ((m[j] - med) / sg + 1.5) / 20; t = t <= 0 ? 0 : t >= 1 ? 1 : Math.sqrt(t);
        const v = (t * 255) | 0; id.data[j * 4] = id.data[j * 4 + 1] = id.data[j * 4 + 2] = v; id.data[j * 4 + 3] = 255;
      }
      tmp.getContext("2d").putImageData(id, 0, 0);
      const ctx = cv.getContext("2d"); ctx.imageSmoothingEnabled = false;
      const map = W ? drawSkyWcs(ctx, tmp, W, refPixel, fovDeg) : null;
      if (!map) ctx.drawImage(tmp, 0, 0, 360, 360);
      // object path (first → last frame) as a thin line beside the trail
      const sc = 360 / S, tr = d.meta.track, qa = tr[0], qb = tr[tr.length - 1];
      const a = map ? map(qa) : [(qa[0] + 0.5) * sc, (qa[1] + 0.5) * sc];
      const b = map ? map(qb) : [(qb[0] + 0.5) * sc, (qb[1] + 0.5) * sc];
      ctx.strokeStyle = "rgba(255,196,40,.9)"; ctx.lineWidth = 1.5; ctx.beginPath();
      ctx.arc(a[0], a[1], 4, 0, 2 * Math.PI); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    } catch { /* image strip missing */ }
  } else {
    const im = new Image(); im.onload = () => { const ctx = cv.getContext("2d"); ctx.imageSmoothingEnabled = false; ctx.drawImage(im, 0, 0, 360, 360); };
    im.src = `/api/jobs/${job.id}/img/${c.images.median}`;
  }
  cv.style.transform = W ? "none" : cssT(T || [[1, 0], [0, -1]]);
  // survey image
  const load = async () => {
    const box = $("#sky-ref"), src = $("#sky-src");
    box.innerHTML = `<div class="empty">načítám ${esc(skyState.survey)}…</div>`;
    try {
      const r = await fetch(`/api/jobs/${job.id}/candidates/${c.id}/sky?survey=${skyState.survey}&fov=${fovDeg.toFixed(5)}`);
      if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.statusText); }
      const blob = await r.blob();
      box.innerHTML = `<img src="${URL.createObjectURL(blob)}">${SKYMARK}<svg class="skyvsx" viewBox="0 0 360 360"></svg>`;
      drawVsx();
      src.textContent = `${surveys.find((s) => s[0] === skyState.survey)[1]} · zdroj: ${r.headers.get("X-Sky-Source") || "?"}`;
    } catch (e) { box.innerHTML = `<div class="empty">${esc(e.message)}</div>`; src.textContent = "přehlídka"; }
  };
  // VSX variables: crosses in the survey image (north up, east left; the cut-out is fovDeg wide)
  let vsxStars = null, vsxSrc = "", vsxNote = "";
  const drawVsx = async () => {
    const svg = wrap.querySelector(".skyvsx"), list = $("#sky-vsx-list");
    if (!svg) return;
    svg.innerHTML = ""; list.innerHTML = "";
    if (!skyState.vsx || c.ra == null) return;
    try {
      if (vsxStars == null) {
        const r = await fetch(`/api/jobs/${job.id}/candidates/${c.id}/vsx?fov=${fovDeg.toFixed(5)}`);
        if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.statusText); }
        const d = await r.json();
        vsxStars = d.stars || []; vsxSrc = d.source || ""; vsxNote = d.note || "";
      }
      const k = 360 / fovDeg, inside = [];
      for (const v of vsxStars) {
        const x = 180 - v.de * k, y = 180 - v.dn * k;
        if (x >= -4 && x <= 364 && y >= -4 && y <= 364) inside.push([v, x, y]);
      }
      // dense fields (Milky Way, Gaia/OGLE variables): crosses for all (the brightest 60), names only for the
      // <= 3 nearest the centre (within 1/6 of the field) - the rest on hover, a click opens VSX
      const dc = (e) => Math.hypot(e[1] - 180, e[2] - 180);
      inside.sort((a, b) => (a[0].max_mag ?? 99) - (b[0].max_mag ?? 99));
      const shown = inside.slice(0, 60).sort((a, b) => dc(a) - dc(b));
      const named = new Set(shown.filter((e) => dc(e) <= 60).slice(0, 3).map((e) => e[0]));
      shown.forEach(([v, x, y]) => {
        const t = `${v.name}${v.type ? " · " + v.type : ""} · ${v.range}${v.period ? " · P " + v.period + " d" : ""}`;
        // the label on the side away from the centre, so that it never covers the candidate
        const right = x >= 180, below = y >= 180;
        const lab = named.has(v) ? `<text x="${right ? x + 8 : x - 8}" y="${below ? y + 15 : y - 8}" text-anchor="${right ? "start" : "end"}">${esc(v.name)}</text>` : "";
        svg.insertAdjacentHTML("beforeend", `<a href="${esc(v.url)}" target="_blank" rel="noopener"><title>${esc(t)}</title>
          <circle cx="${x}" cy="${y}" r="9" fill="transparent"/>
          <path d="M${x - 9} ${y}h5M${x + 4} ${y}h5M${x} ${y - 9}v5M${x} ${y + 4}v5" stroke="#ff5ad2" stroke-width="1.6"/>${lab}</a>`);
      });
      const more = inside.length > shown.length ? ` <span class="sub">(+${inside.length - shown.length} slabších neukázáno)</span>` : "";
      list.innerHTML = (shown.length
        ? "Proměnné v poli: " + shown.map(([v]) => `<a href="${esc(v.url)}" target="_blank" rel="noopener">${esc(v.name)}</a> <span class="sub">(${v.type ? esc(v.type) + ", " : ""}${esc(v.range)}${v.period ? ", P " + esc(v.period) + " d" : ""})</span>`).join(" · ") + more
        : `<span class="sub">V poli není žádná známá proměnná hvězda.</span>`)
        + ` <span class="sub">· zdroj: ${esc(vsxSrc)}${vsxNote ? " · " + esc(vsxNote) : ""}</span>`;
    } catch (e) { list.innerHTML = `<span class="sub">VSX: ${esc(e.message)}</span>`; }
  };
  $("#sky-vsx").onchange = (e) => {
    skyState.vsx = e.target.checked;
    try { localStorage.setItem("st-vsx", skyState.vsx ? "1" : "0"); } catch { /* storage unavailable */ }
    drawVsx();
  };
  wrap.querySelectorAll("[data-s]").forEach((b) => b.onclick = () => {
    skyState.survey = b.dataset.s;
    wrap.querySelectorAll("[data-s]").forEach((x) => x.classList.toggle("active", x === b)); load();
  });
  if (c.ra == null) { $("#sky-ref").innerHTML = `<div class="empty">Kandidát nemá souřadnice (bez WCS).</div>`; return; }
  load();
}
const SKYMARK = `<svg class="skymark" viewBox="0 0 360 360"><circle cx="180" cy="180" r="16" fill="none" stroke="rgba(80,200,255,.9)" stroke-width="1.5"/></svg>`;

// Local linear WCS for old jobs that predate candidate.sky_matrix. Nearby
// candidates sample the already solved WCS densely enough for a 4–6′ crop.
function localSkyMatrix(cands, target) {
  if (target.ra == null || target.dec == null) return null;
  const p = cands.filter((c) => c.ra != null && c.dec != null && c.x != null && c.y != null)
    .sort((a, b) => (a.x - target.x) ** 2 + (a.y - target.y) ** 2 - (b.x - target.x) ** 2 - (b.y - target.y) ** 2)
    .slice(0, 100);
  if (p.length < 6) return null;
  const cd = Math.cos(target.dec * Math.PI / 180);
  const rows = p.map((c) => { let dr = c.ra - target.ra; if (dr > 180) dr -= 360; if (dr < -180) dr += 360;
    return [c.x - target.x, c.y - target.y, dr * cd, c.dec - target.dec]; });
  const fit = (k) => {
    let n = 0, sx = 0, sy = 0, sq = 0, sxx = 0, syy = 0, sxy = 0, sxq = 0, syq = 0;
    for (const r of rows) { n++; sx += r[0]; sy += r[1]; sq += r[k]; sxx += r[0] * r[0]; syy += r[1] * r[1]; sxy += r[0] * r[1]; sxq += r[0] * r[k]; syq += r[1] * r[k]; }
    const mx = sx / n, my = sy / n, mq = sq / n, cxx = sxx / n - mx * mx, cyy = syy / n - my * my;
    const cxy = sxy / n - mx * my, cxq = sxq / n - mx * mq, cyq = syq / n - my * mq, det = cxx * cyy - cxy * cxy;
    return Math.abs(det) < 1e-9 ? null : [(cxq * cyy - cyq * cxy) / det, (cyq * cxx - cxq * cxy) / det];
  };
  const east = fit(2), north = fit(3); if (!east || !north) return null;
  const det = east[0] * north[1] - east[1] * north[0];
  return Math.abs(det) < 1e-12 ? null : [east, north];
}

function trackAtReference(meta) {
  const n = Math.min(meta.track?.length || 0, meta.dt_min?.length || 0);
  if (!n) return [meta.size / 2, meta.size / 2];
  let st = 0, stt = 0, sx = 0, sy = 0, stx = 0, sty = 0;
  for (let i = 0; i < n; i++) { const t = +meta.dt_min[i]; st += t; stt += t * t; sx += meta.track[i][0]; sy += meta.track[i][1]; stx += t * meta.track[i][0]; sty += t * meta.track[i][1]; }
  const den = n * stt - st * st;
  if (Math.abs(den) < 1e-9) return [sx / n, sy / n];
  return [(sx * stt - st * stx) / den, (sy * stt - st * sty) / den];
}

function drawSkyWcs(ctx, source, W, ref, fovDeg) {
  if (!W || !fovDeg) return null;
  const p = fovDeg / 360, a = -W[0][0] / p, cc = -W[0][1] / p, b = -W[1][0] / p, d = -W[1][1] / p;
  const e = 180 - a * ref[0] - cc * ref[1], f = 180 - b * ref[0] - d * ref[1];
  ctx.save(); ctx.setTransform(a, b, cc, d, e, f); ctx.drawImage(source, 0, 0); ctx.restore();
  return (q) => [a * q[0] + cc * q[1] + e, b * q[0] + d * q[1] + f];
}


// ------------------------------------------------------------------ approximate ephemeris
// Linear motion (constant rate along a great circle at the measured PA) from the reference
// epoch of the series – like Tycho Tracker's "where will it be in X hours". The uncertainty
// comes from the scatter of the three sub-stack positions around the fitted line (floor
// 0.25 px); the real path bends (Earth's motion, parallax), so after a day the error is
// usually larger than shown – the note says so.
const EPH_HOURS = [1, 2, 3, 4, 6, 8, 12, 18, 24, 36, 48];
function ephemPos(ra0, dec0, paDeg, rhoArcsec) {
  const r = Math.PI / 180, d0 = dec0 * r, pa = paDeg * r, rho = rhoArcsec / 3600 * r;
  const sd = Math.sin(d0) * Math.cos(rho) + Math.cos(d0) * Math.sin(rho) * Math.cos(pa);
  const dec = Math.asin(Math.max(-1, Math.min(1, sd)));
  const ra = ra0 * r + Math.atan2(Math.sin(pa) * Math.sin(rho) * Math.cos(d0), Math.cos(rho) - Math.sin(d0) * sd);
  return [((ra / r) % 360 + 360) % 360, dec / r];
}
function ephemSigma(c, scale) {
  // sub-stack positions → tangent-plane offsets (″) vs. time (min); residuals of a straight line
  const P = (c.positions || []).filter((p) => p.ok !== false && p.ra != null && p.dec != null && p.dt_min != null);
  const floor = 0.25 * (scale || 1);
  if (P.length < 2) return { pos: floor, rate: null, n: P.length };
  const cd = Math.cos(c.dec * Math.PI / 180);
  const t = P.map((p) => p.dt_min), tm = t.reduce((a, b) => a + b, 0) / t.length;
  const sxx = t.reduce((a, b) => a + (b - tm) ** 2, 0);
  let rss = 0;
  for (const k of [0, 1]) {
    const y = P.map((p) => k === 0 ? (((p.ra - c.ra + 540) % 360) - 180) * cd * 3600 : (p.dec - c.dec) * 3600);
    const ym = y.reduce((a, b) => a + b, 0) / y.length;
    const b1 = sxx > 0 ? y.reduce((a, v, i) => a + (t[i] - tm) * (v - ym), 0) / sxx : 0;
    y.forEach((v, i) => { rss += (v - ym - b1 * (t[i] - tm)) ** 2; });
  }
  const dof = Math.max(1, 2 * P.length - 4);
  const pos = Math.max(floor, Math.sqrt(rss / dof));
  return { pos, rate: sxx > 0 ? pos / Math.sqrt(sxx) : null, n: P.length };
}
function fmtUtc(mjd) {
  const d = new Date((mjd - 40587) * 86400000);
  return d.toISOString().replace("T", " ").slice(0, 16);
}
function openEphem(job, c) {
  if (!c) return;
  document.querySelector(".eph-modal")?.remove();
  const mjd0 = job.summary?.mjd_ref;
  const wrap = document.createElement("div");
  wrap.className = "modal eph-modal";
  if (c.ra == null || c.rate_arcsec_min == null || c.pa_deg == null || mjd0 == null) {
    wrap.innerHTML = `<div class="modal-box"><div class="modal-head"><b>#${c.id} · efemerida</b><button class="btn small" data-close>✕</button></div>
      <div class="empty">Kandidát nemá polohu, rychlost nebo čas reference (WCS) – efemeridu nelze spočítat.</div></div>`;
  } else {
    const scale = +(job.summary?.pixel_scale || 0);
    const sg = ephemSigma(c, scale);
    const nowMjd = Date.now() / 86400000 + 40587;
    const rows = EPH_HOURS.map((h) => ({ h, mjd: mjd0 + h / 24 }));
    const sinceH = (nowMjd - mjd0) * 24;
    if (sinceH > 0 && sinceH < 24 * 14) rows.push({ h: sinceH, mjd: nowMjd, now: true });
    rows.sort((a, b) => a.h - b.h);
    const unc = (hh) => sg.rate != null ? Math.hypot(sg.pos, sg.rate * hh * 60) : null;
    const line = (r) => {
      const [ra, dec] = ephemPos(c.ra, c.dec, c.pa_deg, c.rate_arcsec_min * r.h * 60);
      const u = unc(r.h);
      return { ...r, ra, dec, u, txt: `${fmtUtc(r.mjd)}  ${("+" + r.h.toFixed(r.now ? 1 : 0) + " h").padStart(8)}  ${raStr(ra)}  ${decStr(dec)}  ${u != null ? "±" + (u >= 120 ? (u / 60).toFixed(1) + "′" : u.toFixed(0) + "″") : ""}` };
    };
    const L = rows.map(line);
    const text = [`# ${job.name} · kandidát #${c.id} · lineární aproximace (velký kruh)`,
      `# reference ${fmtUtc(mjd0)} UTC  RA ${raStr(c.ra)}  Dec ${decStr(c.dec)}  ${c.rate_arcsec_min.toFixed(3)}″/min  PA ${c.pa_deg.toFixed(1)}°`,
      "# UTC               +čas      RA (J2000)    Dec          nejistota", ...L.map((r) => r.txt)].join("\n");
    wrap.innerHTML = `<div class="modal-box eph-box"><div class="modal-head"><b>#${c.id} · přibližná efemerida</b>
        <span class="sub">${c.rate_arcsec_min.toFixed(3)}″/min · PA ${c.pa_deg.toFixed(1)}° · reference ${fmtUtc(mjd0)} UTC</span>
        <button class="btn small" data-close>✕</button></div>
      <div class="eph-body"><table class="tbl"><thead><tr><th>UTC</th><th>za</th><th>RA (J2000)</th><th>Dec</th><th title="z rozptylu poloh ve třech podstackách kolem přímky (min. 0,25 px); jen statistická část">± nejistota</th></tr></thead>
        <tbody>${L.map((r) => `<tr class="${r.now ? "eph-now" : ""}"><td class="mono">${fmtUtc(r.mjd)}${r.now ? " (teď)" : ""}</td><td class="mono">+${r.h.toFixed(r.now ? 1 : 0)} h</td><td class="mono">${raStr(r.ra)}</td><td class="mono">${decStr(r.dec)}</td><td class="mono">${r.u != null ? "±" + (r.u >= 120 ? (r.u / 60).toFixed(1) + "′" : r.u.toFixed(0) + "″") : "–"}</td></tr>`).join("")}
          <tr><td colspan="5"><label>Vlastní čas: za <input type="number" id="eph-h" min="0" max="720" step="0.5" value="" style="width:70px"> h</label> <span class="mono" id="eph-out"></span></td></tr></tbody></table>
        <div class="rng"><div class="rng-head"><b>Ranging – oblast možných poloh (jako NEOCP)</b>
          <div class="seg" id="rng-model">${[["all", "vše (e ≤ 1)"], ["mb", "hlavní pás"], ["neo", "NEO"], ["comet", "kometa"]].map(([k, l]) => `<button data-m="${k}" class="${(state.rngModel || "all") === k ? "active" : ""}">${l}</button>`).join("")}</div>
          <div class="seg" id="rng-rho" title="Nejmenší uvažovaná vzdálenost od Země. Velmi blízká řešení (pod 0,1 AU) natahují oblast na stupně; pomalý objekt (pár ″/min) tak blízko je málo pravděpodobný, ale ne vyloučený.">${[[0.003, "od Země ≥ 0,003 AU"], [0.05, "≥ 0,05 AU"], [0.1, "≥ 0,1 AU"]].map(([k, l]) => `<button data-r="${k}" class="${(state.rngRho || 0.003) === k ? "active" : ""}">${l}</button>`).join("")}</div></div>
          <div id="rng-out" class="empty">počítám…</div></div>
        <p class="sub">Lineární aproximace: stálá rychlost a směr podél velkého kruhu z polohy v čase reference. Skutečná dráha se zakřivuje (pohyb Země, paralaxa), takže po desítkách hodin bývá odchylka větší než uvedená nejistota – u planetek hlavního pásu typicky jednotky obloukových minut za 1–2 dny. Nejistota je jen statistická (${sg.n} podstacky).</p></div>
      <div class="modal-foot"><span></span><button class="btn small" id="eph-copy">Kopírovat jako text</button></div></div>`;
    wrap.querySelector("#eph-h").oninput = (e) => {
      const h = parseFloat(e.target.value);
      $("#eph-out").textContent = isFinite(h) ? (() => { const r = line({ h, mjd: mjd0 + h / 24 }); return `${fmtUtc(r.mjd)} UTC  ${raStr(r.ra)}  ${decStr(r.dec)}  ${r.u != null ? "±" + (r.u >= 120 ? (r.u / 60).toFixed(1) + "′" : r.u.toFixed(0) + "″") : ""}`; })() : "";
    };
    wrap.querySelector("#eph-copy").dataset.base = text;
    wrap.querySelector("#eph-copy").onclick = async () => {
      try { await navigator.clipboard.writeText(text); toast("Efemerida zkopírována."); }
      catch { const ta = document.createElement("textarea"); ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove(); toast("Efemerida zkopírována."); }
    };
  }
  document.body.appendChild(wrap);
  if (wrap.querySelector("#rng-model")) {
    wrap.querySelectorAll("#rng-model button").forEach((b) => b.onclick = () => {
      state.rngModel = b.dataset.m;
      wrap.querySelectorAll("#rng-model button").forEach((x) => x.classList.toggle("active", x === b));
      loadRanging(job, c, b.dataset.m);
    });
    wrap.querySelectorAll("#rng-rho button").forEach((b) => b.onclick = () => {
      state.rngRho = +b.dataset.r;
      wrap.querySelectorAll("#rng-rho button").forEach((x) => x.classList.toggle("active", x === b));
      loadRanging(job, c, state.rngModel || "all");
    });
    loadRanging(job, c, state.rngModel || "all");
  }
  const close = () => { wrap.remove(); document.removeEventListener("keydown", esc_, true); };
  const esc_ = (e) => { if (e.target?.id === "eph-h" && e.key !== "Escape") return; if (e.key === "Escape" || e.key.toLowerCase() === "e") { e.stopImmediatePropagation(); e.preventDefault(); close(); } };
  document.addEventListener("keydown", esc_, true);
  wrap.onclick = (e) => { if (e.target === wrap || e.target.closest("[data-close]")) close(); };
}

// ranging: cloud of possible positions from admissible orbits (server: synthtrack/ranging.py)
const fmtArcmin = (v) => v >= 120 ? (v / 60).toFixed(1) + "°" : v >= 10 ? v.toFixed(0) + "′" : v.toFixed(1) + "′";
async function loadRanging(job, c, model) {
  const out = document.getElementById("rng-out");
  if (!out) return;
  out.className = "empty"; out.textContent = "počítám…";
  let r;
  try { r = await api(`/api/jobs/${job.id}/candidates/${c.id}/ranging?model=${model}&rho_min=${state.rngRho || 0.003}`); }
  catch (e) { out.textContent = String(e.message || e); return; }
  if (!document.body.contains(out)) return;
  if (!r.n) { out.textContent = r.note || "Žádná přípustná dráha."; return; }
  out.className = "";
  const T = r.times;
  let sel = T.findIndex((t) => t.hours === 24); if (sel < 0) sel = 0;
  const pr = (a) => a.map((v) => v.toFixed(2)).join(" / ");
  out.innerHTML = `<table class="tbl rng-tbl"><thead><tr><th>za</th><th>UTC</th><th>střed RA</th><th>střed Dec</th>
      <th title="95 % vzorků: délka × šířka oblasti a směr její delší osy (od severu přes východ)">oblast 95 %</th><th title="vzdálenost lineární předpovědi od středu oblasti">lineární</th></tr></thead>
    <tbody>${T.map((t, i) => `<tr data-i="${i}" class="${i === sel ? "sel" : ""}"><td class="mono">+${t.hours} h</td><td class="mono">${fmtUtc(t.mjd)}</td>
      <td class="mono">${raStr(t.ra)}</td><td class="mono">${decStr(t.dec)}</td>
      <td class="mono">${fmtArcmin(t.major_arcmin)} × ${fmtArcmin(t.minor_arcmin)}, PA ${t.major_pa_deg.toFixed(0)}°</td><td class="mono">${fmtArcmin(t.linear_offset_arcmin)}</td></tr>`).join("")}</tbody></table>
    <div class="rng-plot"><canvas id="rng-cv" width="380" height="380"></canvas>
      <div class="rng-info"><div><b>${esc(r.model_label)}</b></div>
        <div>${r.n} přípustných drah</div>
        <div>${r.rho_min > 0.004 ? `uvažováno jen ≥ ${r.rho_min} AU od Země · ` : ""}vzdálenost od Země ${r.rho_au[0] < 0.1 ? r.rho_au[0].toFixed(3) : r.rho_au[0].toFixed(2)}–${r.rho_au[1].toFixed(2)} AU (95 %)</div>
        <div>a ${pr(r.a_au)} AU</div><div>e ${pr(r.e)}</div><div>q ${pr(r.q_au)} AU</div>
        <div class="sub">(5 % / medián / 95 %)</div>
        <div class="sub" style="margin-top:8px">body = možné polohy · <span style="color:#f90">●</span> lineární předpověď · čárkovaně zorné pole ${r.fov_deg ? r.fov_deg[0].toFixed(1) + "° × " + r.fov_deg[1].toFixed(1) + "°" : "neznámé"} (bez rotace) · sever nahoře, východ vlevo · klik na řádek = jiný čas</div></div></div>
    <p class="sub">Hustota bodů závisí na předpokladu (vzorkování rovnoměrně v log vzdálenosti); podstatný je rozsah oblasti. Dráhy dvou těles (Slunce), bez poruch planet; nejistota polohy a rychlosti z podstacků (min. FWHM/SNR, těžké okraje).</p>`;
  const draw = (i) => {
    const t = T[i], cv = document.getElementById("rng-cv");
    if (!cv) return;
    const ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
    ctx.fillStyle = "#0b0e13"; ctx.fillRect(0, 0, W, H);
    const pts = t.points;
    let ext = 0;
    for (const [x, y] of pts) ext = Math.max(ext, Math.abs(x), Math.abs(y));
    ext = Math.max(ext, Math.abs(t.linear_xy[0]), Math.abs(t.linear_xy[1]), 0.5);
    const fov = r.fov_deg ? [r.fov_deg[0] * 60, r.fov_deg[1] * 60] : null;
    if (fov) ext = Math.max(ext, Math.min(fov[0], fov[1]) * 0.6);
    const sc = (W / 2 - 14) / (ext * 1.05);
    const X = (x) => W / 2 - x * sc, Y = (y) => H / 2 - y * sc;        // east left, north up
    if (fov) {
      ctx.setLineDash([5, 4]); ctx.strokeStyle = "#5b6b80"; ctx.lineWidth = 1;
      ctx.strokeRect(W / 2 - fov[0] / 2 * sc, H / 2 - fov[1] / 2 * sc, fov[0] * sc, fov[1] * sc); ctx.setLineDash([]);
    }
    ctx.fillStyle = "rgba(120,190,255,0.55)";
    for (const [x, y] of pts) ctx.fillRect(X(x) - 1, Y(y) - 1, 2, 2);
    ctx.strokeStyle = "#fff"; ctx.beginPath(); ctx.moveTo(W / 2 - 6, H / 2); ctx.lineTo(W / 2 + 6, H / 2); ctx.moveTo(W / 2, H / 2 - 6); ctx.lineTo(W / 2, H / 2 + 6); ctx.stroke();
    ctx.fillStyle = "#f90"; ctx.beginPath(); ctx.arc(X(t.linear_xy[0]), Y(t.linear_xy[1]), 3.5, 0, 2 * Math.PI); ctx.fill();
    // scale bar
    const nice = [0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200].find((v) => v * sc > 50) || 1200;
    ctx.strokeStyle = "#ccc"; ctx.beginPath(); ctx.moveTo(14, H - 14); ctx.lineTo(14 + nice * sc, H - 14); ctx.stroke();
    ctx.fillStyle = "#ccc"; ctx.font = "11px sans-serif"; ctx.fillText(fmtArcmin(nice), 14, H - 20);
    ctx.fillText("S", W / 2 - 4, 12); ctx.fillText("V", 4, H / 2 + 4);
    ctx.fillText(`+${t.hours} h`, W - 44, 14);
  };
  draw(sel);
  out.querySelectorAll(".rng-tbl tbody tr").forEach((tr) => tr.onclick = () => {
    out.querySelectorAll(".rng-tbl tbody tr").forEach((x) => x.classList.toggle("sel", x === tr));
    draw(+tr.dataset.i);
  });
  const btn = document.getElementById("eph-copy");
  if (btn) {
    const extra = ["", `# ranging (${r.model_label}), ${r.n} drah: střed oblasti a rozsah 95 %`,
      ...T.map((t) => `${fmtUtc(t.mjd)}  ${("+" + t.hours + " h").padStart(8)}  ${raStr(t.ra)}  ${decStr(t.dec)}  ${fmtArcmin(t.major_arcmin)} x ${fmtArcmin(t.minor_arcmin)} PA ${t.major_pa_deg.toFixed(0)}`)].join("\n");
    const old = btn.onclick;
    btn.dataset.rng = extra;
    btn.onclick = async () => {
      const base = btn.dataset.base || "";
      try { await navigator.clipboard.writeText(base + "\n" + extra); toast("Efemerida i ranging zkopírovány."); }
      catch { old && old(); }
    };
  }
}
