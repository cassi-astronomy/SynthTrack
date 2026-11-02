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
    for (const g of s.gpus) chips.push(`<span class="chip"><b>${esc(g.name.replace("NVIDIA GeForce ", ""))}</b> · ${g.util_pct ?? "?"}% · ${((g.total_mb - g.free_mb) / 1024).toFixed(1)}/${(g.total_mb / 1024).toFixed(0)} GB</span>`);
  }
  chips.push(`<span class="chip ${s.backend.backend === "numpy" ? "warn" : "ok"}" title="${esc(s.backend.note)}">backend <b>${s.backend.backend === "numpy" ? "CPU" : "CUDA"}</b>${s.backend.backend !== "numpy" ? ` · limit ${Math.round(s.gpu_memory_fraction * 100)}%` : ""}</span>`);
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
    <div class="tabs">${[["cands", "Kandidáti"], ["known", "Známé objekty"], ["summary", "Souhrn"], ["export", "Export"], ["log", "Log"]]
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
    </div><div style="display:flex;gap:8px">
      ${busy ? `<button class="btn" id="jcancel">Zrušit</button>` : `<button class="btn" id="jrerun">Spustit znovu</button>`}
      <button class="btn danger" id="jdel" ${busy ? "disabled" : ""}>Smazat</button></div></div>`;
  const c = $("#jcancel"); if (c) c.onclick = async () => { await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); toast("Rušení…"); };
  const r = $("#jrerun"); if (r) r.onclick = async () => {
    const thr = prompt("Práh SNR (prázdné = beze změny):", job.params?.snr_threshold ?? "");
    const params = thr ? { snr_threshold: parseFloat(thr) } : {};
    const nj = await api(`/api/jobs/${job.id}/rerun`, { method: "POST", body: { params } });
    toast("Úloha zařazena"); go("job", nj.id);
  };
  const d = $("#jdel"); if (d) d.onclick = async () => { if (!confirm("Smazat úlohu včetně výsledků?")) return; try { await api(`/api/jobs/${job.id}`, { method: "DELETE" }); go("home"); } catch (e) { toast("Smazání selhalo: " + e.message); } };
}

async function renderTab(job) {
  const body = $("#tabbody");
  if (state.tab === "log") { body.innerHTML = `<pre class="log">${esc(await api(`/api/jobs/${job.id}/log`))}</pre>`; return; }
  if (job.status !== "done") { body.innerHTML = `<div class="empty">Výsledky budou k dispozici po dokončení úlohy.</div>`; return; }
  if (state.tab === "cands") return renderCands(job);
  if (state.tab === "known") return renderKnown(job);
  if (state.tab === "summary") return renderSummary(job);
  if (state.tab === "export") return renderExport(job);
}

async function renderCands(job) {
  state.cands = await api(`/api/jobs/${job.id}/candidates`);
  state.orientSky = skyTransform(state.cands);
  const counts = { unreviewed: 0, accepted: 0, unsure: 0, rejected: 0, injected: 0, all: state.cands.filter((c) => c.status !== "injected").length };
  state.cands.forEach((c) => {
    counts[c.status] = (counts[c.status] || 0) + 1;
    if (c.status === "rejected") { const k = c.reviewed ? "rej_manual" : "rej_auto"; counts[k] = (counts[k] || 0) + 1; }
  });
  const filters = [["unreviewed", "neprohlédnuté"], ["accepted", "přijaté"], ["unsure", "nejisté"], ["rej_manual", "zamítnuté mnou"], ["rej_auto", "zamítnuté automaticky"], ["all", "vše"]];
  if (counts.injected) filters.push(["injected", "injekce"]);
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
      <span class="keys"><kbd>J</kbd>/<kbd>K</kbd> pohyb · <kbd>A</kbd> přijmout · <kbd>R</kbd> zamítnout · <kbd>U</kbd> nejisté · <kbd>Z</kbd> zpět · <kbd>P</kbd> pauza · <kbd>X</kbd> kříž · <kbd>D</kbd> DSS · <kbd>←</kbd>/<kbd>→</kbd> snímek</span>
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
  document.querySelectorAll("[data-f]").forEach((b) => b.onclick = () => { state.filter = b.dataset.f; state.focus = 0; renderCands(job); });
  $("#undo").onclick = () => undoStatus(job);
  drawCards(job);
}

function visibleCands() {
  const fOk = (c) => state.filter === "all" ? c.status !== "injected"
    : state.filter === "rej_auto" ? c.status === "rejected" && !c.reviewed
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
    return `<div class="card st-${c.status} ${i === state.focus ? "focus" : ""}" data-cid="${c.id}" data-i="${i}">
      <div class="card-head"><span class="id">#${c.id}</span><span>${badge} ${flags} ${starGroup}${c.injected != null ? ` <span class="badge">injekce #${c.injected}</span>` : ""}</span><span class="snr">${c.confidence ? `<span class="badge conf conf-${c.confidence}" title="${esc((c.confidence_reasons || []).length ? "Proč ne vyšší: " + c.confidence_reasons.join("; ") : "Splňuje všechna kritéria")}">spolehlivost: ${CONF[c.confidence][1]}</span> ` : ""}${c.ml_score != null ? `<span class="badge ml" title="Pravděpodobnost skutečného objektu podle klasifikátoru">ML ${Math.round(c.ml_score * 100)} %</span> ` : ""}SNR ${c.snr.toFixed(1)}</span></div>
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
        ${c.images?.snr ? `<div class="thumb"><img src="${img(c, "snr")}" loading="lazy">${CROSS}<div class="cap">mapa SNR podél stopy</div></div>` : ""}
        <div class="facts">
          <span class="k">Rychlost</span><span class="v">${c.rate_arcsec_min != null ? c.rate_arcsec_min.toFixed(3) + "″/min" : "–"}</span>
          <span class="k">PA</span><span class="v">${c.pa_deg != null ? c.pa_deg.toFixed(1) + "°" : "–"}</span>
          <span class="k">RA</span><span class="v">${raStr(c.ra)}</span>
          <span class="k">Dec</span><span class="v">${decStr(c.dec)}</span>
          <span class="k">Mag</span><span class="v">${c.mag != null ? c.mag.toFixed(2) : (c.known?.pred_mag != null ? "≈" + c.known.pred_mag + " (předpověď)" : "–")}</span>
          <span class="k">x, y</span><span class="v">${c.x.toFixed(1)}, ${c.y.toFixed(1)}</span>
          <span class="k">Snímky</span><span class="v">${c.frames_used}/${job.n_frames}</span>
          <span class="k">Seg SNR</span><span class="v">${c.seg_snr.map((v) => v.toFixed(1)).join(" / ")}</span>
          <span class="k">Medián/průměr</span><span class="v">${c.median_ratio.toFixed(2)}</span>
          <span class="k">χ² snímků</span><span class="v">${(c.chi2 ?? 0).toFixed(2)}</span>
          <span class="k">SNR pro −v</span><span class="v">${(c.snr_reverse ?? 0).toFixed(1)}</span>
          <span class="k">FWHM</span><span class="v">${c.fwhm != null ? c.fwhm.toFixed(2) + " px" : "–"}</span>
          ${c.throughput_min != null ? `<span class="k">Propustnost min</span><span class="v">${Math.round(c.throughput_min * 100)} %</span>` : ""}
        </div>
      </div>
      <div class="card-actions">
        <button class="btn small ${c.status === "accepted" ? "ok" : ""}" data-set="accepted">Přijmout</button>
        <button class="btn small ${c.status === "unsure" ? "warnb" : ""}" data-set="unsure">Nejisté</button>
        <button class="btn small ${c.status === "rejected" ? "badb" : ""}" data-set="rejected">Zamítnout</button>
        <button class="btn small" data-sky title="Porovnat s přehlídkou DSS / 2MASS (D)">DSS</button>
        ${c.status !== "unreviewed" ? `<button class="btn small" data-set="unreviewed" title="Zrušit označení (zpět mezi neprohlédnuté)">Zrušit označení</button>` : ""}
        <input placeholder="trkSub" value="${esc(c.trksub || "")}" maxlength="7" data-trk>
      </div></div>`;
  }).join("");
  document.querySelectorAll(".card").forEach((card) => {
    card.onclick = () => { state.focus = +card.dataset.i; markFocus(); };
    card.querySelectorAll("[data-set]").forEach((b) => b.onclick = (e) => { e.stopPropagation(); setStatus(job, +card.dataset.cid, b.dataset.set); });
    const sg = card.querySelector("[data-star-group]");
    if (sg) sg.onclick = (e) => { e.stopPropagation(); const k = sg.dataset.starGroup; state.starGroupsOpen.has(k) ? state.starGroupsOpen.delete(k) : state.starGroupsOpen.add(k); drawCards(job); };
    const sk = card.querySelector("[data-sky]");
    if (sk) sk.onclick = (e) => { e.stopPropagation(); openSky(job, state.cands.find((x) => x.id === +card.dataset.cid)); };
    const t = card.querySelector("[data-trk]");
    t.onchange = () => api(`/api/jobs/${job.id}/candidates/${card.dataset.cid}`, { method: "PATCH", body: { trksub: t.value } });
  });
  animAttach(job);
}

// ------------------------------------------------------------------ animation viewer
// Frames come as one 8-bit strip per candidate (q = C + B·asinh(x/A), x in σ);
// sub-stacks, black/white level and gamma are computed here, so they can be
// changed instantly for all candidates.
const ANIM_DEF = { k: 4, black: -1.5, white: 8, gamma: 1.0, speed: 350, play: true, align: true, cross: true,
  orient: "sky", fix: true, skipObs: true };
state.anim = Object.assign({}, ANIM_DEF, (() => { try { return JSON.parse(localStorage.getItem("st.anim") || "{}"); } catch { return {}; } })());
state.anim.play = true;
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

function animSubs(d, k) {
  const A = state.anim, list = animFrameList(d), N = list.length;
  k = (!k || k >= N) ? N : k;
  const key = `${k}${A.align ? "a" : "s"}${A.fix ? "f" : ""}${N}`;
  if (d.subs[key]) return d.subs[key];
  const S = d.meta.size, out = [], sh = (A.fix && d.meta.shift) || null;
  const skipped = d.meta.n - N;
  let start = 0;
  for (let g = 0; g < k; g++) {
    const cnt = Math.floor(N / k) + (g < N % k ? 1 : 0);
    const grp = list.slice(start, start + cnt);
    const m = new Float32Array(S * S); let px = 0, py = 0;
    for (const i of grp) { px += d.meta.track[i][0]; py += d.meta.track[i][1]; }
    px /= cnt; py /= cnt;
    for (const i of grp) {
      const f = d.frames[i];
      // local star shift of frame i (stars and object of that frame are displaced alike)
      let dx = sh ? sh[i][0] : 0, dy = sh ? sh[i][1] : 0;
      if (A.align && cnt > 1) {
        // stack on the object: its position in frame i lands on the group's mean position;
        // stars get slightly trailed but stay put
        dx += d.meta.track[i][0] - px; dy += d.meta.track[i][1] - py;
      }
      if (dx || dy) animShiftAdd(m, f, S, dx, dy);
      else for (let j = 0; j < m.length; j++) m[j] += f[j];
    }
    px *= cnt; py *= cnt;
    for (let j = 0; j < m.length; j++) m[j] /= cnt;
    const smp = []; for (let j = 0; j < m.length; j += 3) smp.push(m[j]);
    smp.sort((a, b) => a - b);
    const med = smp[smp.length >> 1];
    const dev = smp.map((v) => Math.abs(v - med)).sort((a, b) => a - b);
    out.push({ m, x: px / cnt, y: py / cnt, med, sig: 1.4826 * dev[dev.length >> 1] || 1, skipped });
    start += cnt;
  }
  // one noise level for all images of the animation (no flicker)
  const sig = out.map((o) => o.sig).sort((a, b) => a - b)[out.length >> 1];
  out.forEach((o) => o.sig = sig);
  d.subs[key] = out;
  return out;
}

function animDraw(cv, d) {
  const A = state.anim, subs = animSubs(d, A.k), idx = animTick % subs.length, g = subs[idx], S = d.meta.size;
  const lo = A.black, hi = Math.max(A.white, A.black + 0.1), inv = 1 / (hi - lo), ig = 1 / A.gamma;
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
  const gap = Math.max(2 * d.fwhm, 5) * sc * 0.6, arm = Math.max(2.5 * d.fwhm, 6) * sc * 0.6;
  if (A.cross) {
    ctx.strokeStyle = "rgb(255,196,40)"; ctx.lineWidth = 2; ctx.beginPath();
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      ctx.moveTo(cx + dx * gap, cy + dy * gap); ctx.lineTo(cx + dx * (gap + arm), cy + dy * (gap + arm));
    }
    ctx.stroke();
  }
  const cnt = cv.parentNode && cv.parentNode.querySelector(".cnt");
  if (cnt) cnt.textContent = `${idx + 1}/${subs.length}${g.skipped ? ` · bez ${g.skipped}` : ""}`;
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
    if (!state.anim.play || state.view !== "job" || state.tab !== "cands") return;
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
async function setStatus(job, cid, status) {
  const old = state.cands.find((x) => x.id === cid);
  if (!old || old.status === status) return;
  const body = { status };
  if (status === "unreviewed") body.reviewed = false;
  const c = await api(`/api/jobs/${job.id}/candidates/${cid}`, { method: "PATCH", body });
  state.history.push({ jobId: job.id, cid, prev: old.status, prevReviewed: !!old.reviewed });
  if (state.history.length > 200) state.history.shift();
  const i = state.cands.findIndex((x) => x.id === cid); state.cands[i] = c;
  toast(`#${cid} → ${STATUS_CZ[status] || status}   ·   Z = zpět`);
  renderCands(job);
}
async function undoStatus(job) {
  let k = -1;
  for (let j = state.history.length - 1; j >= 0; j--) if (state.history[j].jobId === job.id) { k = j; break; }
  if (k < 0) { toast("Není co vracet."); return; }
  const h = state.history.splice(k, 1)[0];
  const c = await api(`/api/jobs/${job.id}/candidates/${h.cid}`, { method: "PATCH", body: { status: h.prev, reviewed: h.prevReviewed } });
  const i = state.cands.findIndex((x) => x.id === h.cid); if (i >= 0) state.cands[i] = c;
  await renderCands(job);
  const idx = visibleCands().findIndex((x) => x.id === h.cid);
  if (idx >= 0) { state.focus = idx; markFocus(); }
  toast(`Vráceno: #${h.cid} je zase ${STATUS_CZ[h.prev] || h.prev}.`);
}
document.addEventListener("keydown", (e) => {
  if (state.view !== "job" || state.tab !== "cands" || e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
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
  else if (e.key === "ArrowRight") { e.preventDefault(); animStep(1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); animStep(-1); }
});

async function renderKnown(job) {
  const known = await api(`/api/jobs/${job.id}/known`);
  if (!known.length) { $("#tabbody").innerHTML = `<div class="empty">Žádné známé objekty v poli (nebo chybí WCS / MPCORB).</div>`; return; }
  $("#tabbody").innerHTML = `<table class="tbl"><tr><th>Objekt</th><th>Typ</th><th>Mag (předp.)</th><th>Rychlost ″/min</th><th>PA</th><th>RA</th><th>Dec</th><th>x, y</th><th>Detekován</th><th title="SNR změřené přímo na předpovězené stopě">Vynucené SNR</th></tr>
    ${known.map((k) => `<tr><td>${esc(k.name)}</td><td>${k.kind === "comet" ? "kometa" : "planetka"}</td><td>${k.mag != null ? k.mag.toFixed(1) : "–"}</td>
      <td>${k.rate_arcsec_min.toFixed(3)}</td><td>${k.pa_deg.toFixed(1)}</td><td class="mono">${raStr(k.ra)}</td><td class="mono">${decStr(k.dec)}</td>
      <td class="mono">${k.x?.toFixed(0)}, ${k.y?.toFixed(0)}</td><td>${k.matched_candidate != null ? "✓ #" + k.matched_candidate : "–"}</td>
      <td>${k.forced_snr != null ? k.forced_snr.toFixed(1) : "–"}${k.forced_note ? ` <span class="badge flag" title="Hodnota může být zkreslená">${esc(k.forced_note)}</span>` : ""}</td></tr>`).join("")}</table>
    <p class="sub">Vynucené SNR = signifikance změřená přímo na předpovězené stopě. Ukazuje, zda byl objekt v datech nad prahem (${esc(job.params?.snr_threshold ?? "výchozí")}), i když ho hledání nenašlo.</p>`;
}

function renderSummary(job) {
  const s = job.summary || {};
  const m = s.masters || {};
  const sel = (x) => x ? `<div class="sel ${x.path ? "ok" : "bad"}"><b>${x.kind}</b>: ${esc(x.path || "—")}<br><span class="sub">${esc(x.reason)}</span>${(x.warnings || []).map((w) => `<br><span class="w">${esc(w)}</span>`).join("")}</div>` : "";
  const rows = [["Snímků", s.n_frames], ["Expozice", s.exptime + " s"], ["Filtr", s.filter], ["Délka sekvence", s.span_min + " min"],
    ["Měřítko", s.pixel_scale + " ″/px"], ["FWHM", s.fwhm_px + " px"], ["Hypotéz rychlosti", s.grid_size], ["Backend", s.backend],
    ["Detekcí", s.detections], ["Po automatickém prověření", s.candidates], ["Známých v poli", s.known_in_field],
    ["Známých spárováno s kandidáty", s.known_detected],
    ["Známých s nuceným SNR nad prahem", s.known_forced_above_threshold ?? "–"],
    ["Kandidátů na neznámé", s.unknown_candidates], ["WCS", s.wcs ? "ano" : "ne"]];
  if (s.known_rejected) rows.push(["Známé objekty zamítnuté prověřením", s.known_rejected.length
    ? s.known_rejected.map((k) => `${k.name} (${k.flags.join(", ") || "–"})`).join("; ")
    : "žádné – pravidla nezahodila žádnou identifikovanou planetku"]);
  if (s.throughput) rows.push(["Propustnost (mapa)", `rozdíly v poli až ${Math.round(s.throughput.spread * 100)} %, nejhorší snímek ${Math.round(Math.min(...s.throughput.frame_min) * 100)} %`]);
  if (s.injection) rows.push(["Injekce", `${s.injection.detected}/${s.injection.n} nalezeno, ${s.injection.passed} prošlo prověřením · `
    + s.injection.bins.map((b) => `SNR ${b.snr_from}–${b.snr_to ?? "∞"}: ${b.passed}/${b.n}`).join(", ")]);
  $("#tabbody").innerHTML = `<div class="grid2"><div class="panel"><table class="tbl">${rows.map(([k, v]) => `<tr><td class="sub">${k}</td><td>${esc(v)}</td></tr>`).join("")}</table></div>
    <div class="panel"><h3 style="margin-top:0">Kalibrace</h3>${sel(m.dark)}${sel(m.flat)}
    <h3>Časy (s)</h3><table class="tbl">${Object.entries(s.timings_s || {}).map(([k, v]) => `<tr><td class="sub">${k}</td><td>${v}</td></tr>`).join("")}</table></div></div>`;
}

function renderExport(job) {
  $("#tabbody").innerHTML = `<div class="panel">
    <p>Export obsahuje jen <b>přijaté</b> kandidáty (MPC) nebo všechny (CSV).</p>
    <div class="warnbox">Astrometrie zatím vychází z WCS řešení ASTAP bez refitu proti Gaia DR3 – před odesláním na MPC zkontrolujte.</div>
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
      <label>Přístroj</label><select id="nj-inst">${state.instruments.map((i) => `<option value="${esc(i.id)}">${esc(i.name || i.id)} (${i.pixel_scale_arcsec}″/px)</option>`).join("")}</select>
      <label>Adresář se snímky</label><div class="row"><input id="nj-dir" spellcheck="false"><button class="btn" data-browse="nj-dir">Procházet…</button></div>
      <label>Název úlohy</label><input id="nj-name" placeholder="automaticky z adresáře">
      <label>Rychlost min / max (″/min)</label><div class="row"><input id="nj-rmin" type="number" step="0.05" value="${s.rate_min_arcsec_min}"><input id="nj-rmax" type="number" step="0.5" value="${s.rate_max_arcsec_min}"></div>
      <label>Práh SNR</label><input id="nj-snr" type="number" step="0.5" value="${s.snr_threshold}">
      <label>Max. odchylka segmentu (σ)</label><input id="nj-seg" type="number" step="0.5" value="${s.segment_z_max}">
      <label>Syntetické objekty (injekce)</label><div><input id="nj-inj" type="number" min="0" step="10" value="${s.inject_n ?? 0}"><div class="hint">Počet umělých planetek vložených do snímků (např. 50). Změří úplnost a dodá příklady pro klasifikátor; v seznamu kandidátů se neukazují.</div></div>
    </div>
    <div id="nj-fields"></div>
    <h3>Kalibrace, kterou program použije</h3><div id="nj-prev" class="sub">Vyberte adresář se snímky…</div>
    <div style="margin-top:14px"><button class="btn primary" id="nj-go">Spustit</button></div></div>`;
  bindBrowse();
  const prev = async () => {
    const dir = $("#nj-dir").value; if (!dir) return;
    try {
      const r = await api(`/api/instruments/${$("#nj-inst").value}/preview`, { method: "POST", body: { lights_dir: dir } });
      if (!r.frames) { $("#nj-fields").innerHTML = ""; $("#nj-prev").innerHTML = `<div class="sel bad">V adresáři nejsou FITS snímky.</div>`; return; }
      const m = r.meta;
      $("#nj-fields").innerHTML = r.dirs ? dirsHtml(r.dirs) : fieldsHtml(r.fields);
      $("#nj-prev").innerHTML = `<div class="sub">${r.dirs ? `celkem ${r.frames} snímků v ${r.dirs.length} adresářích; mastery níže pro první adresář (${esc(r.dirs[0].rel)}) – ostatní viz sloupec Mastery<br>` : ""}${r.dirs ? r.dirs[0].frames : r.frames} snímků · expozice ${m.exptime} s · filtr ${esc(m.filter)} (${esc(m.raw_filter)}) · ${m.width}×${m.height} · bin ${m.xbin}x${m.ybin} · T ${m.temperature ?? "?"} °C · gain ${m.gain ?? "?"}</div>`
        + Object.values(r.selection).map((x) => `<div class="sel ${x.path ? "ok" : "bad"}"><b>${x.kind}</b>: ${esc(x.path || "nenalezen")}<br><span class="sub">${esc(x.reason)}</span>${(x.warnings || []).map((w) => `<br><span class="w">${esc(w)}</span>`).join("")}</div>`).join("");
    } catch (e) { $("#nj-prev").innerHTML = `<div class="sel bad">${esc(e.message)}</div>`; }
  };
  $("#nj-dir").onchange = prev; $("#nj-inst").onchange = prev;
  $("#nj-go").onclick = async () => {
    try {
      const boxes = [...document.querySelectorAll("[data-field]")];
      const dboxes = [...document.querySelectorAll("[data-dir]")];
      const body = {
        instrument_id: $("#nj-inst").value, lights_dir: $("#nj-dir").value, name: $("#nj-name").value,
        params: { rate_min_arcsec_min: +$("#nj-rmin").value, rate_max_arcsec_min: +$("#nj-rmax").value,
          snr_threshold: +$("#nj-snr").value, segment_z_max: +$("#nj-seg").value, inject_n: +$("#nj-inj").value || 0 } };
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

function fieldsHtml(fs) {
  if (!fs || !fs.fields || fs.fields.length < 2 && !(fs.unassigned || []).length) return fs && fs.notes && fs.notes.length ? `<div class="sub">${fs.notes.map(esc).join("<br>")}</div>` : "";
  const tol = fs.tolerance_deg != null ? (fs.tolerance_deg * 60).toFixed(1) + "′" : "?";
  return `<h3>Zorná pole v adresáři: ${fs.fields.length}</h3>
    <div class="sub">Každé pole se zpracuje jako samostatná úloha, postupně jedno po druhém. Snímky s rozdílem pointace pod ${tol} patří k jednomu poli.</div>
    <table class="tbl"><tr><th></th><th>Pole</th><th>RA</th><th>Dec</th><th>Snímků</th><th>Dithering</th><th>Čas</th></tr>
    ${fs.fields.map((f) => `<tr><td><input type="checkbox" data-field="${f.index}" ${f.n >= 5 ? "checked" : "disabled title='méně než 5 snímků'"}></td>
      <td>${esc(f.label)}</td><td class="mono">${f.ra_deg == null ? "–" : raStr(f.ra_deg)}</td><td class="mono">${f.dec_deg == null ? "–" : decStr(f.dec_deg)}</td>
      <td>${f.n}${f.n < 5 ? ' <span class="w">(málo)</span>' : ""}</td><td>${f.spread_arcmin.toFixed(1)}′ · ${Math.round(f.spread_px)} px</td>
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
    ${dirs.map((d) => `<tr><td><input type="checkbox" data-dir="${esc(d.path)}" ${d.frames >= 5 ? "checked" : "disabled title='méně než 5 snímků'"}></td>
      <td class="mono">${esc(d.rel)}</td><td>${d.frames}${d.frames < 5 ? ' <span class="w">(málo)</span>' : ""}</td>
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
    <p class="sub">Program sleduje kořenový adresář i všechny podadresáře. Jakmile adresář obsahuje nastavený počet snímků
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
    ${Object.keys(w.pending || {}).length ? `<table class="tbl"><tr><th>Adresář</th><th>Snímků</th><th>Poslední změna</th></tr>${Object.entries(w.pending).map(([d, p]) => `<tr><td class="mono">${esc(d)}</td><td>${p.count} / ${c.frames_per_set}</td><td>před ${p.age_s} s</td></tr>`).join("")}</table>` : `<div class="sub">Žádné rozpracované sady.</div>`}
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
    <label>GPU backend</label><select id="s-gpu"><option value="auto">automaticky (CUDA, jinak CPU)</option><option value="cupy">CUDA (CuPy)</option><option value="numpy">jen CPU</option></select>
    <label>Zorná pole</label><label class="switch"><input type="checkbox" id="s-fsplit" ${s.field_split ? "checked" : ""}> rozdělit adresář s více poli na samostatné úlohy (podle souřadnic v hlavičce)</label>
    <label>Tolerance pole</label><input id="s-ftol" type="number" step="0.05" min="0.05" max="0.6" value="${s.field_tolerance_fov}">
    <div class="hint">Snímky patří k jednomu poli, pokud se jejich pointace liší méně než tento zlomek menší strany zorného pole (0,25 = čtvrtina). Dithering musí být menší, sousední pole mozaiky vzdálenější.</div>
    <label>Limit paměti GPU</label><input id="s-gmem" type="number" step="0.05" min="0.1" max="1" value="${s.gpu.memory_fraction}">
    <label>Online: stahování MPC</label><label class="switch"><input type="checkbox" id="s-on-mpc" ${s.online.mpc_download ? "checked" : ""}> povolit tlačítko aktualizace</label>
    <label>Online: SkyBoT</label><label class="switch"><input type="checkbox" id="s-on-sky" ${s.online.skybot ? "checked" : ""}> křížová kontrola identifikací (připravuje se)</label>
    <label>Online: náhledy DSS</label><label class="switch"><input type="checkbox" id="s-on-dss" ${s.online.sky_images ? "checked" : ""}> výřez z přehlídky DSS/2MASS u kandidáta na vyžádání (CDS, záložně STScI a NASA SkyView)</label>
    <label>Online: astrometry.net</label><label class="switch"><input type="checkbox" id="s-on-ast" ${s.online.astrometry_net ? "checked" : ""}> záložní řešení pole (připravuje se)</label>
  </div><div style="margin-top:14px"><button class="btn primary" id="s-save">Uložit nastavení</button></div></div>`;
  $("#s-gpu").value = s.gpu.backend;
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
      mpcorb_max_age_days: +$("#s-age").value, ephemeris: $("#s-eph").value || "auto",
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
let skyState = { survey: "DSS2c" };
async function openSky(job, c) {
  if (!c) return;
  document.querySelector(".sky-modal")?.remove();
  const scale = +(job.summary?.pixel_scale || 0);
  const frames = c.images?.frames;
  const size = frames?.size || 64;
  const W = c.sky_matrix || localSkyMatrix(state.cands, c);
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
  const aladin = c.ra != null ? `https://aladin.cds.unistra.fr/AladinLite/?target=${encodeURIComponent(c.ra.toFixed(5) + " " + c.dec.toFixed(5))}&fov=${Math.max(fovDeg, 0.05).toFixed(3)}&survey=P%2FDSS2%2Fcolor` : null;
  wrap.innerHTML = `<div class="modal-box sky-box"><div class="modal-head"><b>#${c.id} · porovnání s přehlídkou</b>
      <span class="sub">${c.ra != null ? raStr(c.ra) + " " + decStr(c.dec) : ""} · pole ${(fovDeg * 60).toFixed(1)}′ · ${W ? "WCS: sever nahoře, východ vlevo" : "orientace přibližná"}</span>
      <button class="btn small" data-close>✕</button></div>
    <div class="sky-panels">
      <figure><div class="sky-img"><canvas id="sky-ours" width="360" height="360"></canvas>${SKYMARK}</div><figcaption>naše snímky – průměr všech (hvězdy ostré, objekt jako krátká stopa)</figcaption></figure>
      <figure><div class="sky-img" id="sky-ref"><div class="empty">načítám…</div></div><figcaption id="sky-src">přehlídka</figcaption></figure>
    </div>
    <div class="modal-foot"><div class="seg" id="sky-surv">${surveys.map(([k, l]) => `<button data-s="${k}" class="${skyState.survey === k ? "active" : ""}">${l}</button>`).join("")}</div>
      ${aladin ? `<a href="${aladin}" target="_blank" rel="noopener" class="btn small">Aladin Lite ↗</a>` : ""}</div></div>`;
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
      box.innerHTML = `<img src="${URL.createObjectURL(blob)}">${SKYMARK}`;
      src.textContent = `${surveys.find((s) => s[0] === skyState.survey)[1]} · zdroj: ${r.headers.get("X-Sky-Source") || "?"}`;
    } catch (e) { box.innerHTML = `<div class="empty">${esc(e.message)}</div>`; src.textContent = "přehlídka"; }
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
