"use strict";
// LocalClip editor. No frameworks, no network calls except to this server.
// All user and transcript text goes through textContent, never innerHTML.

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function h(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k in e) e[k] = v;
    else e.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return e;
}

function fmt(t) {
  if (!isFinite(t)) return "0:00.0";
  const m = Math.floor(t / 60);
  return m + ":" + (t - m * 60).toFixed(1).padStart(4, "0");
}

function say(msg, err) {
  const s = $("status");
  s.textContent = msg || "";
  s.className = err ? "err" : "";
}

async function api(path, opt = {}) {
  const r = await fetch(path, opt);
  let d = null;
  try { d = await r.json(); } catch (e) { /* non-JSON body */ }
  if (!r.ok) throw new Error((d && d.error) || "HTTP " + r.status);
  return d;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

let cur = null; // {id, meta, words, wordEls, clips, active, stopAt}
let uidN = 0;

// ---- projects list --------------------------------------------------------
async function loadProjects() {
  const { projects } = await api("/api/projects");
  const ul = $("projects");
  ul.replaceChildren(...projects.map((p) => h("li", {
    class: cur && cur.id === p.id ? "on" : "", onclick: () => openProject(p.id),
  }, h("span", { class: "nm", text: p.name, title: p.name }),
  h("span", { class: "st", text: p.status === "ready" ? p.clips + " clips" + (p.exports ? ", " + p.exports + " exp" : "") : p.status }))));
  if (!projects.length) ul.replaceChildren(h("li", { class: "st", text: "none yet" }));
}

// ---- upload ---------------------------------------------------------------
function analyzeOptions() {
  return {
    count: +$("o-count").value, min_dur: +$("o-min").value, max_dur: +$("o-max").value,
    transcribe_engine: $("o-engine").value, use_llm: $("o-llm").checked, use_faces: $("o-faces").checked,
  };
}

function upload(file) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", "/api/upload?name=" + encodeURIComponent(file.name));
    x.setRequestHeader("Content-Type", "application/octet-stream");
    const bar = $("uprog");
    bar.hidden = false;
    bar.value = 0;
    x.upload.onprogress = (e) => { if (e.lengthComputable) bar.value = (100 * e.loaded) / e.total; };
    x.onload = () => {
      bar.hidden = true;
      let d = {};
      try { d = JSON.parse(x.responseText); } catch (e) { /* ignore */ }
      x.status === 201 ? resolve(d.id) : reject(new Error(d.error || "upload failed: HTTP " + x.status));
    };
    x.onerror = () => { bar.hidden = true; reject(new Error("upload failed: connection error")); };
    x.send(file);
  });
}

async function newFromFile(file) {
  if (!file) return;
  try {
    say("Uploading " + file.name + " ...");
    const id = await upload(file);
    say("");
    await loadProjects();
    await startAnalyze(id);
  } catch (e) {
    say(e.message, true);
  }
}

async function startAnalyze(id) {
  try {
    const { job } = await post("/api/projects/" + id + "/analyze", analyzeOptions());
    await openProject(id, job);
  } catch (e) {
    say(e.message, true);
  }
}

// ---- jobs -----------------------------------------------------------------
async function followJob(jid, onLog) {
  for (;;) {
    const j = await api("/api/jobs/" + jid);
    onLog(j.log.join("\n"));
    if (j.status === "done") return j;
    if (j.status === "error") throw new Error(j.error || "job failed");
    await sleep(1000);
  }
}

// ---- project view ---------------------------------------------------------
function show(which) {
  for (const id of ["empty", "busy", "editor"]) $(id).hidden = id !== which;
}

function busyView(title, log, retryId) {
  show("busy");
  $("busy-title").textContent = title;
  $("busy-log").textContent = log || "";
  const old = $("busy").querySelector("button");
  if (old) old.remove();
  if (retryId) $("busy").append(h("button", { class: "btn primary", text: "Analyze", onclick: () => startAnalyze(retryId) }));
}

async function openProject(id, jobId) {
  say("");
  const meta = await api("/api/projects/" + id);
  cur = { id, meta, words: [], wordEls: [], clips: [], active: 0, stopAt: null };
  loadProjects();
  const job = jobId || (meta.job && meta.job.id);
  if (job) {
    busyView("Analyzing " + meta.name, "queued ...");
    try {
      await followJob(job, (t) => { if (cur && cur.id === id) $("busy-log").textContent = t; });
    } catch (e) {
      if (cur && cur.id === id) busyView("Analyze failed", e.message, id);
      loadProjects();
      return;
    }
    if (!cur || cur.id !== id) return;
    return openProject(id);
  }
  if (meta.status !== "ready") {
    return busyView(meta.status === "error" ? "Analyze failed" : meta.name, meta.error || "Not analyzed yet.", id);
  }
  cur.words = (await api("/api/projects/" + id + "/transcript")).words;
  cur.clips = meta.clips.map((c) => ({ ...c, uid: ++uidN, include: true }));
  show("editor");
  const vid = $("vid");
  vid.src = "/api/projects/" + id + "/video";
  $("picknote").textContent = "picked by " + meta.selection.engine + " (" + meta.selection.note + ")";
  const o = meta.options || {};
  if (o.aspect) $("x-aspect").value = o.aspect.join(":");
  $("x-upper").checked = o.upper !== false;
  renderWords();
  renderCands();
  renderClips();
  renderExports();
  drawTimeline();
}

// ---- transcript -----------------------------------------------------------
function renderWords() {
  const box = $("words");
  cur.wordEls = cur.words.map((w, i) => h("span", { text: w.text, "data-i": i, title: fmt(w.start) }));
  box.replaceChildren(...cur.wordEls.flatMap((e) => [e, " "]));
  box.onclick = (ev) => {
    const el = ev.target.closest("span[data-i]");
    if (!el) return;
    const w = cur.words[+el.dataset.i];
    const c = cur.clips[cur.active];
    if (!c) return;
    if (ev.shiftKey) setField(cur.active, "end", w.end + 0.1);
    else setField(cur.active, "start", Math.max(0, w.start - 0.05));
    $("vid").currentTime = c.start;
  };
  markWords();
}

function markWords() {
  const c = cur.clips[cur.active];
  cur.wordEls.forEach((e, i) => {
    const w = cur.words[i];
    e.classList.toggle("in", !!c && w.end > c.start && w.start < c.end);
  });
}

function bsearch(words, t) {
  let lo = 0, hi = words.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (words[mid].start <= t) { ans = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return ans;
}

// ---- candidates -----------------------------------------------------------
function renderCands() {
  $("cands").replaceChildren(...cur.meta.candidates.map((c) => h("li", {},
    h("span", { class: "t", text: fmt(c.start) + "-" + fmt(c.end) + " s" + c.score }),
    h("span", { class: "tx", text: c.text, title: c.text }),
    h("button", { class: "btn small", text: "Add", onclick: () => addClip({
      start: c.start, end: c.end, title: c.text.split(/\s+/).slice(0, 8).join(" "),
      selected_by: "candidate", why: "candidate: " + c.reasons.join("; "), score: c.score,
    }) }),
  )));
  if (!cur.meta.candidates.length) $("cands").replaceChildren(h("li", { class: "t", text: "none" }));
}

// ---- clips ----------------------------------------------------------------
function addClip(base) {
  const dur = cur.meta.duration;
  const c = {
    uid: ++uidN, id: null, include: true, captions: true, crop: "auto", crop_x: 0.5, snap: true,
    selected_by: "manual", why: "", score: null, title: "", ...base,
  };
  c.end = Math.min(dur, c.end);
  cur.clips.push(c);
  cur.active = cur.clips.length - 1;
  renderClips();
  drawTimeline();
  markWords();
}

function setField(i, key, val) {
  const c = cur.clips[i];
  c[key] = Math.round(Math.max(0, Math.min(cur.meta.duration, val)) * 100) / 100;
  const card = $("clips").children[i];
  if (card) {
    card.querySelector(`[data-f=${key}]`).value = c[key];
    refreshCard(i);
  }
  drawTimeline();
  markWords();
}

function problem(c) {
  const d = c.end - c.start;
  if (!(d > 0)) return "end must be after start";
  if (d < 1) return "shorter than 1 s";
  if (d > 180) return "longer than 180 s";
  return "";
}

function refreshCard(i) {
  const c = cur.clips[i];
  const card = $("clips").children[i];
  if (!card) return;
  const d = c.end - c.start;
  card.querySelector(".dur").textContent = (isFinite(d) ? d.toFixed(1) : "?") + " s";
  const p = card.querySelector(".prob");
  p.textContent = problem(c);
  card.classList.toggle("active", i === cur.active);
}

function numField(i, key, label) {
  const c = cur.clips[i];
  const inp = h("input", {
    type: "number", step: "0.1", min: "0", value: c[key], "data-f": key,
    onchange: () => { setField(i, key, parseFloat(inp.value) || 0); },
  });
  const nudge = (d) => h("button", { class: "btn small", text: (d > 0 ? "+" : "") + d, onclick: () => setField(i, key, c[key] + d) });
  return h("div", {}, h("label", {}, label, inp),
    h("div", {}, nudge(-0.5), " ", nudge(-0.1), " ", nudge(0.1), " ", nudge(0.5), " ",
      h("button", { class: "btn small", text: "playhead", title: "set from playhead", onclick: () => setField(i, key, $("vid").currentTime) })));
}

function renderClips() {
  $("clips").replaceChildren(...cur.clips.map((c, i) => {
    const slider = h("input", {
      type: "range", min: "0", max: "1", step: "0.01", value: c.crop_x ?? 0.5,
      oninput: () => { c.crop_x = +slider.value; drawGuide(); },
    });
    const cropLbl = h("label", {}, "Crop center", slider);
    const sel = h("select", {
      onchange: () => { c.crop = sel.value; cropLbl.hidden = c.crop !== "manual"; if (c.crop === "manual" && c.crop_x == null) c.crop_x = 0.5; drawGuide(); },
    }, ["auto", "face", "center", "manual"].map((m) => h("option", { value: m, text: m, selected: m === c.crop })));
    cropLbl.hidden = sel.value !== "manual";
    const card = h("div", { class: "clip" + (i === cur.active ? " active" : ""), onclick: () => activate(i) },
      h("div", { class: "row" },
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: c.include, onchange: (e) => { c.include = e.target.checked; drawTimeline(); updateExportNote(); } }), "Export"),
        h("b", { text: "#" + (i + 1) }),
        h("div", { class: "title" }, h("label", {}, "Title", h("input", { type: "text", value: c.title, maxLength: 120, oninput: (e) => { c.title = e.target.value; } }))),
        h("button", { class: "btn small", text: "Remove", onclick: (e) => { e.stopPropagation(); cur.clips.splice(i, 1); cur.active = Math.max(0, Math.min(cur.active, cur.clips.length - 1)); renderClips(); drawTimeline(); markWords(); } }),
      ),
      h("div", { class: "row" },
        numField(i, "start", "Start (s)"), numField(i, "end", "End (s)"),
        h("div", {}, h("label", {}, "Length"), h("span", { class: "dur" }), " ", h("span", { class: "prob bad" })),
        h("button", { class: "btn", text: "Play clip", onclick: (e) => { e.stopPropagation(); playClip(i); } }),
      ),
      h("div", { class: "row" },
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: c.captions, onchange: (e) => { c.captions = e.target.checked; } }), "Burn captions"),
        h("label", { class: "check", title: "move a cut that lands inside a word to just outside it" }, h("input", { type: "checkbox", checked: c.snap, onchange: (e) => { c.snap = e.target.checked; } }), "Snap cuts to words"),
        h("label", {}, "Framing", sel), cropLbl,
      ),
      h("div", { class: "row" }, h("span", { class: "meta", text: [c.selected_by, c.score != null ? "score " + c.score : "", c.why].filter(Boolean).join(" | ") })),
    );
    return card;
  }));
  cur.clips.forEach((_, i) => refreshCard(i));
  updateExportNote();
  drawGuide();
}

function activate(i) {
  if (cur.active === i) return;
  cur.active = i;
  [...$("clips").children].forEach((_, k) => refreshCard(k));
  drawTimeline();
  markWords();
  drawGuide();
}

function playClip(i) {
  const c = cur.clips[i];
  activate(i);
  const v = $("vid");
  v.currentTime = c.start;
  cur.stopAt = c.end;
  v.play();
}

function updateExportNote() {
  const n = cur.clips.filter((c) => c.include).length;
  $("xnote").textContent = n + " clip" + (n === 1 ? "" : "s") + " selected";
  $("b-export").disabled = n === 0;
}

// ---- timeline, guide, playhead -------------------------------------------
function drawTimeline() {
  const tl = $("timeline");
  const dur = cur.meta.duration || 1;
  const kids = cur.clips.map((c, i) => h("div", {
    class: "seg" + (c.include ? "" : " off") + (i === cur.active ? " active" : ""),
    style: `left:${(100 * c.start) / dur}%;width:${Math.max(0.4, (100 * (c.end - c.start)) / dur)}%`,
    title: "#" + (i + 1) + " " + c.title,
  }));
  kids.push(h("div", { class: "head", id: "head", style: `left:${(100 * $("vid").currentTime) / dur}%` }));
  tl.replaceChildren(...kids);
}

function drawGuide() {
  const g = $("guide");
  const c = cur && cur.clips[cur.active];
  if (!c || c.crop !== "manual" || !cur.meta.size) { g.hidden = true; return; }
  const [aw, ah] = $("x-aspect").value.split(":").map(Number);
  const [w, hh] = cur.meta.size;
  const frac = Math.min(1, (hh * aw) / ah / w);
  if (frac >= 0.999) { g.hidden = true; return; }
  const left = Math.min(1 - frac, Math.max(0, (c.crop_x ?? 0.5) - frac / 2));
  g.style.left = 100 * left + "%";
  g.style.width = 100 * frac + "%";
  g.hidden = false;
}

function onTime() {
  const v = $("vid");
  const t = v.currentTime;
  $("now").textContent = fmt(t);
  const head = $("head");
  if (head) head.style.left = (100 * t) / (cur.meta.duration || 1) + "%";
  if (cur.stopAt != null && t >= cur.stopAt) { v.pause(); cur.stopAt = null; }
  const i = bsearch(cur.words, t);
  if (cur.lastWord !== i) {
    if (cur.lastWord != null && cur.wordEls[cur.lastWord]) cur.wordEls[cur.lastWord].classList.remove("cur");
    if (i >= 0 && cur.words[i].end >= t - 0.3) cur.wordEls[i].classList.add("cur");
    cur.lastWord = i;
  }
}

// ---- export ---------------------------------------------------------------
async function doExport() {
  const chosen = cur.clips.filter((c) => c.include);
  const bad = chosen.map((c, i) => problem(c) && "clip " + (cur.clips.indexOf(c) + 1) + ": " + problem(c)).filter(Boolean);
  if (bad.length) { say(bad[0], true); return; }
  const id = cur.id;
  const log = $("export-log");
  log.hidden = false;
  log.textContent = "queued ...";
  $("b-export").disabled = true;
  try {
    const { job } = await post("/api/projects/" + id + "/export", {
      aspect: $("x-aspect").value, upper: $("x-upper").checked,
      clips: chosen.map((c) => ({
        id: c.id, start: c.start, end: c.end, title: c.title, captions: c.captions,
        crop: c.crop === "manual" ? "auto" : c.crop, crop_x: c.crop === "manual" ? c.crop_x : null,
        snap: c.snap, selected_by: c.selected_by, why: c.why,
      })),
    });
    say("Exporting ...");
    await followJob(job, (t) => { log.textContent = t; });
    say("Export done.");
    if (cur && cur.id === id) {
      cur.meta = await api("/api/projects/" + id);
      renderExports();
      loadProjects();
    }
  } catch (e) {
    say(e.message, true);
  } finally {
    if (cur) updateExportNote();
  }
}

function renderExports() {
  const box = $("exports");
  const ex = [...cur.meta.exports].reverse();
  if (!ex.length) { box.replaceChildren(h("div", { class: "meta", text: "Nothing exported yet." })); return; }
  box.replaceChildren(...ex.map((e) => {
    const base = "/api/projects/" + cur.id + "/exports/" + e.id + "/";
    return h("div", { class: "export" },
      h("h3", { text: e.created + "  |  " + e.aspect + "  |  " + e.clips.length + " clips  |  " + e.seconds + " s" }),
      h("div", { class: "vids" }, e.clips.map((c) => h("figure", {},
        h("video", { src: base + c.name, controls: true, preload: "none" }),
        h("figcaption", {}, c.title || c.name, h("br"), c.duration.toFixed(1) + " s, " + c.crop + (c.captions ? ", captions" : ""), h("br"),
          h("a", { href: base + c.name + "?dl=1", text: "download " + c.name }))))),
      e.sheet ? h("img", { class: "sheet", src: base + e.sheet, alt: "contact sheet", loading: "lazy" }) : null);
  }));
}

// ---- wiring ---------------------------------------------------------------
$("file").addEventListener("change", (e) => { newFromFile(e.target.files[0]); e.target.value = ""; });
const drop = $("drop");
for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
drop.addEventListener("drop", (e) => newFromFile(e.dataTransfer.files[0]));
$("b-export").addEventListener("click", doExport);
$("x-aspect").addEventListener("change", drawGuide);
$("vid").addEventListener("timeupdate", () => { if (cur && cur.meta.duration) onTime(); });
$("vid").addEventListener("pause", () => { if (cur) cur.stopAt = null; });
$("timeline").addEventListener("click", (e) => {
  const r = $("timeline").getBoundingClientRect();
  $("vid").currentTime = ((e.clientX - r.left) / r.width) * cur.meta.duration;
});
$("b-add").addEventListener("click", () => {
  const t = $("vid").currentTime;
  addClip({ start: Math.round(t * 100) / 100, end: Math.min(cur.meta.duration, Math.round((t + 30) * 100) / 100), title: "" });
});
$("b-start").addEventListener("click", () => cur.clips.length && setField(cur.active, "start", $("vid").currentTime));
$("b-end").addEventListener("click", () => cur.clips.length && setField(cur.active, "end", $("vid").currentTime));

loadProjects().catch((e) => say(e.message, true));
