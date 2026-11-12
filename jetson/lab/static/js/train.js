// Train tab: setup wizard + live training dashboard + past runs.
import { api, toast, state, $, $$, esc, emoji, drawBoxes, fmtTime } from "./util.js";

const LRS = [0.0005, 0.001, 0.003, 0.01, 0.02, 0.05, 0.1];
const SIZES = [320, 416, 480, 512, 640];
const MAX_POINTS = 600;   // keep charts light on the Jetson

let classes = [], chosen = new Set(), model = "yolov8n";
let charts = {}, view = null, pollTimer = null;

// ── setup ───────────────────────────────────────────────────────
export async function init() {
  renderModels();
  bindSlider("tr-per", v => v);
  bindSlider("tr-epochs", v => v);
  bindSlider("tr-batch", v => v);
  bindSlider("tr-lr", v => LRS[v]);
  bindSlider("tr-imgsz", v => SIZES[v] + " px");
  if (state.info.jetson) $("#tr-batch").max = 16;
  $("#tr-opt").onchange = () => {            // AdamW wants a smaller step than SGD
    $("#tr-lr").value = $("#tr-opt").value === "AdamW" ? 1 : 3;
    $("#tr-lr").dispatchEvent(new Event("input"));
  };
  ["tr-per", "tr-lib", "tr-mine"].forEach(id => $("#" + id).addEventListener("input", picHint));
  // locking random "eyes" would make no sense, so the two options exclude each other
  $("#tr-freeze").onchange = () => { if ($("#tr-freeze").checked) $("#tr-scratch").checked = false; };
  $("#tr-scratch").onchange = () => { if ($("#tr-scratch").checked) $("#tr-freeze").checked = false; };
  $("#tr-start").onclick = start;
  $("#tr-stop").onclick = stop;
  $("#pv-epoch").oninput = () => renderPreview(+$("#pv-epoch").value);
  $("#live-card details").addEventListener("toggle", e => { if (e.target.open) loadLog(); });
  makeCharts();
  selectModel("yolov8n");
}

export async function onShow() {
  await loadClasses();
  await loadRuns();
  const cur = await api("/api/train/current");
  if (cur.running && (!view || view.id !== cur.run)) attach(cur.run);
}

function bindSlider(id, fmt) {
  const el = $("#" + id), out = $("#" + id + "-v");
  const upd = () => (out.textContent = fmt(+el.value));
  el.addEventListener("input", upd);
  upd();
}

function renderModels() {
  $("#tr-models").innerHTML = state.info.models.map(m => `
    <div class="model-card" data-id="${m.id}"><div class="emo">${m.emoji}</div>
      <div><b>${esc(m.name)}</b> <span class="meta">· ${m.params} numbers · ${m.speed}</span>
      <div class="meta">${esc(m.about)}</div></div></div>`).join("");
  $$("#tr-models .model-card").forEach(el => el.onclick = () => selectModel(el.dataset.id));
}

function selectModel(id) {
  model = id;
  $$("#tr-models .model-card").forEach(el => el.classList.toggle("active", el.dataset.id === id));
  const m = state.info.models.find(x => x.id === id);
  $("#tr-batch").value = m.batch[state.info.jetson ? "jetson" : "pc"];
  $("#tr-batch").dispatchEvent(new Event("input"));
}

async function loadClasses() {
  classes = await api("/api/classes");
  $("#tr-classes").innerHTML = classes.map(c => {
    const n = c.library + c.mine;
    return `<button class="chip ${chosen.has(c.label) ? "active" : ""}" data-l="${esc(c.label)}" ${n < 2 ? "disabled" : ""}
      title="${c.library} library + ${c.mine} of yours${c.mine_unchecked ? ` (${c.mine_unchecked} still need boxes)` : ""}">
      ${emoji(c.label)} ${esc(c.label)} <small>${n}</small></button>`;
  }).join("") || `<p class="muted">No pictures yet. Add some in the Pictures tab.</p>`;
  $$("#tr-classes .chip").forEach(b => b.onclick = () => {
    const l = b.dataset.l;
    chosen.has(l) ? chosen.delete(l) : chosen.add(l);
    b.classList.toggle("active", chosen.has(l));
    picHint();
  });
  picHint();
}

function picHint() {
  const per = +$("#tr-per").value, lib = $("#tr-lib").checked, mine = $("#tr-mine").checked;
  const parts = [...chosen].map(l => {
    const c = classes.find(x => x.label === l) || { library: 0, mine: 0 };
    const avail = (lib ? c.library : 0) + (mine ? c.mine : 0);
    return `${emoji(l)} ${Math.min(per, avail)}${avail < per ? ` (only ${avail} available)` : ""}`;
  });
  $("#tr-pic-hint").textContent = parts.length ? "Will use: " + parts.join(" · ") : "";
}

async function start() {
  $("#tr-err").textContent = "";
  if (!chosen.size) { $("#tr-err").textContent = "Pick at least one object in step 1."; return; }
  const body = {
    name: $("#tr-name").value.trim() || [...chosen].join(" & ") + " finder",
    classes: [...chosen],
    per_class: +$("#tr-per").value,
    use_library: $("#tr-lib").checked,
    use_mine: $("#tr-mine").checked,
    model,
    epochs: +$("#tr-epochs").value,
    batch: +$("#tr-batch").value,
    imgsz: SIZES[+$("#tr-imgsz").value],
    lr: LRS[+$("#tr-lr").value],
    optimizer: $("#tr-opt").value,
    augment: $("#tr-aug").checked,
    pretrained: !$("#tr-scratch").checked,
    freeze: $("#tr-freeze").checked,
  };
  $("#tr-start").disabled = true;
  try {
    const r = await api("/api/train/start", { json: body });
    toast(`Training started with ${r.dataset.train} practice + ${r.dataset.val} quiz pictures`);
    await loadRuns();
    attach(r.run);
    $("#live-card").scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    $("#tr-err").textContent = e.message;
  }
  $("#tr-start").disabled = false;
}

async function stop() {
  if (!view) return;
  const force = $("#tr-stop").dataset.force === "1";
  if (!force && !confirm("Stop training? The AI keeps what it has learned so far.")) return;
  await api("/api/train/stop", { json: { force } });
  $("#live-status").textContent = force ? "Stopping now…" : "⏹ Finishing this batch, then stopping… (press again to stop immediately)";
  $("#tr-stop").dataset.force = "1";
}

// ── charts ──────────────────────────────────────────────────────
function makeCharts() {
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.animation = false;            // cheap redraws on the Jetson
  Chart.defaults.maintainAspectRatio = false;
  const lin = (title, extra = {}) => ({ type: "linear", title: { display: !!title, text: title }, ...extra });
  const opts = (xTitle, yExtra = {}) => ({
    responsive: true, parsing: false, normalized: true,
    interaction: { mode: "nearest", intersect: false },
    plugins: { legend: { labels: { boxWidth: 12 } } },
    scales: { x: lin(xTitle, { min: 0 }), y: lin("", yExtra) },
  });
  charts.loss = new Chart($("#ch-loss"), {
    type: "line",
    data: { datasets: [
      { label: "Practice (live)", data: [], borderColor: "#5b5bf0", pointRadius: 0, borderWidth: 1.5 },
      { label: "Quiz (each epoch)", data: [], borderColor: "#e5484d", pointRadius: 3, borderWidth: 2 },
    ] },
    options: opts("epoch", { min: 0 }),
  });
  charts.score = new Chart($("#ch-score"), {
    type: "line",
    data: { datasets: [
      { label: "mAP50 (box score)", data: [], borderColor: "#17a673", borderWidth: 3, pointRadius: 2 },
      { label: "Precision", data: [], borderColor: "#5b5bf0", borderWidth: 1.5, pointRadius: 0 },
      { label: "Recall", data: [], borderColor: "#f0a500", borderWidth: 1.5, pointRadius: 0 },
    ] },
    options: opts("epoch", { min: 0, max: 1 }),
  });
  charts.change = new Chart($("#ch-change"), {
    type: "bar",
    data: { labels: [], datasets: [
      { label: "👁️ Eyes (backbone)", data: [], backgroundColor: "#5b5bf0" },
      { label: "🔀 Mixer (neck)", data: [], backgroundColor: "#17a673" },
      { label: "🎯 Decider (head)", data: [], backgroundColor: "#f0a500" },
    ] },
    options: { responsive: true, plugins: { legend: { labels: { boxWidth: 12 } },
      tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(3)}% changed` } } },
      scales: { x: { title: { display: true, text: "epoch" } }, y: { title: { display: true, text: "% changed" }, min: 0 } } },
  });
  charts.lr = new Chart($("#ch-lr"), {
    type: "line",
    data: { datasets: [{ label: "Learning rate", data: [], borderColor: "#9b59b6", borderWidth: 2, pointRadius: 2 }] },
    options: opts("epoch", { min: 0 }),
  });
}

function resetCharts() {
  for (const c of Object.values(charts)) {
    c.data.datasets.forEach(d => (d.data = []));
    if (c.data.labels) c.data.labels = [];
    c.update("none");
  }
}

function thin(arr) {
  // drop every other point once a series gets long
  if (arr.length > MAX_POINTS) return arr.filter((_, i) => i % 2 === 0 || i === arr.length - 1);
  return arr;
}

const total = l => l ? ["box", "cls", "dfl"].reduce((s, k) => s + (l[k] || 0), 0) : null;

// ── live view ───────────────────────────────────────────────────
async function attach(runId) {
  clearTimeout(pollTimer);
  view = { id: runId, next: 0, epochs: 0, nb: 0, previews: {}, first: null, best: 0, last: null, running: true, run: null };
  resetCharts();
  $("#previews").innerHTML = "";
  $("#train-log").textContent = "";
  $("#tr-stop").dataset.force = "0";
  $("#pv-epoch").disabled = true;
  try { view.run = await api(`/api/runs/${runId}`); } catch { /* not ready yet */ }
  $("#live-title").textContent = `📊 ${view.run?.name || "Training"}`;
  $("#live-status").textContent = "Getting ready…";
  poll();
}

async function poll() {
  if (!view) return;
  const v = view;
  try {
    const d = await api(`/api/runs/${v.id}/events?since=${v.next}`);
    if (v !== view) return;
    v.next = d.next;
    for (const e of d.events) handle(e);
    if (d.events.length) for (const c of Object.values(charts)) c.update("none");
    v.running = d.running;
    $("#tr-stop").disabled = !d.running;
    if ($("#live-card details").open) loadLog();
    if (d.running || d.events.length) pollTimer = setTimeout(poll, 1000);
    else finished();
  } catch (e) {
    pollTimer = setTimeout(poll, 3000);
  }
}

function handle(e) {
  const v = view;
  switch (e.type) {
    case "status":
      $("#live-status").textContent = e.msg;
      break;
    case "start":
      // a second "start" means training restarted (e.g. smaller batch after running out of memory)
      resetCharts();
      v.first = null; v.best = 0;
      $("#live-stats").innerHTML = "";
      v.epochs = e.epochs; v.nb = e.nb;
      setTimeout(loadRuns, 1000);         // its state just changed from "starting" to "running"
      charts.loss.options.scales.x.max = e.epochs;
      charts.score.options.scales.x.max = e.epochs;
      charts.lr.options.scales.x.max = e.epochs;
      $("#live-status").textContent = `🚀 Learning to find ${e.classes.map(c => emoji(c) + " " + c).join(", ")}…`;
      stats({ "🧠 Brain size": (e.params / 1e6).toFixed(1) + " M numbers", "⚙️ Running on": e.device.startsWith("cuda") ? "GPU" : "CPU",
              "📦 Batch": e.batch, "🔎 Picture size": e.imgsz + " px" });
      break;
    case "batch": {
      const x = (e.epoch - 1) + e.batch / e.nb;
      const ds = charts.loss.data.datasets[0];
      ds.data.push({ x, y: total(e.loss) });
      ds.data = thin(ds.data);
      $("#b-txt").textContent = `${e.batch} / ${e.nb} batches`;
      $("#b-bar").style.width = (100 * e.batch / e.nb) + "%";
      $("#ep-txt").textContent = `${e.epoch} / ${v.epochs}`;
      $("#ep-bar").style.width = (100 * (e.epoch - 1 + e.batch / e.nb) / Math.max(1, v.epochs)) + "%";
      break;
    }
    case "epoch": {
      v.last = e;
      const quiz = total(e.val_loss);
      if (v.first == null && quiz) v.first = quiz;
      charts.loss.data.datasets[1].data.push({ x: e.epoch, y: quiz });
      charts.score.data.datasets[0].data.push({ x: e.epoch, y: e.map50 || 0 });
      charts.score.data.datasets[1].data.push({ x: e.epoch, y: e.precision || 0 });
      charts.score.data.datasets[2].data.push({ x: e.epoch, y: e.recall || 0 });
      charts.lr.data.datasets[0].data.push({ x: e.epoch, y: e.lr });
      charts.change.data.labels.push(e.epoch);
      ["backbone", "neck", "head"].forEach((k, i) => charts.change.data.datasets[i].data.push(e.change?.[k] || 0));
      v.best = Math.max(v.best, e.map50 || 0);
      $("#ep-txt").textContent = `${e.epoch} / ${e.epochs}`;
      $("#ep-bar").style.width = (100 * e.epoch / e.epochs) + "%";
      $("#live-status").textContent = mood(e, quiz);
      stats({ "🏆 Best score": pct(v.best), "🎯 Precision": pct(e.precision), "🔍 Recall": pct(e.recall),
              "⏱️ Per epoch": fmtTime(e.seconds), "⌛ Time left": fmtTime(e.eta) });
      break;
    }
    case "preview":
      v.previews[e.epoch] = e.items;
      $("#pv-epoch").max = e.epoch;
      $("#pv-epoch").disabled = false;
      if (+$("#pv-epoch").value >= e.epoch - 1 || !v.running) { $("#pv-epoch").value = e.epoch; renderPreview(e.epoch); }
      break;
    case "final":
      stats({ "🏁 Final score (mAP50)": pct(e.map50), "🎯 Precision": pct(e.precision), "🔍 Recall": pct(e.recall) });
      break;
    case "done":
      setTimeout(loadRuns, 1500);
      $("#live-status").textContent = `🎉 Done! Final score: ${pct(v.best)}. Try it in the Test tab, or give it to the car.`;
      break;
    case "stopped":
      setTimeout(loadRuns, 1500);
      $("#live-status").textContent = "⏹ Stopped. The AI kept everything it learned up to now.";
      break;
    case "error":
      setTimeout(loadRuns, 1500);
      $("#live-status").textContent = "😢 " + e.msg;
      break;
  }
}

function finished() {
  $("#tr-stop").disabled = true;
  loadRuns();
}

const pct = x => x == null ? "–" : Math.round(x * 100) + "%";

function mood(e, quiz) {
  const m = e.map50 || 0;
  const face = m < 0.2 ? "🤔 Still confused" : m < 0.5 ? "🙂 Getting the idea" : m < 0.8 ? "😃 Pretty good" : "🤩 Expert";
  let msg = `Epoch ${e.epoch} of ${e.epochs}: ${face} (score ${pct(m)}).`;
  if (view.first && quiz && e.epoch > 1) {
    const drop = 1 - quiz / view.first;
    if (drop > 0.02) msg += ` It makes ${Math.round(drop * 100)}% fewer mistakes than after epoch 1!`;
    else if (drop < -0.05) msg += " Mistakes went up a bit – that happens, keep watching.";
  }
  return msg;
}

function stats(obj) {
  const box = $("#live-stats");
  for (const [k, val] of Object.entries(obj)) {
    let el = [...box.children].find(c => c.dataset.k === k);
    if (!el) { el = document.createElement("div"); el.className = "stat"; el.dataset.k = k; box.appendChild(el); }
    el.innerHTML = `<b>${esc(val)}</b><span>${esc(k)}</span>`;
  }
}

function renderPreview(epoch) {
  if (!view) return;
  const items = view.previews[epoch];
  $("#pv-epoch-v").textContent = epoch;
  if (!items) return;
  const box = $("#previews");
  if (box.children.length !== items.length) {
    box.innerHTML = items.map(it => `<div><div class="stage"><img src="/runs/${view.id}/preview/${encodeURIComponent(it.image)}" alt="">
      <div class="box-layer"></div></div><div class="cap"></div></div>`).join("");
  }
  items.forEach((it, i) => {
    const cell = box.children[i];
    drawBoxes(cell.querySelector(".box-layer"),
      it.dets.map(d => ({ label: d[0], text: `${d[0]} ${Math.round(d[1] * 100)}%`, x1: d[2], y1: d[3], x2: d[4], y2: d[5] })));
    cell.querySelector(".cap").textContent = it.dets.length
      ? it.dets.map(d => `${d[0]} (${Math.round(d[1] * 100)}% sure)`).join(", ") : "Doesn't see anything yet";
  });
}

async function loadLog() {
  if (!view) return;
  const res = await fetch(`/api/runs/${view.id}/log`);
  const pre = $("#train-log");
  const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 10;
  // strip terminal colour codes and progress-bar redraws
  pre.textContent = (await res.text()).replace(/\x1b\[[0-9;]*[A-Za-z]/g, "").replace(/\r[^\n]*(?=\r)/g, "");
  if (atBottom) pre.scrollTop = pre.scrollHeight;
}

// ── past runs ───────────────────────────────────────────────────
async function loadRuns() {
  const runs = await api("/api/runs");
  $("#runs").innerHTML = runs.map(r => {
    const when = new Date(r.created * 1000).toLocaleString();
    const score = r.final?.map50 != null ? `score ${pct(r.final.map50)}` : "";
    const c = r.cfg || {};
    return `<div class="run" data-id="${r.id}">
      <div class="grow"><div class="nm">${esc(r.name)} <span class="pill ${r.state}">${r.state}</span></div>
      <div class="meta">${(r.classes || []).map(x => emoji(x) + " " + esc(x)).join(", ")} · ${esc(c.model || "")}${c.freeze ? " 🔒" : ""} · ${c.epochs || "?"} epochs · ${score} · ${when}</div>
      ${r.error ? `<div class="meta error">${esc(r.error)}</div>` : ""}</div>
      <button class="ghost" data-act="view">📊 View</button>
      ${r.best ? `<button class="ghost" data-act="test">🔍 Test</button><button class="ghost" data-act="car">🚗 Use for car</button>` : ""}
      <button class="ghost" data-act="del">🗑</button></div>`;
  }).join("") || `<p class="muted">No trained AIs yet.</p>`;
  $$("#runs .run button").forEach(b => b.onclick = async () => {
    const id = b.closest(".run").dataset.id, r = runs.find(x => x.id === id);
    if (b.dataset.act === "view") { attach(id); $("#live-card").scrollIntoView({ behavior: "smooth" }); }
    if (b.dataset.act === "test") window.dispatchEvent(new CustomEvent("lab:test", { detail: r.best }));
    if (b.dataset.act === "car") {
      try { await api("/api/models/active", { json: { path: r.best } }); toast(`🚗 The car now uses "${r.name}"`); }
      catch (e) { toast(e.message, true); }
    }
    if (b.dataset.act === "del" && confirm(`Delete "${r.name}"?`)) {
      try { await api(`/api/runs/${id}`, { method: "DELETE" }); loadRuns(); } catch (e) { toast(e.message, true); }
    }
  });
}
