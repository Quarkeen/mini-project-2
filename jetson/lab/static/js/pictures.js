// Pictures tab: library browser, uploads, box editor.
import { api, toast, state, $, $$, esc, emoji, drawBoxes, fromCxcywh, colorFor } from "./util.js";
import { BoxEditor } from "./annotate.js";

const PAGE = 60;
let libLabel = null, libOffset = 0;
let mineFilter = "unchecked", mineItems = [], mineOffset = 0, mineTotal = 0;
// edIndex: picture currently on the canvas (-1 while loading). edWant: last picture asked for.
let editor, edIndex = -1, edWant = -1, edDirty = false, edToken = 0;

export async function init() {
  editor = new BoxEditor($("#ed-canvas"), {
    getLabel: () => $("#ed-label").value.trim().toLowerCase(),
    onChange: () => { edDirty = true; renderBoxList(); },
  });

  $("#lib-close").onclick = () => { $("#lib-gallery-wrap").classList.add("hidden"); libLabel = null; renderLibClasses(window._classes || []); };
  $("#lib-more").onclick = () => loadLibrary(false);
  $("#mine-more").onclick = () => loadMine(false);
  $("#vw-close").onclick = () => $("#viewer").classList.add("hidden");
  $("#viewer").onclick = e => { if (e.target.id === "viewer") $("#viewer").classList.add("hidden"); };

  $$("#mine-filters .chip").forEach(b => b.onclick = () => {
    mineFilter = b.dataset.f;
    $$("#mine-filters .chip").forEach(x => x.classList.toggle("active", x === b));
    loadMine(true);
  });

  $("#up-files").onchange = e => upload([...e.target.files]);
  const dz = $("#dropzone");
  dz.ondragover = e => { e.preventDefault(); dz.classList.add("over"); };
  dz.ondragleave = () => dz.classList.remove("over");
  dz.ondrop = e => { e.preventDefault(); dz.classList.remove("over"); upload([...e.dataTransfer.files].filter(f => f.type.startsWith("image/"))); };

  // editor buttons
  $("#ed-close").onclick = closeEditor;
  $("#ed-prev").onclick = () => openEditor(edWant - 1);
  $("#ed-next").onclick = () => openEditor(edWant + 1);
  $("#ed-del-box").onclick = () => editor.deleteSelected();
  $("#ed-save").onclick = () => saveCurrent(true);
  $("#ed-suggest").onclick = suggest;
  $("#ed-del-pic").onclick = deleteCurrent;
  document.addEventListener("keydown", e => {
    if ($("#editor").classList.contains("hidden") || ["INPUT", "SELECT"].includes(e.target.tagName)) return;
    // Enter/Space on a focused button already "clicks" it; don't also run the shortcut
    if (e.target.tagName === "BUTTON" && (e.key === "Enter" || e.key === " ")) return;
    if (e.key === "Delete" || e.key === "Backspace") { editor.deleteSelected(); e.preventDefault(); }
    if (e.key === "ArrowRight") openEditor(edWant + 1);
    if (e.key === "ArrowLeft") openEditor(edWant - 1);
    if (e.key === "Enter") saveCurrent(true);
    if (e.key === "Escape") closeEditor();
  });
}

export async function onShow() {
  await refreshClasses();
  loadMine(true);
}

// ── classes & labels ────────────────────────────────────────────
export async function refreshClasses() {
  const classes = await api("/api/classes");
  window._classes = classes;
  renderLibClasses(classes);
  const known = new Set([...classes.map(c => c.label), ...Object.keys(state.emoji)]);
  $("#label-list").innerHTML = [...known].sort().map(l => `<option value="${esc(l)}">`).join("");
  return classes;
}

function renderLibClasses(classes) {
  const lib = classes.filter(c => c.library > 0);
  $("#lib-classes").innerHTML = lib.map(c => `
    <div class="class-card ${c.label === libLabel ? "active" : ""}" data-label="${esc(c.label)}">
      <div class="emo">${emoji(c.label)}</div><div class="nm">${esc(c.label)}</div>
      <div class="ct">${c.library} pictures</div></div>`).join("") || `<p class="muted">The library is empty. Run <code>python scripts/build_library.py</code>.</p>`;
  $$("#lib-classes .class-card").forEach(el => el.onclick = () => { libLabel = el.dataset.label; renderLibClasses(classes); loadLibrary(true); });
}

function thumbHtml(store, it, i) {
  const boxes = it.boxes.map(fromCxcywh);
  const badge = store === "mine" ? (it.reviewed ? "✅" : (it.boxes.length ? "✏️ check" : "✏️ needs boxes")) : "";
  const tags = [...new Set(it.boxes.map(b => b[0]))].join(", ") || it.hint || "";
  return `<div class="thumb" data-i="${i}">
      <div class="stage" style="width:100%;height:100%"><img loading="lazy" src="/img/${store}/${encodeURIComponent(it.name)}?thumb=1" alt="">
      <div class="box-layer">${boxes.map(b => `<div class="bx" style="--c:${colorFor(b.label)};left:${b.x1 * 100}%;top:${b.y1 * 100}%;width:${(b.x2 - b.x1) * 100}%;height:${(b.y2 - b.y1) * 100}%"></div>`).join("")}</div></div>
      ${badge ? `<span class="badge">${badge}</span>` : ""}${tags ? `<span class="tags">${esc(tags)}</span>` : ""}</div>`;
}

// object-fit: contain letterboxes the thumbnail, so box overlays need the image's real box:
// we size the stage to the image instead once it has loaded.
function fitThumbs(root) {
  $$(".thumb img", root).forEach(img => {
    const fit = () => {
      const st = img.parentElement, th = st.parentElement;
      const r = Math.min(th.clientWidth / img.naturalWidth, th.clientHeight / img.naturalHeight);
      st.style.width = img.naturalWidth * r + "px";
      st.style.height = img.naturalHeight * r + "px";
      st.style.margin = `${(th.clientHeight - img.naturalHeight * r) / 2}px auto 0`;
      img.style.objectFit = "fill";
    };
    if (img.complete && img.naturalWidth) fit(); else img.addEventListener("load", fit, { once: true });
  });
}

// ── library ─────────────────────────────────────────────────────
let libItems = [];
async function loadLibrary(reset) {
  if (reset) { libOffset = 0; libItems = []; }
  const d = await api(`/api/pictures/library?label=${encodeURIComponent(libLabel)}&limit=${PAGE}&offset=${libOffset}`);
  libItems = libItems.concat(d.items);
  libOffset += d.items.length;
  $("#lib-gallery-wrap").classList.remove("hidden");
  $("#lib-gallery-title").textContent = `${emoji(libLabel)} ${libLabel} — ${d.total} pictures`;
  $("#lib-gallery").innerHTML = libItems.map((it, i) => thumbHtml("library", it, i)).join("");
  $("#lib-more").classList.toggle("hidden", libOffset >= d.total);
  fitThumbs($("#lib-gallery"));
  $$("#lib-gallery .thumb").forEach(el => el.onclick = () => view("library", libItems[+el.dataset.i]));
}

function view(store, it) {
  $("#vw-title").textContent = [...new Set(it.boxes.map(b => b[0]))].join(", ");
  const img = $("#vw-img");
  img.src = `/img/${store}/${encodeURIComponent(it.name)}`;
  drawBoxes($("#vw-boxes"), it.boxes.map(fromCxcywh));
  $("#viewer").classList.remove("hidden");
}

// ── my pictures ─────────────────────────────────────────────────
async function upload(files) {
  if (!files.length) return;
  const label = $("#up-label").value.trim().toLowerCase();
  const st = $("#up-status");
  let done = 0, failed = [];
  // small batches keep memory low on the Jetson when kids drop 50 phone photos at once
  for (let i = 0; i < files.length; i += 4) {
    const fd = new FormData();
    fd.append("label", label);
    files.slice(i, i + 4).forEach(f => fd.append("files", f));
    st.textContent = `Uploading ${Math.min(i + 4, files.length)} / ${files.length}…`;
    try {
      const r = await api("/api/mine/upload", { method: "POST", body: fd });
      done += r.names.length;
      failed = failed.concat(r.errors);
    } catch (e) { failed.push(e.message); }
  }
  st.textContent = `Added ${done} picture${done === 1 ? "" : "s"}.` + (failed.length ? ` ${failed.length} failed.` : "");
  if (failed.length) toast(failed[0], true);
  $("#up-files").value = "";
  mineFilter = "unchecked";
  $$("#mine-filters .chip").forEach(x => x.classList.toggle("active", x.dataset.f === "unchecked"));
  await loadMine(true);
  refreshClasses();
  if (done) toast("Now draw boxes on your new pictures ✏️");
}

async function loadMine(reset) {
  if (reset) { mineOffset = 0; mineItems = []; }
  const d = await api(`/api/pictures/mine?filter=${mineFilter}&limit=${PAGE}&offset=${mineOffset}`);
  mineItems = mineItems.concat(d.items);
  mineOffset += d.items.length;
  mineTotal = d.total;
  $("#mine-count").textContent = `${d.total} picture${d.total === 1 ? "" : "s"}`;
  $("#mine-gallery").innerHTML = mineItems.map((it, i) => thumbHtml("mine", it, i)).join("") ||
    `<p class="muted">${mineFilter === "unchecked" ? "Nothing waiting for boxes 🎉" : "No pictures yet."}</p>`;
  $("#mine-more").classList.toggle("hidden", mineOffset >= d.total);
  fitThumbs($("#mine-gallery"));
  $$("#mine-gallery .thumb").forEach(el => el.onclick = () => openEditor(+el.dataset.i));
}

// ── editor ──────────────────────────────────────────────────────
async function openEditor(i) {
  const token = ++edToken;      // a newer openEditor call wins; stale image loads are ignored
  if (edDirty && edIndex >= 0 && !(await saveCurrent(false))) return;
  if (i >= mineItems.length && mineOffset < mineTotal) await loadMine(false);
  if (token !== edToken) return;
  if (i < 0 || i >= mineItems.length) { if (i >= mineItems.length) toast("That was the last picture 👍"); return; }
  edIndex = -1;                 // nothing is saved until this picture has loaded
  edWant = i;
  const it = mineItems[i];
  $("#editor").classList.remove("hidden");
  $("#ed-pos").textContent = `${i + 1} / ${mineTotal}`;
  $("#ed-title").textContent = it.reviewed ? "✅ Checked picture" : "✏️ Draw boxes";
  $("#ed-label").value = it.hint || it.boxes[0]?.[0] || $("#ed-label").value || $("#up-label").value || "";
  $("#ed-helper").textContent = "";
  await editor.load(`/img/mine/${encodeURIComponent(it.name)}`, it.boxes);
  if (token !== edToken) return;
  edIndex = i;
  edDirty = false;
}

function renderBoxList() {
  const labels = [...new Set([...(window._classes || []).map(c => c.label), ...editor.boxes.map(b => b.label)])].sort();
  $("#ed-boxes").innerHTML = editor.boxes.map((b, i) => `
    <div class="box-item ${i === editor.sel ? "sel" : ""}" data-i="${i}">
      <span class="sw" style="background:${colorFor(b.label)}"></span>
      <select data-i="${i}">${labels.map(l => `<option ${l === b.label ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>
      ${b.unsure ? "❓" : ""}</div>`).join("") || `<p class="hint">No boxes yet.</p>`;
  $$("#ed-boxes .box-item").forEach(el => el.onclick = e => { if (e.target.tagName !== "SELECT") editor.select(+el.dataset.i); });
  $$("#ed-boxes select").forEach(s => s.onchange = () => { editor.setLabel(+s.dataset.i, s.value); edDirty = true; renderBoxList(); });
}

async function saveCurrent(markChecked) {
  const it = mineItems[edIndex];
  if (!it) return true;
  const boxes = editor.getBoxes();
  if (markChecked && !boxes.length) { toast("Draw at least one box first (or delete the picture).", true); return false; }
  if (markChecked && editor.boxes.some(b => b.unsure)) { toast("Check the dashed ❓ boxes first: move or resize them if they're wrong.", true); return false; }
  try {
    const rec = await api(`/api/mine/${encodeURIComponent(it.name)}/boxes`,
      { json: { boxes, reviewed: markChecked ? true : it.reviewed && boxes.length > 0 } });
    Object.assign(it, rec);
    edDirty = false;
    if (markChecked) {
      toast("Saved ✅");
      refreshThumb(edIndex);
      openEditor(edIndex + 1);
    }
    return true;
  } catch (e) { toast(e.message, true); return false; }
}

function refreshThumb(i) {
  const el = $(`#mine-gallery .thumb[data-i="${i}"]`);
  if (!el) return;
  el.outerHTML = thumbHtml("mine", mineItems[i], i);
  const n = $(`#mine-gallery .thumb[data-i="${i}"]`);
  fitThumbs(n.parentElement);
  n.onclick = () => openEditor(i);
}

async function suggest() {
  const it = mineItems[edIndex];
  if (!it) return;                 // picture still loading
  const label = $("#ed-label").value.trim().toLowerCase();
  if (!label) { toast("Type what to look for in 'Label for new boxes'.", true); return; }
  const btn = $("#ed-suggest");
  btn.disabled = true;
  $("#ed-helper").textContent = "🤖 Looking…";
  try {
    const r = await api(`/api/mine/${encodeURIComponent(it.name)}/suggest`, { json: { labels: [label] } });
    editor.addBoxes(r.boxes.map(b => [...b.slice(0, 5), null, true]));   // every suggestion starts "unsure"
    $("#ed-helper").textContent = r.boxes.length
      ? `Found ${r.boxes.length} (${r.used.join(", ")}). Dashed boxes are guesses: fix them, then save.`
      : `Couldn't find any ${label}. Draw the box yourself!`;
    if (r.warning) $("#ed-helper").textContent += " " + r.warning;
  } catch (e) { $("#ed-helper").textContent = e.message; }
  btn.disabled = false;
}

async function deleteCurrent() {
  const it = mineItems[edIndex];
  if (!it || !confirm("Delete this picture?")) return;
  await api(`/api/mine/${encodeURIComponent(it.name)}`, { method: "DELETE" });
  mineItems.splice(edIndex, 1);
  mineTotal--;
  edDirty = false;
  await loadMineKeepEditor();
}

async function loadMineKeepEditor() {
  const i = edIndex;
  await loadMine(true);
  if (mineItems.length) openEditor(Math.min(i, mineItems.length - 1)); else closeEditor();
}

async function closeEditor() {
  if (edDirty && edIndex >= 0) await saveCurrent(false);
  $("#editor").classList.add("hidden");
  edIndex = -1;
  loadMine(true);
  refreshClasses();
}
