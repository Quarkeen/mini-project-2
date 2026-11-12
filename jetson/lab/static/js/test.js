// Test tab: try a model on an uploaded picture, this device's webcam, or the Jetson camera.
import { api, toast, state, $, $$, esc, emoji, drawBoxes } from "./util.js";

let src = "upload", stream = null, camGen = 0, jetTimer = null;

export async function init() {
  $("#ts-conf").oninput = () => ($("#ts-conf-v").textContent = (+$("#ts-conf").value / 100).toFixed(2));
  $("#ts-file").onchange = e => e.target.files[0] && runUpload(e.target.files[0]);
  $$("#ts-sources .chip").forEach(b => b.onclick = () => setSource(b.dataset.src));
  $("#ts-cam-start").onclick = startWebcam;
  $("#ts-cam-stop").onclick = stopWebcam;
  $("#ts-jet-start").onclick = startJetson;
  $("#ts-jet-stop").onclick = stopJetson;
  if (state.info.jetson) setSource("jetson");      // the Jetson's own camera is the useful one here
  window.addEventListener("lab:test", async e => {
    document.querySelector('#tabs button[data-tab="test"]').click();
    await loadModels(e.detail);
  });
}

export async function onShow() { await loadModels(); }
export function onHide() { stopWebcam(); stopJetson(); }

async function loadModels(select) {
  const d = await api("/api/models");
  const sel = $("#ts-model"), keep = select || sel.value || d.active;
  sel.innerHTML = d.models.map(m => `<option value="${esc(m.path)}">${m.kind === "trained" ? "🧠" : m.kind === "pretrained" ? "🌍" : "📦"} ${esc(m.name)}${m.classes ? " — " + m.classes.join(", ") : ""}</option>`).join("");
  if ([...sel.options].some(o => o.value === keep)) sel.value = keep;
}

function setSource(s) {
  stopWebcam(); stopJetson();
  src = s;
  $$("#ts-sources .chip").forEach(b => b.classList.toggle("active", b.dataset.src === s));
  ["upload", "webcam", "jetson"].forEach(k => $("#ts-" + k).classList.toggle("hidden", k !== s));
  clearView();
}

function clearView() {
  $("#ts-img").removeAttribute("src");
  $("#ts-img").classList.remove("hidden");
  $("#ts-video").classList.add("hidden");
  $("#ts-boxes").innerHTML = "";
  $("#ts-empty").classList.remove("hidden");
  $("#ts-say").textContent = "🤖 …";
  $("#ts-list").innerHTML = "";
  $("#ts-speed").textContent = "";
}

function form(blob) {
  const fd = new FormData();
  fd.append("image", blob, "photo.jpg");
  fd.append("model", $("#ts-model").value);
  fd.append("conf", +$("#ts-conf").value / 100);
  return fd;
}

function showResult(r) {
  drawBoxes($("#ts-boxes"), r.detections.map(d => ({
    label: d.label, text: `${d.label} ${Math.round(d.confidence * 100)}%`,
    x1: d.norm[0], y1: d.norm[1], x2: d.norm[2], y2: d.norm[3],
  })));
  $("#ts-say").textContent = "🤖 " + r.description;
  $("#ts-list").innerHTML = r.detections.map(d => `<div class="det">${emoji(d.label)} <b>${esc(d.label)}</b>
    <div class="bar"><div style="width:${d.confidence * 100}%"></div></div>${Math.round(d.confidence * 100)}%</div>`).join("");
  $("#ts-speed").textContent = r.speed_ms != null ? `Thought for ${r.speed_ms} ms on the ${r.device === "cpu" ? "CPU" : "GPU"}` : "";
}

// ── picture ─────────────────────────────────────────────────────
async function runUpload(file) {
  clearView();
  $("#ts-img").src = URL.createObjectURL(file);
  $("#ts-empty").classList.add("hidden");
  $("#ts-say").textContent = "🤖 Looking…";
  try { showResult(await api("/api/detect", { method: "POST", body: form(file) })); }
  catch (e) { $("#ts-say").textContent = "😢 " + e.message; }
  $("#ts-file").value = "";
}

// ── this device's webcam ────────────────────────────────────────
async function startWebcam() {
  if (stream) return;            // already running
  const gen = ++camGen;
  if (!navigator.mediaDevices?.getUserMedia) {
    // plain http://<ip> is not a "secure context": the browser hides its camera API completely
    toast("This browser only allows its webcam on localhost, so I switched to the Jetson camera.", true);
    setSource("jetson");
    return;
  }
  let st;
  try {
    st = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  } catch (e) {
    const none = e.name === "NotFoundError";
    toast(none ? "No webcam found on this device. Try the Jetson camera." : "Camera not available: " + e.message, true);
    if (none) setSource("jetson");
    return;
  }
  if (gen !== camGen) { st.getTracks().forEach(t => t.stop()); return; }   // Stop pressed meanwhile
  stream = st;
  const v = $("#ts-video");
  v.srcObject = stream;
  await v.play();
  v.classList.remove("hidden");
  $("#ts-img").classList.add("hidden");
  $("#ts-empty").classList.add("hidden");
  const canvas = document.createElement("canvas");
  // send the next frame only after the previous answer arrives: a slow Jetson
  // automatically gets fewer frames instead of a growing queue
  while (gen === camGen) {
    canvas.width = v.videoWidth; canvas.height = v.videoHeight;
    canvas.getContext("2d").drawImage(v, 0, 0);
    const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.8));
    try {
      const fd = form(blob);
      fd.append("imgsz", 480);
      const r = await api("/api/detect", { method: "POST", body: fd });
      if (gen === camGen) showResult(r);
    } catch (e) { $("#ts-say").textContent = "😢 " + e.message; await new Promise(r => setTimeout(r, 1000)); }
    await new Promise(r => setTimeout(r, 120));
  }
}

function stopWebcam() {
  camGen++;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  $("#ts-video").classList.add("hidden");
}

// ── Jetson camera (MJPEG from the server) ───────────────────────
function startJetson() {
  clearView();
  const q = new URLSearchParams({ model: $("#ts-model").value, conf: +$("#ts-conf").value / 100, t: Date.now() });
  $("#ts-img").src = "/api/camera/stream?" + q;
  $("#ts-empty").classList.add("hidden");
  $("#ts-say").textContent = "🤖 Starting the camera…";
  clearInterval(jetTimer);
  jetTimer = setInterval(async () => {
    const i = await api("/api/camera/info").catch(() => null);
    if (!i) return;
    if (i.error) { $("#ts-say").textContent = "😢 " + i.error; if (!i.running) stopJetson(); return; }
    if (i.description) {
      $("#ts-say").textContent = "🤖 " + i.description;
      $("#ts-list").innerHTML = (i.detections || []).map(d => `<div class="det">${emoji(d.label)} <b>${esc(d.label)}</b>
        <div class="bar"><div style="width:${d.confidence * 100}%"></div></div>${Math.round(d.confidence * 100)}%</div>`).join("");
      $("#ts-speed").textContent = `Thought for ${i.speed_ms} ms`;
    }
  }, 1000);
}

function stopJetson() {
  clearInterval(jetTimer);
  jetTimer = null;
  if (src === "jetson") $("#ts-img").removeAttribute("src");   // closes the MJPEG connection
}
