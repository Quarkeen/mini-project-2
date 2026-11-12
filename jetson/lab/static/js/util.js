// Shared helpers.

export const state = { info: null, emoji: {} };

export async function api(path, opts = {}) {
  const o = { ...opts };
  if (o.json !== undefined) {
    o.method = o.method || "POST";
    o.headers = { "Content-Type": "application/json" };
    o.body = JSON.stringify(o.json);
    delete o.json;
  }
  const res = await fetch(path, o);
  let data = null;
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
  return data;
}

let toastTimer;
export function toast(msg, bad = false) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.toggle("bad", bad);
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), bad ? 6000 : 3000);
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Stable colour per label
export function colorFor(label) {
  let h = 0;
  for (const ch of String(label)) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return `hsl(${h}, 80%, 45%)`;
}

export const emoji = label => state.emoji[label] || "🔷";

// Boxes as {label, x1, y1, x2, y2 (0-1), text?}
export function drawBoxes(layer, boxes) {
  layer.innerHTML = boxes.map(b => {
    const c = colorFor(b.label);
    return `<div class="bx" style="--c:${c};left:${b.x1 * 100}%;top:${b.y1 * 100}%;` +
      `width:${(b.x2 - b.x1) * 100}%;height:${(b.y2 - b.y1) * 100}%"><span>${esc(b.text ?? b.label)}</span></div>`;
  }).join("");
}

// store format [label, cx, cy, w, h] -> {label, x1..}
export const fromCxcywh = b => ({ label: b[0], x1: b[1] - b[3] / 2, y1: b[2] - b[4] / 2, x2: b[1] + b[3] / 2, y2: b[2] + b[4] / 2 });

export function fmtTime(sec) {
  if (sec == null || !isFinite(sec)) return "–";
  sec = Math.round(sec);
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60), s = sec % 60;
  if (m < 60) return `${m}m ${s}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

// "?" tooltips
export function initTips() {
  const tip = document.getElementById("tip");
  document.addEventListener("pointerover", e => {
    const q = e.target.closest(".q");
    if (!q) return;
    tip.textContent = q.dataset.tip;
    tip.classList.remove("hidden");
    const r = q.getBoundingClientRect();
    tip.style.left = Math.min(window.innerWidth - 300, r.left) + "px";
    tip.style.top = (r.bottom + 6) + "px";
  });
  document.addEventListener("pointerout", e => { if (e.target.closest(".q")) tip.classList.add("hidden"); });
}
