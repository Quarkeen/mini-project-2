import { api, state, $, $$, initTips } from "./util.js";
import * as pictures from "./pictures.js";
import * as train from "./train.js";
import * as test from "./test.js";

const modules = { pictures, train, test };

function showTab(name) {
  $$("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach(s => s.classList.toggle("active", s.id === "tab-" + name));
  Object.entries(modules).forEach(([k, m]) => (k === name ? m.onShow?.() : m.onHide?.()));
  history.replaceState(null, "", "#" + name);
}

async function pollSystem() {
  try {
    const s = await api("/api/system");
    const parts = [];
    if (s.hottest != null) {
      const lim = s.thermal_limit || 90;
      const cls = s.hottest >= lim ? "hot" : s.hottest >= lim - 10 ? "warm" : "";
      parts.push(`<span class="${cls}" title="Hottest sensor">🌡️ ${Math.round(s.hottest)}°C</span>`);
    }
    parts.push(`<span title="CPU use">🧮 CPU ${Math.round(s.cpu)}%</span>`);
    if (s.gpu) parts.push(`<span title="GPU use">⚡ GPU ${Math.round(s.gpu.load)}%</span>`);
    parts.push(`<span title="Memory">💾 ${s.ram_used_gb.toFixed(1)}/${s.ram_total_gb.toFixed(0)} GB</span>`);
    if (s.training) parts.push(`<span class="busy">🧠 training</span>`);
    $("#sys").innerHTML = parts.join("");
  } catch { /* server restarting */ }
}

async function main() {
  initTips();
  state.info = await api("/api/info");
  state.emoji = state.info.label_emoji;
  $$("#tabs button").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));
  for (const m of Object.values(modules)) await m.init?.();
  const start = location.hash.slice(1);
  showTab(start in modules ? start : "pictures");
  pollSystem();
  setInterval(pollSystem, 4000);
}

main();
