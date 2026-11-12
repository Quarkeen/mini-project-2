// Canvas box editor: drag to draw, drag a box to move, drag a corner to resize.
// Works with mouse and touch (pointer events).
import { colorFor } from "./util.js";

const HANDLE_PX = 14;
const MIN_SIZE = 0.01;

export class BoxEditor {
  constructor(canvas, { getLabel, onChange }) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.getLabel = getLabel;
    this.onChange = onChange || (() => {});
    this.img = null;
    this.boxes = [];        // {label, x1, y1, x2, y2, unsure?} normalised
    this.sel = -1;
    this.drag = null;
    canvas.addEventListener("pointerdown", e => this._down(e));
    canvas.addEventListener("pointermove", e => this._move(e));
    canvas.addEventListener("pointerup", e => this._up(e));
    canvas.addEventListener("pointercancel", e => this._up(e));
    new ResizeObserver(() => this.render()).observe(canvas);
  }

  load(url, boxes) {
    const tok = this._loadTok = (this._loadTok || 0) + 1;
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        if (tok !== this._loadTok) { resolve(false); return; }   // a newer picture was requested
        this.img = img;
        const k = Math.min(1, 1400 / Math.max(img.naturalWidth, img.naturalHeight));
        this.cv.width = Math.round(img.naturalWidth * k);
        this.cv.height = Math.round(img.naturalHeight * k);
        this.setBoxes(boxes);
        resolve(true);
      };
      img.onerror = reject;
      img.src = url;
    });
  }

  setBoxes(boxes) {
    this.boxes = (boxes || []).map(b => ({
      label: b[0], x1: b[1] - b[3] / 2, y1: b[2] - b[4] / 2, x2: b[1] + b[3] / 2, y2: b[2] + b[4] / 2, unsure: !!b[6],
    }));
    this.sel = -1;
    this.render();
    this.onChange();
  }

  addBoxes(boxes) {
    for (const b of boxes) {
      this.boxes.push({ label: b[0], x1: b[1] - b[3] / 2, y1: b[2] - b[4] / 2, x2: b[1] + b[3] / 2, y2: b[2] + b[4] / 2, unsure: !!b[6] });
    }
    this.render();
    this.onChange();
  }

  getBoxes() {
    return this.boxes.map(b => {
      const x1 = Math.max(0, Math.min(b.x1, b.x2)), x2 = Math.min(1, Math.max(b.x1, b.x2));
      const y1 = Math.max(0, Math.min(b.y1, b.y2)), y2 = Math.min(1, Math.max(b.y1, b.y2));
      return [b.label, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1];
    });
  }

  select(i) { this.sel = i; this.render(); this.onChange(); }

  deleteSelected() {
    if (this.sel < 0) return;
    this.boxes.splice(this.sel, 1);
    this.sel = -1;
    this.render();
    this.onChange();
  }

  setLabel(i, label) {
    if (this.boxes[i]) { this.boxes[i].label = label; this.boxes[i].unsure = false; this.render(); }
  }

  // ── pointer handling ───────────────────────────────────────────
  _pt(e) {
    const r = this.cv.getBoundingClientRect();
    return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height, r };
  }

  _corner(b, p) {
    const hx = HANDLE_PX / p.r.width, hy = HANDLE_PX / p.r.height;
    const corners = [["x1", "y1"], ["x2", "y1"], ["x1", "y2"], ["x2", "y2"]];
    return corners.find(([kx, ky]) => Math.abs(p.x - b[kx]) < hx && Math.abs(p.y - b[ky]) < hy);
  }

  _hit(p) {
    // smallest box under the pointer wins (so small boxes inside big ones stay clickable)
    let best = -1, area = Infinity;
    this.boxes.forEach((b, i) => {
      const x1 = Math.min(b.x1, b.x2), x2 = Math.max(b.x1, b.x2), y1 = Math.min(b.y1, b.y2), y2 = Math.max(b.y1, b.y2);
      if (p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2 && (x2 - x1) * (y2 - y1) < area) {
        best = i; area = (x2 - x1) * (y2 - y1);
      }
    });
    return best;
  }

  _down(e) {
    if (!this.img) return;
    this.cv.setPointerCapture(e.pointerId);
    const p = this._pt(e);
    const selBox = this.boxes[this.sel];
    const corner = selBox && this._corner(selBox, p);
    if (corner) {
      this.drag = { mode: "resize", kx: corner[0], ky: corner[1] };
      return;
    }
    const hit = this._hit(p);
    if (hit >= 0 && hit === this.sel) {
      // only the selected box moves; dragging anywhere else draws, so a big
      // box covering the whole photo never stops kids from drawing new ones
      this.drag = { mode: "move", ox: p.x, oy: p.y, start: { ...selBox } };
    } else {
      // wait to see if this is a click (select) or a drag (draw)
      this.drag = { mode: "pending", ox: p.x, oy: p.y, hit, px: e.clientX, py: e.clientY };
    }
    this.render();
    this.onChange();
  }

  _move(e) {
    const p = this._pt(e);
    if (!this.drag) {
      const selBox = this.boxes[this.sel];
      this.cv.style.cursor = selBox && this._corner(selBox, p) ? "nwse-resize" : this._hit(p) >= 0 ? "move" : "crosshair";
      return;
    }
    if (this.drag.mode === "pending") {
      if (Math.hypot(e.clientX - this.drag.px, e.clientY - this.drag.py) < 6) return;
      const { ox, oy } = this.drag;
      this.boxes.push({ label: this.getLabel() || "object", x1: ox, y1: oy, x2: ox, y2: oy });
      this.sel = this.boxes.length - 1;
      this.drag = { mode: "draw" };
    }
    const b = this.boxes[this.sel];
    const cx = Math.max(0, Math.min(1, p.x)), cy = Math.max(0, Math.min(1, p.y));
    if (this.drag.mode === "draw") { b.x2 = cx; b.y2 = cy; }
    else if (this.drag.mode === "resize") { b[this.drag.kx] = cx; b[this.drag.ky] = cy; b.unsure = false; }
    else {
      const s = this.drag.start, w = s.x2 - s.x1, h = s.y2 - s.y1;
      const dx = Math.max(-Math.min(s.x1, s.x2), Math.min(1 - Math.max(s.x1, s.x2), p.x - this.drag.ox));
      const dy = Math.max(-Math.min(s.y1, s.y2), Math.min(1 - Math.max(s.y1, s.y2), p.y - this.drag.oy));
      b.x1 = s.x1 + dx; b.y1 = s.y1 + dy; b.x2 = b.x1 + w; b.y2 = b.y1 + h; b.unsure = false;
    }
    this.render();
  }

  _up() {
    if (!this.drag) return;
    if (this.drag.mode === "pending") {      // plain click: select the box under the pointer (or none)
      this.sel = this.drag.hit;
      this.drag = null;
      this.render();
      this.onChange();
      return;
    }
    const b = this.boxes[this.sel];
    if (b && (Math.abs(b.x2 - b.x1) < MIN_SIZE || Math.abs(b.y2 - b.y1) < MIN_SIZE)) {
      this.boxes.splice(this.sel, 1);       // a click, not a drag
      this.sel = -1;
    }
    this.drag = null;
    this.render();
    this.onChange();
  }

  // ── drawing ────────────────────────────────────────────────────
  render() {
    const { ctx, cv } = this;
    if (!this.img) return;
    const W = cv.width, H = cv.height;
    const scale = W / Math.max(1, cv.getBoundingClientRect().width);  // keep strokes the same on-screen size
    ctx.drawImage(this.img, 0, 0, W, H);
    ctx.font = `bold ${Math.round(14 * scale)}px sans-serif`;
    this.boxes.forEach((b, i) => {
      const x = Math.min(b.x1, b.x2) * W, y = Math.min(b.y1, b.y2) * H;
      const w = Math.abs(b.x2 - b.x1) * W, h = Math.abs(b.y2 - b.y1) * H;
      const c = colorFor(b.label);
      ctx.lineWidth = (i === this.sel ? 4 : 3) * scale;
      ctx.strokeStyle = c;
      ctx.setLineDash(b.unsure ? [8 * scale, 6 * scale] : []);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
      if (i === this.sel) { ctx.fillStyle = "rgba(255,255,255,.15)"; ctx.fillRect(x, y, w, h); }
      const text = b.label + (b.unsure ? " ?" : "");
      const tw = ctx.measureText(text).width + 10 * scale, th = 20 * scale;
      ctx.fillStyle = c;
      ctx.fillRect(x, Math.max(0, y - th), tw, th);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, x + 5 * scale, Math.max(0, y - th) + th - 6 * scale);
      if (i === this.sel) {
        const hs = HANDLE_PX * scale * 0.7;
        ctx.fillStyle = "#fff"; ctx.strokeStyle = c; ctx.lineWidth = 2 * scale;
        for (const [hx, hy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) {
          ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs);
          ctx.strokeRect(hx - hs / 2, hy - hs / 2, hs, hs);
        }
      }
    });
  }
}
