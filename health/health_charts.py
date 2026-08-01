"""
health/health_charts.py — custom QPainter chart widgets for the LoRA Health dashboard.

Deliberately custom-painted rather than QtCharts: the whole app is a dark-navy Consolas telemetry UI,
and QPieSeries/QChart theming fights that at every turn. Hand-painting rings/bars/curves against the
shared.theme palette means zero re-styling, first-class gauges, and lightweight widgets that drop
straight into the existing results layout. All widgets guard empty/degenerate data (never raise in paint).

Widgets:
  RingGauge  — a donut gauge for the summary row (health score, rank utilization, completeness)
  LossCurve  — checkpoint-loss line+area chart with the best checkpoint marked
  ModuleBars — per-module magnitude bars, coloured by status (the Module Analysis at a glance)
"""
from __future__ import annotations

from PySide6.QtCore    import Qt, QRectF, QPointF
from PySide6.QtGui     import QPainter, QColor, QPen, QFont, QPainterPath, QPolygonF
from PySide6.QtWidgets import QWidget, QSizePolicy

from shared.theme import BG, PAN, CAR, ACC, GRN, RED, MUT, PRI, SEC, AMB, FONT

STATUS_COLOR = {"pass": GRN, "warn": AMB, "fail": RED, "info": ACC}


def _c(hexstr: str, alpha: int | None = None) -> QColor:
    q = QColor(hexstr)
    if alpha is not None:
        q.setAlpha(alpha)
    return q


def _centered_text(p: QPainter, cx: float, cy: float, s: str, px: int,
                   color: str, bold: bool = False) -> None:
    """Draw `s` centred on (cx, cy) — QPainter has no vertical-centre text primitive."""
    f = QFont(FONT); f.setPixelSize(px); f.setBold(bold)
    p.setFont(f); p.setPen(_c(color))
    fm = p.fontMetrics()
    tw = fm.horizontalAdvance(s)
    p.drawText(QPointF(cx - tw / 2, cy + (fm.ascent() - fm.descent()) / 2), s)


# ── Ring / donut gauge ────────────────────────────────────────────────────────
class RingGauge(QWidget):
    """A donut gauge: a full track + a value arc from 12 o'clock clockwise, with centred big/small text.
    `fraction` 0..1 sets the arc; `color` the arc colour (usually a status colour)."""

    def __init__(self, fraction: float, big: str, small: str = "", color: str = ACC,
                 diameter: int = 150, parent=None):
        super().__init__(parent)
        self._frac  = max(0.0, min(1.0, float(fraction)))
        self._big   = big
        self._small = small
        self._color = color
        self._dia   = diameter
        self.setMinimumSize(diameter, diameter)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def set_values(self, fraction: float, big: str, small: str = "", color: str | None = None):
        self._frac = max(0.0, min(1.0, float(fraction)))
        self._big, self._small = big, small
        if color:
            self._color = color
        self.update()

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(self._dia, self._dia)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        side = min(w, h)
        thick = max(9.0, side * 0.11)
        m = thick / 2 + 2
        rect = QRectF((w - side) / 2 + m, (h - side) / 2 + m, side - 2 * m, side - 2 * m)

        track = QPen(_c(PAN)); track.setWidthF(thick); track.setCapStyle(Qt.FlatCap)
        p.setPen(track); p.drawArc(rect, 0, 360 * 16)

        if self._frac > 0:
            arc = QPen(_c(self._color)); arc.setWidthF(thick); arc.setCapStyle(Qt.RoundCap)
            p.setPen(arc)
            p.drawArc(rect, 90 * 16, int(-self._frac * 360 * 16))   # top → clockwise

        cx, cy = rect.center().x(), rect.center().y()
        _centered_text(p, cx, cy - side * 0.015, self._big, int(side * 0.21), PRI, bold=True)
        if self._small:
            _centered_text(p, cx, cy + side * 0.17, self._small, max(9, int(side * 0.085)), SEC)
        p.end()


# ── Loss curve ────────────────────────────────────────────────────────────────
class LossCurve(QWidget):
    """Line + translucent area over checkpoint losses. `points` = list of (step:int, loss:float);
    the lowest-loss checkpoint is marked green. Degrades gracefully for 0/1 points."""

    def __init__(self, points: list[tuple[int, float]], parent=None):
        super().__init__(parent)
        self._pts = sorted((int(s), float(v)) for s, v in points if v is not None)
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        L, R, T, B = 48, 14, 14, 26          # margins (left axis, right, top, bottom axis)
        pw, ph = max(1, W - L - R), max(1, H - T - B)

        if not self._pts:
            _centered_text(p, W / 2, H / 2, "no checkpoint-loss data", 12, MUT)
            p.end(); return

        losses = [v for _, v in self._pts]
        lo, hi = min(losses), max(losses)
        if hi - lo < 1e-9:
            hi = lo + max(1e-6, abs(lo) * 0.1)   # avoid a flat divide-by-zero on a single value
        steps = [s for s, _ in self._pts]
        smin, smax = steps[0], steps[-1]
        srange = (smax - smin) or 1

        def X(s): return L + (s - smin) / srange * pw
        def Y(v): return T + (hi - v) / (hi - lo) * ph

        # gridlines + y labels (hi / mid / lo)
        p.setFont(_mono(10))
        for frac, val in ((0.0, hi), (0.5, (hi + lo) / 2), (1.0, lo)):
            y = T + frac * ph
            pen = QPen(_c(MUT)); pen.setWidthF(1.0); p.setPen(pen)
            p.drawLine(QPointF(L, y), QPointF(L + pw, y))
            p.setPen(_c(SEC))
            lbl = f"{val:.4f}"
            fm = p.fontMetrics()
            p.drawText(QPointF(L - 6 - fm.horizontalAdvance(lbl), y + fm.ascent() / 2 - 1), lbl)

        # best (lowest-loss) checkpoint
        best_i = min(range(len(self._pts)), key=lambda i: self._pts[i][1])

        # single point → just a marker
        if len(self._pts) == 1:
            s, v = self._pts[0]
            p.setBrush(_c(GRN)); p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(X(s), Y(v)), 5, 5)
            p.end(); return

        pts = [QPointF(X(s), Y(v)) for s, v in self._pts]

        # translucent area under the line
        area = QPainterPath()
        area.moveTo(pts[0].x(), T + ph)
        for pt in pts:
            area.lineTo(pt)
        area.lineTo(pts[-1].x(), T + ph)
        area.closeSubpath()
        p.setPen(Qt.NoPen); p.setBrush(_c(ACC, 46)); p.drawPath(area)

        # the line
        line = QPen(_c(ACC)); line.setWidthF(2.4)
        line.setJoinStyle(Qt.RoundJoin); line.setCapStyle(Qt.RoundCap)
        p.setPen(line); p.setBrush(Qt.NoBrush)
        p.drawPolyline(QPolygonF(pts))

        # per-point dots only on a sparse series (per-checkpoint); a dense per-step curve stays a clean
        # line. The best (lowest-loss) point is always marked.
        if len(pts) <= 48:
            p.setBrush(_c(ACC)); p.setPen(Qt.NoPen)
            for i, pt in enumerate(pts):
                if i != best_i:
                    p.drawEllipse(pt, 3.2, 3.2)
        p.setBrush(_c(GRN)); p.setPen(QPen(_c(CAR), 2))
        p.drawEllipse(pts[best_i], 5.5, 5.5)

        # x labels: first, last, and 'best' step
        p.setPen(_c(SEC)); p.setFont(_mono(10))
        fm = p.fontMetrics()
        def _xlabel(s, text, color=SEC):
            x = X(s); t = str(text)
            tw = fm.horizontalAdvance(t)
            x = min(max(L, x - tw / 2), L + pw - tw)
            p.setPen(_c(color)); p.drawText(QPointF(x, H - 8), t)
        _xlabel(smin, smin)
        _xlabel(smax, smax)
        bstep = self._pts[best_i][0]
        if bstep not in (smin, smax):
            _xlabel(bstep, f"best·{bstep}", GRN)
        else:
            # annotate the endpoint that is best
            _xlabel(bstep, f"best·{bstep}", GRN)
        p.end()


# ── Per-module magnitude bars ─────────────────────────────────────────────────
class ModuleBars(QWidget):
    """Horizontal bars — one per module group — length ∝ mean-abs magnitude, coloured by status.
    `rows` = list of dicts with keys: label, mean_abs, status, balance(optional), eff(optional 0..1)."""

    _ROW_H = 30
    _PAD_T = 6

    def __init__(self, rows: list[dict], parent=None):
        super().__init__(parent)
        self._rows = list(rows)
        self.setMinimumHeight(self._PAD_T * 2 + self._ROW_H * max(1, len(self._rows)))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        name_w = 150               # left column for module label
        val_w  = 118               # right column for the numeric readouts
        track_x = name_w + 8
        track_w = max(20, W - track_x - val_w - 8)

        if not self._rows:
            _centered_text(p, W / 2, H / 2, "no module data", 12, MUT)
            p.end(); return

        max_mag = max((r.get("mean_abs") or 0.0) for r in self._rows) or 1e-9

        for i, r in enumerate(self._rows):
            y = self._PAD_T + i * self._ROW_H
            cy = y + self._ROW_H / 2
            color = STATUS_COLOR.get(r.get("status", "info"), SEC)
            mag = r.get("mean_abs") or 0.0

            # module label
            p.setFont(_mono(11)); p.setPen(_c(PRI))
            fm = p.fontMetrics()
            label = r.get("label", "?")
            while fm.horizontalAdvance(label) > name_w and len(label) > 4:
                label = label[:-2]
            p.drawText(QPointF(2, cy + fm.ascent() / 2 - 2), label)

            # bar track + fill
            bh = 11.0
            by = cy - bh / 2
            p.setPen(Qt.NoPen)
            p.setBrush(_c(PAN)); p.drawRoundedRect(QRectF(track_x, by, track_w, bh), 5, 5)
            fill_w = max(3.0, track_w * min(1.0, mag / max_mag))
            p.setBrush(_c(color)); p.drawRoundedRect(QRectF(track_x, by, fill_w, bh), 5, 5)

            # right-side readouts: mean_abs, balance, eff%
            p.setFont(_mono(11))
            parts = [f"{mag:.4f}"]
            bal = r.get("balance")
            if bal is not None:
                parts.append(f"{bal:.1f}x")
            eff = r.get("eff")
            if eff is not None:
                parts.append(f"{eff*100:.0f}%")
            txt = "  ".join(parts)
            fm2 = p.fontMetrics()
            p.setPen(_c(SEC))
            p.drawText(QPointF(W - 4 - fm2.horizontalAdvance(txt), cy + fm2.ascent() / 2 - 2), txt)
        p.end()


def _mono(px: int) -> QFont:
    f = QFont(FONT); f.setPixelSize(px); return f
