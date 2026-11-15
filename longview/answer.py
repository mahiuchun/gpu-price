"""The 2 to 3 year answer for the README: three pictures and one table.

    uv run python longview/answer.py

Reads only what longview.py and resale.py already wrote to longview/out/, plus the published resale
figures in longview/data/ (no data pack, no keys, no network). Writes into docs/media/:
  longview_levels.png   today's AWS spot rent against each chip's usual level and AWS's list price
  longview_rent.png     rent to Oct 2029: middle path, 10% to 90% range, a 25%-a-year comparison
  longview_resale.png   used-module resale price to Oct 2029, same layout
and prints the table that the README section shows. No number is typed in here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "longview" / "out"
OUT = REPO / "docs" / "media"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BAND = {BLUE: "#cde2fb", ORANGE: "#fbd9c9"}      # light tint of each colour, for ranges
CHIPS = (("H100", ORANGE), ("A100", BLUE))        # one colour per chip, the same in every picture
QUOTED_PART = {"H100": "H100 SXM", "A100": "A100 80GB SXM"}
TREND = "hardware_trend_25"                       # the comparison path: rent falls a steady 25% a year

_have = {f.name for f in font_manager.fontManager.ttflist}
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": [f for f in ("Helvetica Neue", "Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans") if f in _have],
    "font.size": 11, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
    "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    "text.parse_math": False,                     # a label with two dollar signs is money, not a formula
}


def usd(v: float) -> str:
    """Dollars for a GPU module: $340, $2.3k, $12k."""
    if v < 1000:
        return f"${v:,.0f}"
    return f"${v / 1000:.1f}k".replace(".0k", "k")


def load() -> dict:
    key = json.loads((SRC / "key_numbers.json").read_text())
    out = pd.read_csv(SRC / "outlook.csv", parse_dates=["target"])
    weekly = pd.read_csv(SRC / "outlook_paths_weekly.csv", parse_dates=["date"])
    resale = pd.read_csv(SRC / "resale_outlook.csv", parse_dates=["target_date"])
    daily = pd.read_csv(SRC / "series_daily.csv", parse_dates=["date"])
    quotes = pd.read_csv(REPO / "longview" / "data" / "resale_quotes.csv")
    origin = pd.Timestamp(key["step4"]["origin"])
    d = {"origin": origin, "list_cut": pd.Timestamp(key["list_price"]["cut_reached_spot_ceilings_on"]), "chips": {}}
    for chip, _ in CHIPS:
        fit = key["step4"]["fit"][chip]
        now = fit["last_price"]
        main = out[(out.chip == chip) & (out.variant == "main")].sort_values("years_ahead")
        trend = out[(out.chip == chip) & (out.variant == TREND)].sort_values("years_ahead")
        w = weekly[(weekly.chip == chip) & (weekly.variant == "main")].sort_values("date")
        w = w[w.date < main.target.iloc[-1]]
        # the weekly file stops a few days short of the 3-year target: start at today, end on the target row
        path = pd.DataFrame({
            "date": [origin, *w.date, main.target.iloc[-1]],
            "mid": [now, *w.q50, main.q50.iloc[-1]],
            "low": [now, *w.q10_widened, main.q10_widened.iloc[-1]],
            "high": [now, *w.q90_widened, main.q90_widened.iloc[-1]],
        })
        hist = daily[(daily.series == f"{chip}_pack") & (daily.date >= d["list_cut"])]
        rs = resale[(resale.chip == chip) & (resale.scenario == "aws_spot_outlook")].sort_values("years_ahead")
        rt = resale[(resale.chip == chip) & (resale.scenario == TREND)].sort_values("years_ahead")
        q = quotes[(quotes.gpu == QUOTED_PART[chip]) & quotes.unit.str.contains("bare module")].iloc[0]
        lo, hi = (float(v) for v in q.value.split("-"))
        assert (lo + hi) / 2 == rs.quote_today_bare_module_usd.iloc[0], "the range drawn is not the one the model used"
        d["chips"][chip] = dict(
            now=now, usual=fit["usual_level"], cap=key["step4"]["list_price_cap"][chip],
            rent=main, rent_trend=trend, path=path, hist=hist, resale=rs, resale_trend=rt, quote_range=(lo, hi),
        )
    return d


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(REPO))


def end_labels(ax, x, items, gap):
    """Labels at the right end of the lines, pushed apart so that none overlap each other or the axis."""
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    ys[0] = max(ys[0], gap * 0.6)
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for y, (_, text, color, size) in zip(ys, items):
        ax.text(x, y, text, va="center", ha="left", color=color, fontsize=size)


def years_axis(ax, d):
    """Both time charts share one axis: from AWS's June 2025 list-price cut to three years ahead."""
    ax.set_xlim(d["list_cut"] - pd.Timedelta(days=25), d["origin"] + pd.Timedelta(days=3 * 365 + 250))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def fig_levels(d: dict) -> None:
    chips = [c for c, _ in CHIPS][::-1]            # drawn bottom to top
    colour = dict(CHIPS)
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    for y, chip in enumerate(chips):
        c, col = d["chips"][chip], colour[chip]
        ax.plot([c["usual"], c["now"]], [y, y], color=AXIS, linewidth=2, zorder=2)
        ax.plot([c["cap"]] * 2, [y - 0.17, y + 0.17], color=INK2, linewidth=2, zorder=2, solid_capstyle="butt")
        ax.scatter([c["usual"]], [y], s=70, facecolor=SURFACE, edgecolor=col, linewidth=2.2, zorder=3)
        ax.scatter([c["now"]], [y], s=100, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(c["usual"] - 0.13, y, f"${c['usual']:.2f}", va="center", ha="right", color=INK2, fontsize=10)
        ax.text(c["now"] + 0.13, y, f"${c['now']:.2f}", va="center", color=INK2, fontsize=10)
        ax.text((c["usual"] + c["now"]) / 2, y + 0.13, f"{100 * (c['now'] / c['usual'] - 1):+.0f}% above usual", va="bottom",
                ha="center", color=INK2, fontsize=10)
        ax.text(c["cap"] + 0.1, y, f"${c['cap']:.2f}", va="center", color=INK2, fontsize=10)
    ax.set_yticks(range(len(chips)), chips)
    ax.set_xlim(0, 8.6)
    ax.set_ylim(-0.6, len(chips) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("USD per GPU-hour")
    ax.set_title("Both chips rent well above their usual level, and under AWS's list price")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=SURFACE, markeredgecolor=INK2,
               markeredgewidth=2, label="usual level: the middle of the chip's own price history"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=INK2, label=f"AWS spot on {d['origin']:%-d %b %Y}"),
        Line2D([], [], marker="|", linestyle="", markersize=13, markeredgewidth=2, color=INK2,
               label="AWS list price: no zone has traded above it since June 2025"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, fontsize=9.5)
    fig.text(0.01, -0.36, "AWS EC2 Linux spot, median across zones (H100: p5.48xlarge, 28 zones; A100: p4d.24xlarge, 19). "
             "Source: longview/out/key_numbers.json", color=MUTED, fontsize=8.5)
    save(fig, "longview_levels.png")


def fig_rent(d: dict) -> None:
    origin = d["origin"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, (chip, color) in zip(axes, CHIPS):
        c = d["chips"][chip]
        p, r, t, h = c["path"], c["rent"], c["rent_trend"], c["hist"]
        top = c["cap"] * 1.1
        before, after = h[h.segment == "before_hole"], h[h.segment == "after_hole"]
        ax.axvspan(before.date.max(), after.date.min(), color=GRID, alpha=0.55, linewidth=0, zorder=0)
        ax.text(before.date.max() + (after.date.min() - before.date.max()) / 2, top * 0.03, "no data", rotation=90,
                ha="center", va="bottom", color=MUTED, fontsize=8.5)
        for k, seg in enumerate((before, after)):      # two separate lines: nothing is drawn across the gap
            ax.plot(seg.date, seg.usd_per_gpu_hour, color=color, linewidth=1.3,
                    label=f"price to {origin:%-d %b %Y}" if k == 0 else None)
        ax.plot(p.date, p.mid, color=color, linewidth=2.2, label="middle of 20,000 simulated futures")
        ax.fill_between(p.date, p.low, p.high, color=BAND[color], linewidth=0, label="10% to 90% range")
        ax.plot([origin, *t.target], [c["now"], *t.q50], color=INK2, linewidth=1.4, linestyle=(0, (4, 3)),
                label="comparison: a steady fall of 25% a year")
        ax.plot([d["list_cut"], p.date.iloc[-1]], [c["cap"]] * 2, color=INK2, linewidth=1, linestyle=(0, (1, 2.5)),
                label=f"AWS list price ${c['cap']:.2f}: a ceiling")
        ax.scatter([origin], [c["now"]], s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        recent = after[after.date > origin - pd.Timedelta(days=330)].usd_per_gpu_hour.max()
        ax.text(origin - pd.Timedelta(days=6), max(recent, c["now"]) + top * 0.035, f"now ${c['now']:.2f}", color=INK2,
                fontsize=10, ha="right", va="bottom")
        two = r[r.years_ahead == 2].iloc[0]
        ax.scatter(r.target.iloc[1:], r.q50.iloc[1:], s=45, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.text(two.target, two.q50 + top * 0.035, f"{two.target:%b %Y}\n${two.q50:.2f}", color=INK2, fontsize=9.5,
                ha="center", va="bottom", linespacing=1.25)
        last, x = r.iloc[-1], p.date.iloc[-1] + pd.Timedelta(days=30)
        end_labels(ax, x, [
            (last.q90_widened, f"high ${last.q90_widened:.2f}", MUTED, 9.5),
            (last.q50, f"middle ${last.q50:.2f}", INK2, 10),
            (t.q50.iloc[-1], f"25%/yr ${t.q50.iloc[-1]:.2f}", MUTED, 9.5),
            (last.q10_widened, f"low ${last.q10_widened:.2f}", MUTED, 9.5),
        ], gap=top * 0.062)
        ax.set_title(f"{chip} rent: ${last.q50:.2f} in {last.target:%b %Y} "
                     f"(range ${last.q10_widened:.2f}–${last.q90_widened:.2f})")
        years_axis(ax, d)
        ax.set_ylim(0, top)
        ax.set_ylabel("USD per GPU-hour")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9, columnspacing=1.6)
    fig.tight_layout(w_pad=2.5)
    fig.text(0.01, -0.03, f"AWS EC2 Linux spot, median across zones, forecast from {origin:%-d %b %Y}. History starts at AWS's "
             "June 2025 list-price cut. Tested to about six months; three years is untested. "
             "Source: longview/out/outlook.csv", color=MUTED, fontsize=8.5)
    save(fig, "longview_rent.png")


def fig_resale(d: dict) -> None:
    origin = d["origin"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, (chip, color) in zip(axes, CHIPS):
        c = d["chips"][chip]
        r, t = c["resale"], c["resale_trend"]
        lo, mid, hi = r.resale_usd_bare_module_low, r.resale_usd_bare_module_central, r.resale_usd_bare_module_high
        alt = t.resale_usd_bare_module_central
        today, top = mid.iloc[0], hi.max() * 1.1
        ax.fill_between(r.target_date, lo, hi, color=BAND[color], linewidth=0, label="10% to 90% range")
        ax.plot(r.target_date, mid, color=color, linewidth=2.2, label="middle")
        ax.plot(t.target_date, alt, color=INK2, linewidth=1.4, linestyle=(0, (4, 3)),
                label="comparison: rent falls a steady 25% a year")
        ax.plot([origin] * 2, c["quote_range"], color=color, linewidth=2, solid_capstyle="butt", zorder=3)
        ax.scatter([origin], [today], s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.text(origin - pd.Timedelta(days=26), today, f"now ${today:,.0f}\nused price, published\n"
                f"range {usd(c['quote_range'][0])}–{usd(c['quote_range'][1])}", color=INK2, fontsize=9.5,
                ha="right", va="center", linespacing=1.35)
        ax.scatter(r.target_date.iloc[1:], mid.iloc[1:], s=45, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
        for k in (1, 2):
            ax.text(r.target_date.iloc[k], mid.iloc[k] + top * 0.035, f"{r.target_date.iloc[k]:%b %Y}\n{usd(mid.iloc[k])}",
                    color=INK2, fontsize=9.5, ha="center", va="bottom", linespacing=1.25)
        same = alt.iloc[-1] == mid.iloc[-1]            # both at the floor: one label says so
        items = [(hi.iloc[-1], f"high {usd(hi.iloc[-1])}", MUTED, 9.5),
                 (mid.iloc[-1], f"middle {usd(mid.iloc[-1])}" + (" (and at 25%/yr)" if same else ""), INK2, 10),
                 (lo.iloc[-1], f"low {usd(lo.iloc[-1])}", MUTED, 9.5)]
        if not same:
            items.append((alt.iloc[-1], f"25%/yr {usd(alt.iloc[-1])}", MUTED, 9.5))
        end_labels(ax, r.target_date.iloc[-1] + pd.Timedelta(days=30), items, gap=top * 0.062)
        ax.set_title(f"{chip} resale: {usd(mid.iloc[-1])} in {r.target_date.iloc[-1]:%b %Y} "
                     f"(range {usd(lo.iloc[-1])}–{usd(hi.iloc[-1])})")
        years_axis(ax, d)
        ax.set_ylim(0, top)
        ax.yaxis.set_major_formatter(lambda v, _: "$0" if v == 0 else f"${v / 1000:.0f}k")
        ax.set_ylabel("USD per used GPU module")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9, columnspacing=1.6)
    fig.tight_layout(w_pad=2.5)
    fig.text(0.01, -0.03, "Bare GPU module (H100 SXM, A100 80GB SXM). Middle: today's published used price × the share of its value "
             "the model has left. Worked out at +1, +2 and +3 years, joined by straight lines.\nToday's price: midpoint of the "
             "used-price range in American Compute, GPU Prices, October 2026, https://www.amcompute.com/gpu-prices. "
             "Source: longview/out/resale_outlook.csv", color=MUTED, fontsize=8.5, va="top", linespacing=1.4)
    save(fig, "longview_resale.png")


def table(d: dict) -> str:
    rows = ["| GPU | When | Rent low | **Rent middle** | Rent high | Resale low | **Resale middle** | Resale high |",
            "|---|---|---|---|---|---|---|---|"]
    for chip, _ in CHIPS:
        c = d["chips"][chip]
        today = c["resale"].resale_usd_bare_module_central.iloc[0]
        rows.append(f"| {chip} | now, {d['origin']:%-d %b %Y} | | ${c['now']:.2f} | | | ${today:,.0f} | |")
        for k in (2, 3):
            r = c["rent"][c["rent"].years_ahead == k].iloc[0]
            s = c["resale"][c["resale"].years_ahead == k].iloc[0]
            mid = s.resale_usd_bare_module_central
            rows.append(
                f"| {chip} | {r.target:%b %Y} (+{k} years) | ${r.q10_widened:.2f} | **${r.q50:.2f}** | ${r.q90_widened:.2f} "
                f"| ${s.resale_usd_bare_module_low:,.0f} | **${mid:,.0f}** ({100 * mid / today:.0f}% of today's) "
                f"| ${s.resale_usd_bare_module_high:,.0f} |")
    return "\n".join(rows)


def main() -> None:
    d = load()
    with plt.rc_context(STYLE):
        fig_levels(d)
        fig_rent(d)
        fig_resale(d)
    print(table(d))


if __name__ == "__main__":
    main()
