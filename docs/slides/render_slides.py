"""Render the slide deck in docs/slides/source/ to PNGs and a PDF, using images from this repo.

    python3 docs/slides/render_slides.py                 # all slides -> docs/slides/png/, docs/slides/RSI4GPR-slides.pdf
    python3 docs/slides/render_slides.py crowd-flow ...  # only these slides (PNGs only)

Needs Google Chrome (or set CHROME=/path/to/chrome) and Pillow (uv run --with pillow).
Fonts come from Google Fonts, so the first run needs network access.
The deck uses Slides elements (<x-icon>, <x-connector>); they are drawn here as inline SVG.
Speaker notes and footers are not drawn.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "docs" / "slides" / "source"
OUT = REPO / "docs" / "slides" / "png"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

# images uploaded to the deck, by asset id -> file in this repo
BLOBS = {
    "161f83370fe2aa40b65c6105d71e886b": "docs/media/hero_scoreboard.png",
    "bd14ca7ac25147572e95a45ab7722056": "docs/media/engine_replay.gif",
    "a462c21574ff230d87a14830baa73953": "docs/media/forecast_6nov.png",
    "36aebe8eef7b0c6fa48c824adf4e6e4b": "findings/figures/fig3_what_moves_the_median.png",
    "7095c1e5e4b279403021e7596ce09002": "findings/figures/fig6_three_tests.png",
    "414ad1dcacba7c9afc7dd9989f3c9b8f": "findings/figures/fig7_self_improvement.png",
    "dc33b328be33648200f86fc9507efa5f": "findings/figures/fig8_rl_runs.png",
    "2a04ba39fa188604af46ddc190397c9a": "findings/figures/fig10_euro_weekday.png",
    "6173ba263bb5c0e562742004f9c82f39": "findings/figures/fig11_coverage_when_moved.png",
    "11baae926109028102046de04697e5e1": "findings/figures/fig9_range_memory.png",
    "28f65509036cadcd7ab44c4b60245049": "findings/figures/fig1_blind_weeks.png",
    "0cfe67f7d4880ff1d4177a5a3b62e354": "findings/figures/fig2_real_vs_shuffled.png",
    "ad8e8e732a80baf4a8afa6e4aedc389e": "docs/media/pipeline.svg",
}

# Lucide-style line icons (ISC licence), for the icon names the deck uses
ICONS = {
    "Database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "Clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "Search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "Activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "Chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "Link": '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
    "Lightning": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "Play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "Wrench": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "Lock": '<rect width="18" height="11" x="3" y="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "Users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "Settings": '<line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/><line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/><line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/><line x1="2" x2="6" y1="14" y2="14"/><line x1="10" x2="14" y1="8" y2="8"/><line x1="18" x2="22" y1="16" y2="16"/>',
    "PaperPlane": '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
    "Verified": '<path d="M3.85 8.62a4 4 0 0 1 4.78-4.77 4 4 0 0 1 6.74 0 4 4 0 0 1 4.78 4.78 4 4 0 0 1 0 6.74 4 4 0 0 1-4.77 4.78 4 4 0 0 1-6.75 0 4 4 0 0 1-4.78-4.77 4 4 0 0 1 0-6.76Z"/><path d="m9 12 2 2 4-4"/>',
    "Star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "Book": '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>',
}

BASE_CSS = ("*{margin:0} body{margin:0} body>section{width:1920px!important;height:1080px!important;"
            "overflow:hidden!important;box-sizing:border-box!important;position:relative} "
            "aside,.footer{display:none!important} *{animation:none!important} x-icon{display:block;flex:none}")


def icon(m):
    return (f'<x-icon{m.group(2)}><svg viewBox="0 0 24 24" width="100%" height="100%" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[m.group(1)]}</svg></x-icon>')


def connector(m):
    a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(0)))
    st = a.get("style", "")
    col = re.search(r"color:([^;\"]+)", st).group(1)
    width = re.search(r"border-width:(\d+)", st).group(1)
    dash = "8 6" if "dashed" in st else ""
    mid = "h" + re.sub(r"\W", "", col)
    head = "" if a.get("head") == "none" else f'marker-end="url(#{mid})"'
    return (f'<svg style="position:absolute;left:0;top:0;overflow:visible" width="1" height="1"><defs>'
            f'<marker id="{mid}" orient="auto" markerWidth="4" markerHeight="4" refX="2" refY="2">'
            f'<path d="M0 0L4 2L0 4Z" fill="{col}"/></marker></defs><line x1="{a["x1"]}" y1="{a["y1"]}" '
            f'x2="{a["x2"]}" y2="{a["y2"]}" stroke="{col}" stroke-width="{width}" stroke-dasharray="{dash}" {head}/></svg>')


def slide_html(slide_id, fonts_href):
    s = (SRC / "slides" / f"{slide_id}.html").read_text()
    s = re.sub(r"/_blob/([0-9a-f]{32})", lambda m: (REPO / BLOBS[m.group(1)]).as_uri(), s)
    s = re.sub(r'<x-icon name="(\w+)"([^>]*)></x-icon>', icon, s)
    s = re.sub(r"<x-connector[^>]*></x-connector>", connector, s)
    css = (SRC / "deck.css").read_text()
    return (f'<!doctype html><meta charset=utf-8><link rel=stylesheet href="{fonts_href}">'
            f"<style>{BASE_CSS}{css}</style>{s}")


def render(slide_ids):
    deck = json.loads((SRC / "deck.json").read_text())
    fonts = next(iter(deck["faces"].values()))["href"]
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for sid in slide_ids:
            page = Path(tmp) / f"{sid}.html"
            page.write_text(slide_html(sid, fonts))
            subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
                            "--force-device-scale-factor=1", "--virtual-time-budget=5000", "--window-size=1920,1080",
                            f"--screenshot={OUT / (sid + '.png')}", page.as_uri()],
                           check=True, capture_output=True)
            print("wrote", (OUT / f"{sid}.png").relative_to(REPO))
    return deck


if __name__ == "__main__":
    deck = json.loads((SRC / "deck.json").read_text())
    ids = sys.argv[1:] or deck["order"]
    render(ids)
    if not sys.argv[1:]:
        from PIL import Image
        pages = [Image.open(OUT / f"{sid}.png").convert("RGB") for sid in ids]
        pdf = REPO / "docs" / "slides" / "RSI4GPR-slides.pdf"
        pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=144)
        print("wrote", pdf.relative_to(REPO), f"({len(pages)} pages)")
