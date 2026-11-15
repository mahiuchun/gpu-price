"""Build the narrated demo: slide stills with a slow zoom, crossfades, narration, captions.

    python3 docs/slides/render_slides.py        # slide PNGs in docs/slides/png/ (needs Pillow: uv run --with pillow)
    python3 docs/media/demo/build_demo.py       # -> docs/media/demo/gpu-price-demo.mp4, demo.srt, poster.jpg

Needs macOS `say` and ffmpeg. Set VOICE and RATE to change the narrator. Narration: docs/media/demo/script.json.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
PNG = f"{REPO}/docs/slides/png"
V = tempfile.mkdtemp()
LEAD, TAIL, X = 0.3, 0.45, 0.6  # silence before/after each line; crossfade length
# Voice: any `say -v '?'` name. A Premium or Enhanced voice (System Settings > Accessibility >
# Spoken Content > Manage Voices) sounds far more natural, e.g. VOICE="Zoe (Premium)".
VOICE = os.environ.get("VOICE", "Samantha")
RATE = os.environ.get("RATE", "192")  # words per minute
TITLE = "RSI4GPR: A Crew of AI Agents and One Coach AI with Recursive Self-Improvement for GPU Price Forecasting"


def run(cmd):
    subprocess.run(cmd, check=True)


def dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                capture_output=True, text=True).stdout)


segs = json.load(open(f"{REPO}/docs/media/demo/script.json"))
starts, lens, durs = [], [], []
t = 0.0
for i, s in enumerate(segs):
    spoken = s["text"].replace("RSI4GPR", "R S I 4 G P R")
    open(f"{V}/n{i:02d}.txt", "w").write(spoken)
    run(["say", "-v", VOICE, "-r", RATE, "-f", f"{V}/n{i:02d}.txt", "-o", f"{V}/n{i:02d}.raw.aiff"])
    # trim the voice's own leading/trailing silence, then even out loudness
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{V}/n{i:02d}.raw.aiff", "-af",
         "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
         "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
         "highpass=f=80,loudnorm=I=-18:TP=-2:LRA=7", "-ar", "48000", f"{V}/n{i:02d}.aiff"])
    d = dur(f"{V}/n{i:02d}.aiff")
    L = round(LEAD + d + TAIL + X, 3)
    starts.append(t); lens.append(L); durs.append(d)
    t += L - X
total = round(starts[-1] + lens[-1], 3)

# 1. one silent clip per segment
for i, s in enumerate(segs):
    L, out = lens[i], f"{V}/v{i:02d}.mp4"
    if "slide" in s:
        vf = (f"scale=w='trunc(1920*(1+0.035*t/{L})/2)*2':h=-2:eval=frame,crop=1920:1080,"
              f"setsar=1,fps=30,format=yuv420p")
        run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-t", str(L), "-i",
             f"{PNG}/{s['slide']}.png", "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-t", str(L), out])
    else:
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-t", str(L), "-i", "color=c=0xFCFCFB:s=1920x1080:r=30",
             "-stream_loop", "-1", "-i", f"{REPO}/docs/media/engine_replay.mp4", "-filter_complex",
             f"[1:v]scale=1600:-2,fps=30[c];[0:v][c]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p,trim=0:{L}[v]",
             "-map", "[v]", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-t", str(L), out])

# 2. crossfade the clips and lay the narration on the same timeline
cmd = ["ffmpeg", "-y", "-loglevel", "error"]
for i in range(len(segs)):
    cmd += ["-i", f"{V}/v{i:02d}.mp4"]
for i in range(len(segs)):
    cmd += ["-i", f"{V}/n{i:02d}.aiff"]
n = len(segs)
fc, prev = [], "[0:v]"
for i in range(1, n):
    lab = f"[x{i}]" if i < n - 1 else "[xv]"
    fc.append(f"{prev}[{i}:v]xfade=transition=fade:duration={X}:offset={starts[i]:.3f}{lab}")
    prev = lab
fc.append(f"[xv]fade=t=in:st=0:d=0.6,fade=t=out:st={total - 0.8:.3f}:d=0.8,format=yuv420p[v]")
for i in range(n):
    ms = int((starts[i] + LEAD) * 1000)
    fc.append(f"[{n + i}:a]aformat=sample_rates=48000:channel_layouts=stereo,adelay={ms}|{ms}[a{i}]")
fc.append("".join(f"[a{i}]" for i in range(n)) + f"amix=inputs={n}:normalize=0,apad,atrim=0:{total}[a]")
cmd += ["-filter_complex", ";".join(fc), "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "medium",
        "-crf", "20", "-r", "30", "-c:a", "aac", "-b:a", "160k", "-t", str(total), f"{V}/body.mp4"]
run(cmd)

# 3. captions, sentence by sentence, timed by length within each line
def ts(x):
    m, s = int(x // 60), x % 60
    ms = int(round((s - int(s)) * 1000)) % 1000
    return f"00:{m:02d}:{int(s):02d},{ms:03d}"


srt, k = [], 1
for i, s in enumerate(segs):
    sents = [x.strip() for x in re.split(r"(?<=[.?!])\s+", s["text"]) if x.strip()]
    tot_chars, c = sum(len(x) for x in sents), starts[i] + LEAD
    for x in sents:
        dt = durs[i] * len(x) / tot_chars
        srt.append(f"{k}\n{ts(c)} --> {ts(c + dt)}\n{x}\n"); k += 1; c += dt
open(f"{REPO}/docs/media/demo/demo.srt", "w").write("\n".join(srt))

out = f"{REPO}/docs/media/demo/gpu-price-demo.mp4"
run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{V}/body.mp4", "-i", f"{REPO}/docs/media/demo/demo.srt",
     "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
     "-metadata", f"title={TITLE}", "-movflags", "+faststart", out])
run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "2.5", "-i", out, "-frames:v", "1", "-vf", "scale=1280:-2",
     f"{REPO}/docs/media/demo/poster.jpg"])
shutil.rmtree(V)
print("total", total, "s;", n, "segments")
