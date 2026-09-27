#!/usr/bin/env bash
# Rebuild the synthetic clip and clean it; the README numbers come from this.
# Extra arguments go to video-cleaner, e.g.:
#   demo/run_demo.sh --model tiny
#   demo/run_demo.sh --transcript demo/output/talking_head.transcript.json   # no Whisper
set -euo pipefail
cd "$(dirname "$0")/.."
out=demo/output

python3 demo/make_clip.py "$out/talking_head.mp4"
python3 -m video_cleaner "$out/talking_head.mp4" \
    -o "$out/talking_head.clean.mp4" --model base --language en "$@"

# Before/after waveforms on the same time scale (copied to docs/ for the README).
duration() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
before=$(duration "$out/talking_head.mp4")
after=$(duration "$out/talking_head.clean.mp4")
width=$(python3 -c "print(round(1200 * $after / $before))")
ffmpeg -hide_banner -loglevel error -y -i "$out/talking_head.mp4" -i "$out/talking_head.clean.mp4" \
    -filter_complex "\
[0:a]showwavespic=s=1200x120:scale=sqrt:colors=0x64748b,\
drawtext=text='before  $(printf %.2f "$before") s':x=4:y=4:fontsize=18:fontcolor=0x334155[a];\
[1:a]showwavespic=s=${width}x120:scale=sqrt:colors=0x2563eb,pad=1200:120:color=white,\
drawtext=text='after  $(printf %.2f "$after") s':x=4:y=4:fontsize=18:fontcolor=0x1e3a8a[b];\
[a][b]vstack,pad=1220:260:10:10:color=white" \
    -frames:v 1 "$out/waveforms.png"
echo "wrote $out/waveforms.png"
