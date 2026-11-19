#!/usr/bin/env bash
# SynthTrack – spuštění GUI (Linux). Pro přístup z jiného počítače: ./start_synthtrack.sh --host 0.0.0.0
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
exec python -m synthtrack serve "$@"
