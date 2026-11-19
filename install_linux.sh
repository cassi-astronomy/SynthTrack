#!/usr/bin/env bash
# SynthTrack – instalace na Linuxu (Ubuntu 24.04 / 26.04 a podobné)
#
#   ./install_linux.sh                         # kód + GPU podle karty (AMD → ROCm, NVIDIA → CUDA, jinak CPU)
#   ./install_linux.sh --data ~/synthtrack-data.tar   # navíc převezme přístroje, mastery, katalogy a nastavení
#                                                     (rozbalí i synthtrack-data-masters-*.tar ze stejné složky)
#   ./install_linux.sh --cpu | --rocm | --cuda    # vynutit backend
#
# Předpoklady pro AMD: ROCm v systému (Ubuntu 26.04: sudo apt install rocm),
# uživatel ve skupinách render,video (sudo usermod -a -G render,video $USER, pak se odhlásit).
set -euo pipefail
cd "$(dirname "$0")"

DATA=""; GPU="auto"
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    --cpu) GPU="cpu"; shift ;;
    --rocm) GPU="rocm"; shift ;;
    --cuda) GPU="cuda"; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "Neznámý parametr: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- Python
# CuPy pro ROCm/CUDA nemusí mít balíčky pro nejnovější Python – přednost má 3.12/3.13
say "Python"
PY=""
for cand in python3.12 python3.13 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    PY="$cand"; break
  fi
done
[ -n "$PY" ] || { echo "Chybí Python ≥ 3.10 (sudo apt install python3)"; exit 1; }
echo "použit: $PY ($("$PY" -V))"
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv || { echo "Chybí venv: sudo apt install python3-venv  (případně python3.12-venv)"; exit 1; }
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q -U pip
say "SynthTrack (základ)"
pip install -q -e ".[cosmics]" || pip install -q -e .

# ---------------------------------------------------------------- GPU
if [ "$GPU" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then GPU="cuda"
  elif [ -e /dev/kfd ] || grep -qs 0x1002 /sys/class/drm/card*/device/vendor; then GPU="rocm"
  else GPU="cpu"; fi
fi
say "GPU: $GPU"
case "$GPU" in
  rocm)
    if ! command -v rocminfo >/dev/null 2>&1 && [ ! -x /opt/rocm/bin/rocminfo ]; then
      echo "ROCm není nainstalovaný. Ubuntu 26.04:  sudo apt install rocm  a  sudo usermod -a -G render,video \$USER"
      echo "Pak se odhlaste/restartujte a spusťte skript znovu. Zatím poběží CPU."
    else
      id -nG | grep -qw render || echo "POZOR: uživatel není ve skupině render (sudo usermod -a -G render,video \$USER, pak odhlásit)."
      (rocminfo 2>/dev/null || /opt/rocm/bin/rocminfo) | grep -m1 -o 'gfx[0-9a-f]*' | sed 's/^/architektura: /' || true
      pip install -q -e ".[rocm]" || echo "POZOR: CuPy pro ROCm se nepodařilo nainstalovat – poběží CPU (pošlete výpis)."
    fi ;;
  cuda)
    pip install -q -e ".[cuda12]" || echo "POZOR: CuPy pro CUDA se nepodařilo nainstalovat – poběží CPU." ;;
esac

# ---------------------------------------------------------------- data
if [ -n "$DATA" ]; then
  say "Data z $DATA"
  if [ -f "$DATA" ]; then                # archiv: rozbalí i ostatní části synthtrack-data*.tar vedle něj
    tmp="$(mktemp -d)"
    for part in "$(dirname "$DATA")"/synthtrack-data*.tar; do echo "rozbaluji $(basename "$part")"; tar -xf "$part" -C "$tmp"; done
    DATA="$(find "$tmp" -maxdepth 2 -type d -name instruments -printf '%h\n' | head -1)"
  fi
  synthtrack import-data "$DATA"
fi

# ---------------------------------------------------------------- kontrola
say "Kontrola"
synthtrack gpu || true
if command -v astap >/dev/null 2>&1 || command -v astap_cli >/dev/null 2>&1 || [ -x /opt/astap/astap ]; then
  echo "ASTAP: nalezen"
else
  echo "ASTAP: CHYBÍ – stáhněte z https://www.hnsky.org/astap.htm balíček astap_amd64.deb"
  echo "       a databázi hvězd (stejnou jako na Windows, např. d50 / w08), nainstalujte:"
  echo "       sudo apt install ./astap_amd64.deb ./d50_star_database.deb"
fi
chmod +x start_synthtrack.sh
say "Hotovo"
echo "Spuštění:  ./start_synthtrack.sh      (GUI na http://localhost:8765, z jiného PC přes SSH tunel)"
echo "Test bez dat:  source .venv/bin/activate && synthtrack demo"
