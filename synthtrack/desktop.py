"""Desktop application: the web GUI in its own native window (no browser, no URL).

``synthtrack app`` (or the console-less ``synthtrack-app.exe`` on Windows)
starts the API server on a free local port in a background thread and opens
a pywebview window on it (Edge WebView2 on Windows, GTK/Qt WebKit on Linux).
Closing the window stops the program; a running job asks for confirmation.

``synthtrack shortcut`` creates a desktop / Start-menu shortcut (Windows .lnk,
Linux .desktop) so the program starts with a double click.

The browser mode (``synthtrack serve``) stays available, e.g. for remote
control of an observatory computer.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from .gpu import NO_WINDOW

ICON_DIR = Path(__file__).parent / "web" / "static"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DesktopApi:
    """Functions callable from the page as ``window.pywebview.api.<name>()``."""

    def __init__(self):
        self._window = None
        self._runner = None

    def _dialog(self, kind_new: str, kind_old: str, **kw):
        import webview

        fd = getattr(webview, "FileDialog", None)
        kind = getattr(fd, kind_new) if fd is not None else getattr(webview, kind_old)
        return self._window.create_file_dialog(kind, **kw)

    def pick_folder(self, start: str = "") -> str | None:
        res = self._dialog("FOLDER", "FOLDER_DIALOG", directory=start or "")
        if not res:
            return None
        return res[0] if isinstance(res, (list, tuple)) else str(res)

    def pick_file(self, start: str = "") -> str | None:
        res = self._dialog("OPEN", "OPEN_DIALOG", directory=start or "")
        if not res:
            return None
        return res[0] if isinstance(res, (list, tuple)) else str(res)

    def save_text(self, filename: str, content: str) -> str | None:
        res = self._dialog("SAVE", "SAVE_DIALOG", save_filename=filename)
        if not res:
            return None
        path = res[0] if isinstance(res, (list, tuple)) else str(res)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def open_folder(self, path: str) -> bool:
        """Show a folder in Explorer / the file manager."""
        try:
            if os.name == "nt":
                os.startfile(path)                                    # noqa: S606
            else:
                subprocess.Popen(["xdg-open", path])
            return True
        except Exception:
            return False


def run_app(settings, width: int = 1500, height: int = 950) -> None:
    try:
        import webview
    except ImportError:
        print("Chybí knihovna pywebview. Nainstalujte: pip install -e \".[desktop]\"", file=sys.stderr)
        raise SystemExit(2)
    import uvicorn

    from .web.app import create_app

    # without a console (synthtrack-app.exe / pythonw) stdout/stderr are None: log to a file
    if sys.stdout is None or sys.stderr is None:
        log = open(Path(settings.root) / "synthtrack-app.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log
    port = _free_port()
    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                           log_config=None))
    th = threading.Thread(target=server.run, name="synthtrack-server", daemon=True)
    th.start()
    for _ in range(200):                      # wait until the server answers
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)

    api = DesktopApi()
    api._runner = app.state.runner
    win = webview.create_window("SynthTrack", f"http://127.0.0.1:{port}/", js_api=api,
                                width=width, height=height, min_size=(900, 600),
                                background_color="#0f1216", text_select=True)
    api._window = win

    def on_closing():
        runner = api._runner
        if runner is not None and runner.current:
            ok = win.create_confirmation_dialog(
                "SynthTrack", "Právě běží zpracování úlohy. Opravdu ukončit? Úloha se při příštím spuštění "
                              "zařadí znovu do fronty.")
            return bool(ok)
        return True

    try:
        win.events.closing += on_closing
    except Exception:
        pass
    icon = ICON_DIR / ("icon.ico" if os.name == "nt" else "icon.png")
    try:
        webview.start(icon=str(icon) if icon.exists() else None, private_mode=False,
                      storage_path=str(Path(settings.root) / "webview"))
    finally:
        server.should_exit = True
        th.join(timeout=5)


# --------------------------------------------------------------------------
# Shortcuts
# --------------------------------------------------------------------------

def _app_command() -> tuple[str, list[str]]:
    """Executable that starts the desktop app without a console window."""
    scripts = Path(sys.executable).parent
    if os.name == "nt":
        exe = scripts / "synthtrack-app.exe"
        if exe.exists():
            return str(exe), []
        pyw = scripts / "pythonw.exe"
        return str(pyw if pyw.exists() else sys.executable), ["-m", "synthtrack", "app"]
    exe = scripts / "synthtrack-app"
    if exe.exists():
        return str(exe), []
    return sys.executable, ["-m", "synthtrack", "app"]


def create_shortcuts(data_dir: str | None = None) -> list[str]:
    target, args = _app_command()
    if data_dir:
        args = ["--data", data_dir] + args if args[:2] != ["-m", "synthtrack"] else \
            ["-m", "synthtrack", "--data", data_dir, "app"]
    made = []
    if os.name == "nt":
        icon = ICON_DIR / "icon.ico"
        home = Path(os.environ.get("USERPROFILE", str(Path.home())))
        places = [home / "Desktop",
                  Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / "Microsoft" / "Windows" /
                  "Start Menu" / "Programs"]
        # OneDrive-redirected desktops
        od = os.environ.get("OneDrive")
        if od and (Path(od) / "Desktop").exists():
            places[0] = Path(od) / "Desktop"
        for place in places:
            if not place.exists():
                continue
            lnk = place / "SynthTrack.lnk"
            arg_str = " ".join(f'"{a}"' if " " in a else a for a in args)
            ps = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
                  f"$s.TargetPath='{target}';$s.Arguments='{arg_str}';"
                  f"$s.WorkingDirectory='{Path(target).parent}';"
                  f"$s.IconLocation='{icon}';$s.Description='SynthTrack – synthetic tracking';$s.Save()")
            r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                               capture_output=True, text=True, stdin=subprocess.DEVNULL, **NO_WINDOW)
            if r.returncode == 0:
                made.append(str(lnk))
            else:
                print("Zástupce se nepodařilo vytvořit:", r.stderr.strip(), file=sys.stderr)
    else:
        icon = ICON_DIR / "icon.png"
        exec_line = " ".join([f'"{target}"'] + [f'"{a}"' for a in args])
        entry = ("[Desktop Entry]\nType=Application\nName=SynthTrack\nComment=Synthetic tracking of asteroids\n"
                 f"Exec={exec_line}\nIcon={icon}\nTerminal=false\nCategories=Science;Astronomy;\n")
        for place in (Path.home() / ".local" / "share" / "applications", Path.home() / "Desktop"):
            if place.exists() or place.parent.exists():
                place.mkdir(parents=True, exist_ok=True)
                f = place / "synthtrack.desktop"
                f.write_text(entry)
                f.chmod(0o755)
                made.append(str(f))
    return made


def gui_main():
    """Entry point of the console-less ``synthtrack-app`` executable."""
    from .cli import main
    main(sys.argv[1:] + ([] if "app" in sys.argv[1:] else ["app"]))
