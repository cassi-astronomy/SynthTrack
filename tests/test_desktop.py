"""Desktop wrapper: importable without pywebview, shortcut on Linux, CLI wiring."""
from pathlib import Path

from synthtrack import desktop


def test_free_port_and_command():
    assert 1024 < desktop._free_port() < 65536
    exe, args = desktop._app_command()
    assert exe
    assert args in ([], ["-m", "synthtrack", "app"])


def test_linux_shortcut(tmp_path, monkeypatch):
    if desktop.os.name == "nt":
        return
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".local" / "share").mkdir(parents=True)
    made = desktop.create_shortcuts(str(tmp_path / "data"))
    assert made
    txt = Path(made[0]).read_text()
    assert "Name=SynthTrack" in txt and "--data" in txt and "Terminal=false" in txt


def test_icons_packaged():
    assert (desktop.ICON_DIR / "icon.png").exists()
    assert (desktop.ICON_DIR / "icon.ico").exists()
