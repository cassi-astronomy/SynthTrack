"""SynthTrack – GPU synthetic tracking pipeline for asteroids and comets."""

__version__ = "0.1.0"


def _configure_iers() -> None:
    """Offline-first Earth-orientation data for astropy.

    Observer positions (EarthLocation.get_gcrs_posvel) need UT1-UTC and polar
    motion. By default astropy downloads IERS-A and *raises* when its table is
    older than 30 days and the download fails. For SynthTrack the effect of stale
    or extrapolated values is negligible (0.1 s of UT1 = ~45 m of observer
    position = far below 0.01" even for close NEAs), so use the bundled tables,
    never download behind the user's back, and only warn on extrapolation."""
    try:
        from astropy.utils import iers
        iers.conf.auto_download = False
        iers.conf.auto_max_age = None
        if hasattr(iers.conf, "iers_degraded_accuracy"):
            iers.conf.iers_degraded_accuracy = "warn"
    except Exception:
        pass


_configure_iers()
