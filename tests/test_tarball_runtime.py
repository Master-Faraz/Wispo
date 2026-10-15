from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REQUIREMENTS = ROOT / "packaging" / "tarball" / "requirements-runtime.txt"


def test_offline_tarball_requirements_are_pinned_to_uv_lock():
    pinned = {}
    for line in RUNTIME_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", maxsplit=1)
            pinned[name] = version

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = {package["name"]: package["version"] for package in lock["package"]}

    assert pinned == {name: locked[name] for name in pinned}
