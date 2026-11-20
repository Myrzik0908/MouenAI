import ctypes
import os
import sys
from pathlib import Path

NAME = "MouenAI"
def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "CMakeLists.txt").exists():
            return p
    raise FileNotFoundError(f"CMakeLists.txt not found above {start}")

ROOT = _find_project_root(Path(__file__).resolve().parent)

NAMES = {
    "win32":  [f"{NAME}.dll"],
    "darwin": [f"{NAME}.dylib", f"lib{NAME}.dylib"],
}.get(sys.platform, [f"{NAME}.so", f"lib{NAME}.so"])

_lib = None
_auto_choise = True

def SelectiveLoading():
    global _auto_choise
    _auto_choise = False

def _priority(p: Path) -> int:
    s = str(p).lower()
    if "release" in s: return 0
    if "debug" in s: return 2
    if "inference" in s: return 3
    return 1

def _find() -> Path:
    if env := os.environ.get("MYLIB_PATH"):
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(f"MYLIB_PATH={env} not found")
        return p

    hits = [p for n in NAMES for p in ROOT.rglob(n)]
    if not hits:
        raise FileNotFoundError(f"{NAME} not found in {ROOT} (looking for {NAMES})")
    return sorted(hits, key=_priority)


def get_lib() -> ctypes.CDLL:
    """Returns the loaded library"""
    global _lib
    if _lib is not None:
        return _lib

    if _auto_choise:
        print("Note: You can manually select a library build by invoking SelectiveLoading()")
        path = _find()[0]
    else:
        solutions = _find()
        for i, x in enumerate(solutions):
            print(f"{i}: '{x}'")
        choice = int(input("select the desired build (specify the build index): "))
        if choice < 0 or choice > len(solutions) - 1:
            raise RuntimeError("Invalid index")
        path = solutions[choice]

    # Windows: allow searching for dependent DLLs in the same folder
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        global _dll_cookie
        _dll_cookie = os.add_dll_directory(str(path.parent))

    _lib = ctypes.CDLL(str(path))
    print(f"[{NAME}] loaded: {path}")
    return _lib