"""Disk checkpointing for expensive notebook cells.

Workflow:
    from notebook_utils import cached, clear_cache
    model = cached("stage2", lambda: expensive_fit())

To force a refit of one cell:    cached("stage2", fn, force=True)
To force a refit of everything:  set FORCE_REFIT = True at the top of the notebook
To clear cached artifacts:       clear_cache()  or  clear_cache("stage2")

Cache files live in notebooks/checkpoints/ and persist across kernel restarts.
"""
import pathlib
import joblib

CHECKPOINT_DIR = pathlib.Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

FORCE_REFIT = False


def cached(name, fit_fn, force=None):
    do_force = FORCE_REFIT if force is None else force
    path = CHECKPOINT_DIR / f"{name}.pkl"
    if path.exists() and not do_force:
        print(f"  [cache hit ] {name}  ← {path.name}")
        return joblib.load(path)
    print(f"  [cache miss] {name}  → fitting...")
    obj = fit_fn()
    joblib.dump(obj, path)
    size_mb = path.stat().st_size / 1e6
    print(f"  [saved     ] {name}  → {path.name} ({size_mb:.1f} MB)")
    return obj


def clear_cache(name=None):
    if name is None:
        files = list(CHECKPOINT_DIR.glob("*.pkl"))
        for f in files:
            f.unlink()
        print(f"Cleared {len(files)} checkpoint(s) from {CHECKPOINT_DIR}/")
    else:
        path = CHECKPOINT_DIR / f"{name}.pkl"
        if path.exists():
            path.unlink()
            print(f"Cleared {path.name}")
        else:
            print(f"No cache at {path.name}")


def list_cache():
    files = sorted(CHECKPOINT_DIR.glob("*.pkl"))
    if not files:
        print(f"(empty — {CHECKPOINT_DIR}/)")
        return
    print(f"{CHECKPOINT_DIR}/")
    for f in files:
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:<40} {size_mb:>7.1f} MB")
