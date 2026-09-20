"""Build syncguard_source.zip -- the source archive handed to judges/reviewers.

Design: the file list comes from `git ls-files`, so the archive contains **exactly the
tracked working tree and nothing else**. Untracked and gitignored files cannot get in by
construction -- that is the primary guarantee, not the deny-list below.

The deny-list is belt-and-braces for three failure modes git alone would not catch:
  1. a secret that was force-added (`git add -f .env`) at some point and is now tracked,
  2. a large runtime artifact (a SQLite DB) that someone tracked by mistake,
  3. externally-sourced third-party data that should not be redistributed.

Anything the deny-list catches is a real problem, so this script **fails loudly** rather
than silently dropping it -- a silent drop would hide a tracked secret instead of surfacing
it. Run with --force to package anyway (the offending paths are still excluded, and the
exclusion is printed).

Usage:
    python make_source_zip.py                 # build syncguard_source.zip
    python make_source_zip.py --verify        # check an existing zip against the repo
    python make_source_zip.py --out other.zip
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = REPO_ROOT / "syncguard_source.zip"

# Glob patterns matched against each repo-relative POSIX path. A match is an error, not a
# quiet skip -- see the module docstring.
DENY_PATTERNS = [
    ".env",              # the local secrets file (OpenCelliD key, etc.)
    ".env.*",            # .env.local, .env.production, ...
    "**/.env",
    "**/.env.*",
    "*.db",              # SQLite runtime DBs
    "*.db-wal",
    "*.db-shm",
    "*.db-journal",
    "**/*.db",
    "**/*.db-wal",
    "**/*.db-shm",
    "**/*.db-journal",
    "data/external/**",  # third-party data, not ours to redistribute
    "**/*.pem",
    "**/*.key",
    "**/id_rsa*",
]

# Exempt from DENY_PATTERNS: a committed template that deliberately carries no real values.
ALLOW_EXACT = {".env.example"}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split("\0") if p]


def denied(path: str) -> str | None:
    """Return the matching deny pattern, or None."""
    if path in ALLOW_EXACT:
        return None
    for pat in DENY_PATTERNS:
        if fnmatch.fnmatch(path, pat):
            return pat
    return None


def build(out_path: Path, force: bool) -> int:
    files = tracked_files()
    if not files:
        print("git ls-files returned nothing -- is this a git repository?", file=sys.stderr)
        return 1

    violations = [(f, pat) for f in files if (pat := denied(f))]
    if violations:
        print(f"\nBLOCKED: {len(violations)} tracked file(s) match an exclusion rule.\n"
              f"These are TRACKED IN GIT, which means they are already in the repository's\n"
              f"history and excluding them from the zip does not undo that:\n", file=sys.stderr)
        for f, pat in violations:
            print(f"  {f}   (matched {pat!r})", file=sys.stderr)
        if not force:
            print(f"\nInvestigate before packaging. Re-run with --force to build the archive\n"
                  f"without them once you have.", file=sys.stderr)
            return 2
        print("\n--force given: excluding them from the archive and continuing.\n",
              file=sys.stderr)

    excluded = {f for f, _ in violations}
    included = [f for f in files if f not in excluded]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in included:
            src = REPO_ROOT / f
            if not src.exists():
                # Tracked but missing from the working tree (deleted, not yet committed).
                print(f"  skip (missing from working tree): {f}", file=sys.stderr)
                continue
            z.write(src, arcname=f)

    size_mb = out_path.stat().st_size / 1_000_000
    print(f"Wrote {out_path.name}: {len(included)} files, {size_mb:.2f} MB")
    print(f"Source: `git ls-files` (tracked only) minus {len(excluded)} excluded path(s).")
    return 0


def verify(out_path: Path) -> int:
    """Check an existing archive against the tracked working tree."""
    if not out_path.exists():
        print(f"{out_path} does not exist.", file=sys.stderr)
        return 1

    with zipfile.ZipFile(out_path) as z:
        in_zip = {n for n in z.namelist() if not n.endswith("/")}

    expected = {f for f in tracked_files() if not denied(f) and (REPO_ROOT / f).exists()}

    leaked = sorted(n for n in in_zip if denied(n))
    missing = sorted(expected - in_zip)
    extra = sorted(in_zip - expected - set(leaked))

    print(f"{out_path.name}: {len(in_zip)} entries")
    ok = True
    if leaked:
        ok = False
        print(f"\nFAIL -- {len(leaked)} excluded path(s) present in the archive:", file=sys.stderr)
        for n in leaked:
            print(f"  {n}", file=sys.stderr)
    if missing:
        ok = False
        print(f"\nFAIL -- {len(missing)} tracked file(s) missing from the archive:", file=sys.stderr)
        for n in missing[:40]:
            print(f"  {n}", file=sys.stderr)
        if len(missing) > 40:
            print(f"  ... and {len(missing) - 40} more", file=sys.stderr)
    if extra:
        ok = False
        print(f"\nFAIL -- {len(extra)} untracked file(s) present in the archive:", file=sys.stderr)
        for n in extra[:40]:
            print(f"  {n}", file=sys.stderr)
        if len(extra) > 40:
            print(f"  ... and {len(extra) - 40} more", file=sys.stderr)

    if ok:
        print("OK -- archive matches the tracked working tree, no excluded paths present.")
        return 0
    return 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true",
                    help="verify an existing archive instead of building one")
    ap.add_argument("--force", action="store_true",
                    help="build even if tracked files match an exclusion rule")
    args = ap.parse_args()
    return verify(args.out) if args.verify else build(args.out, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
