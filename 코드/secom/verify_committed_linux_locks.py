"""Verify regenerated Linux locks are identical to committed evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dependency_lock import validate_linux_dependency_lock_manifest
from environment_provenance import sha256_text_file


PROJECT = Path(__file__).resolve().parents[2]
COMMITTED_MANIFEST = (
    PROJECT / "결과물" / "secom" / "linux_dependency_lock" / "linux_lock_manifest.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, required=True)
    args = parser.parse_args()
    generated_dir = args.generated_dir.resolve()
    committed = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    validate_linux_dependency_lock_manifest(committed, PROJECT)
    pairs = (
        (PROJECT / "pylock.github-actions.toml", generated_dir / "pylock.github-actions.toml"),
        (PROJECT / "pylock.streamlit.toml", generated_dir / "pylock.streamlit.toml"),
        (COMMITTED_MANIFEST, generated_dir / "linux_lock_manifest.json"),
    )
    changed = [
        committed_path.name
        for committed_path, generated_path in pairs
        if not generated_path.is_file()
        or sha256_text_file(committed_path) != sha256_text_file(generated_path)
    ]
    if changed:
        raise RuntimeError(
            "Linux 의존성 해석이 커밋된 잠금과 달라졌습니다. 검토 후 잠금을 갱신하세요: "
            + ", ".join(changed)
        )
    print("커밋된 Linux 잠금과 현재 Ubuntu 재생성 결과가 모두 일치합니다.")


if __name__ == "__main__":
    main()
