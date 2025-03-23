"""Generate requirements.txt with maximum allowed dependency versions."""

from __future__ import annotations

import re
from pathlib import Path


def main() -> None:
    """Extract maximum dependency versions and write to requirements.txt."""
    with Path("pyproject.toml").open() as f:
        content = f.read()

    # Find dependencies section using regex
    deps_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if deps_match:
        deps = [
            d.strip(' "\'')
            for d in deps_match.group(1).strip().split("\n")
            if d.strip()
        ]
        max_reqs = []
        for dep in deps:
            # Check for maximum version constraint
            max_version_match = re.search(r'([^>=<\s]+).*?<\s*([^,\s"\']+)', dep)
            if max_version_match:
                # If there's a max version, use the version just below it
                package, max_ver = max_version_match.groups()
                max_reqs.append(f"{package}<{max_ver}")
            else:
                # If no max version, just use the package name
                package = re.match(r"([^>=<\s]+)", dep).group(1)
                max_reqs.append(package)

        with Path("requirements.txt").open("w") as f:
            f.write("\n".join(max_reqs))


if __name__ == "__main__":
    main()