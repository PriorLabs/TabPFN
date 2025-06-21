"""Generate a requirements.txt file from pyproject.toml dependencies.

This script can operate in two modes:
1. 'min': Extracts minimum versions (>=) and pins them with '=='.
2. 'max': Extracts maximum versions (<) or leaves them unpinned.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_dependency_lines(content: str) -> list[str]:
    """Finds and cleans the dependency lines from the pyproject.toml content."""
    deps_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not deps_match:
        return []

    deps_lines = deps_match.group(1).strip().split("\n")

    cleaned_deps = []
    for line in deps_lines:
        # Assign the stripped line to a new variable to avoid the linter warning.
        stripped_line = line.strip()
        # Skip empty lines or comments
        if not stripped_line or stripped_line.startswith("#"):
            continue
        # Clean the line by removing an optional trailing comma, then stripping quotes.
        clean_dep = stripped_line.rstrip(",").strip("'\"")
        cleaned_deps.append(clean_dep)

    return cleaned_deps


def main() -> None:
    """Main function to parse arguments and generate the requirements file."""
    parser = argparse.ArgumentParser(
        description="Generate requirements.txt from pyproject.toml.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["minimum", "maximum"],
        help="The type of requirements to generate:\n"
        "'minimum' - for minimum versions (e.g., 'package==1.2.3')\n"
        "'maximum' - for maximum/unpinned versions (e.g., 'package<2.0' or 'package')",
    )
    args = parser.parse_args()

    try:
        content = Path("pyproject.toml").read_text()
    except FileNotFoundError:
        return

    # 1. Shared parsing logic
    deps = parse_dependency_lines(content)
    output_reqs = []

    # 2. Mode-specific processing logic
    if args.mode == "maximum":
        for dep in deps:
            # Check for maximum version constraint
            max_version_match = re.search(r'([^>=<\s]+).*?<\s*([^,\s"\']+)', dep)
            if max_version_match:
                package, max_ver = max_version_match.groups()
                output_reqs.append(f"{package}<{max_ver}")
            else:
                # If no max version, just use the package name
                package_match = re.match(r"([^>=<\s]+)", dep)
                if package_match:
                    output_reqs.append(package_match.group(1))

    elif args.mode == "minimum":
        for dep in deps:
            # Check for minimum version constraint
            match = re.match(r'([^>=<\s]+)\s*>=\s*([^,\s"\']+)', dep)
            if match:
                package, min_ver = match.groups()
                output_reqs.append(f"{package}=={min_ver}")

    # 3. Shared writing logic
    output_filename = "requirements.txt"
    with Path(output_filename).open("w") as f:
        f.write("\n".join(output_reqs))


if __name__ == "__main__":
    main()
