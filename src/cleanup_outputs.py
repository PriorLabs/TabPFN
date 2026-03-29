#!/usr/bin/env python3
"""
Cleanup script for outputs directory.

Removes:
- Old timestamped model_comparison_*.csv and model_comparison_*.pkl files (keeps 3 most recent)
- All proba_debug_*.csv files (debug artifacts, no longer needed)
"""

from pathlib import Path
from datetime import datetime
import re
import os

# Output directory:
# - If the OUTPUT_DIR environment variable is set, use that.
# - Otherwise, default to an "outputs" directory next to this script.
OUT_DIR = Path(os.environ.get("OUTPUT_DIR", Path(__file__).resolve().parent / "outputs"))

def get_timestamp(filename):
    """Extract timestamp from filename (format: YYYYMMDD_HHMMSS)"""
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        ts_str = match.group(1)
        try:
            return datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
        except ValueError:
            return None
    return None

def main():
    if not OUT_DIR.exists():
        print(f"Output directory not found: {OUT_DIR}")
        return
    
    print(f"Cleaning up {OUT_DIR}...\n")
    
    # 1. Find and remove debug artifacts
    debug_files = list(OUT_DIR.glob("proba_debug_*.csv"))
    if debug_files:
        print(f"🗑️  Removing {len(debug_files)} debug artifact(s):")
        for f in debug_files:
            print(f"   - {f.name}")
            f.unlink()
    else:
        print("✅ No debug artifacts found.")
    
    # 2. Keep only 3 most recent model_comparison files
    model_csvs = list(OUT_DIR.glob("model_comparison_*.csv"))
    model_pkls = list(OUT_DIR.glob("model_comparison_*.pkl"))
    
    # Sort by timestamp
    model_csvs_sorted = sorted(model_csvs, key=lambda f: get_timestamp(f.name) or datetime.min, reverse=True)
    model_pkls_sorted = sorted(model_pkls, key=lambda f: get_timestamp(f.name) or datetime.min, reverse=True)
    
    # Remove all but 3 most recent of each type
    to_remove_csv = model_csvs_sorted[3:]
    to_remove_pkl = model_pkls_sorted[3:]
    
    if to_remove_csv or to_remove_pkl:
        print(f"\n🗑️  Keeping 3 most recent model_comparison files, removing {len(to_remove_csv) + len(to_remove_pkl)} old file(s):")
        for f in to_remove_csv + to_remove_pkl:
            print(f"   - {f.name}")
            f.unlink()
    else:
        print("\n✅ All model_comparison files are recent (≤3 files).")
    
    # 3. Summary
    remaining_files = list(OUT_DIR.glob("*"))
    print(f"\n📊 Outputs directory status:")
    print(f"   Total files: {len(remaining_files)}")
    print(f"   Model comparison CSVs: {len(list(OUT_DIR.glob('model_comparison_*.csv')))}")
    print(f"   Model comparison PKLs: {len(list(OUT_DIR.glob('model_comparison_*.pkl')))}")
    print(f"   Debug artifacts: {len(list(OUT_DIR.glob('proba_debug_*.csv')))}")
    print(f"\n✨ Cleanup complete!")

if __name__ == '__main__':
    main()
