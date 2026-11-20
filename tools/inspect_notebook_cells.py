#!/usr/bin/env python3
"""List Colab notebook cells (index, type, first line) for dawah_dialectics_v2_colab.ipynb."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
nb = json.loads((REPO / "dawah_dialectics_v2_colab.ipynb").read_text(encoding="utf-8"))

lines = [f"notebook: {REPO / 'dawah_dialectics_v2_colab.ipynb'}", f"cells: {len(nb['cells'])}", ""]
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    first = src.strip().splitlines()[0] if src.strip() else "(empty)"
    lines.append(f"[{i}] {c['cell_type']}: {first[:110]}")

(REPO / "reports" / "notebook_cells.txt").parent.mkdir(parents=True, exist_ok=True)
(REPO / "reports" / "notebook_cells.txt").write_text("\n".join(lines), encoding="utf-8")
print("wrote reports/notebook_cells.txt")
