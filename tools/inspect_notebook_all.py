#!/usr/bin/env python3
"""Dump all code cells [1..7] of dawah_dialectics_v2_colab.ipynb to reports/notebook_cells_all.txt."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
nb = json.loads((REPO / "dawah_dialectics_v2_colab.ipynb").read_text(encoding="utf-8"))
out = []
for idx, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    out.append(f"=================== CELL [{idx}] ===================")
    out.append(src)
    out.append("")
(REPO / "reports" / "notebook_cells_all.txt").write_text("\n".join(out), encoding="utf-8")
print("wrote reports/notebook_cells_all.txt")
