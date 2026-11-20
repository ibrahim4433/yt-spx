#!/usr/bin/env python3
"""Dump code of Colab notebook cells 1, 4, 5 to verify what they do."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
nb = json.loads((REPO / "dawah_dialectics_v2_colab.ipynb").read_text(encoding="utf-8"))
out = []
for idx in (1, 4, 5):
    if idx >= len(nb["cells"]):
        continue
    src = "".join(nb["cells"][idx].get("source", []))
    out.append(f"=================== CELL [{idx}] ===================")
    out.append(src[:3500])
    out.append("")
(REPO / "reports" / "notebook_cells_145.txt").write_text("\n".join(out), encoding="utf-8")
print("wrote reports/notebook_cells_145.txt")
