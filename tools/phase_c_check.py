#!/usr/bin/env python3
"""Phase C: attempt Stage-2 regeneration; verify website endpoints; write reports/phase_c_report.md.

ASCII-safe logging for Windows cp1252 consoles.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

lines = []


def safe(s):
    return s.encode("utf-8", "backslashreplace").decode("utf-8", "replace")


def log(s=""):
    lines.append(s)
    print(safe(s))


log("# Phase C Check - EXECUTION REPORT")
log("")

log("## C1. Stage-2 Re-extraction Attempt (tools/rerun_stage2_for_cards.py)")
log("")
r = subprocess.run(
    [sys.executable, str(REPO / "tools" / "rerun_stage2_for_cards.py")],
    capture_output=True, text=True, timeout=600, cwd=REPO,
)
if r.returncode == 0:
    log(f"RE-EXTRACTION RAN OK: {safe(r.stdout[-2000:])}")
else:
    err_lines = [l for l in (r.stderr or "").splitlines() if l.strip()]
    log("RE-EXTRACTION BLOCKED HERE (needs GPU/torch - run on Colab):")
    for l in err_lines[-6:]:
        log(f"  > {safe(l)}")
    regen_dir = REPO / "reports" / "regen"
    if regen_dir.is_dir():
        files = sorted(regen_dir.glob("*.qa.json"))
        log(f"Existing regenerated shards in reports/regen/: {len(files)}")
log("")

log("## C2. Website Endpoint Sanity (web/app.py via FastAPI TestClient)")
try:
    from fastapi.testclient import TestClient
    import web.app as web_app

    client = TestClient(web_app.app)
    checks = []

    r = client.get("/api/stats")
    checks.append((safe(f"GET /api/stats -> status={r.status_code}"), r.status_code == 200, ""))

    r = client.get("/api/videos/0Y7cFAUlil4/scenario")
    ok = r.status_code == 200 and len(r.json().get("speakers", [])) >= 2 and len(r.json().get("turns", [])) >= 100
    checks.append((safe(f"GET /api/videos/0Y7cFAUlil4/scenario -> status={r.status_code} speakers+turns populated"), ok, ""))

    r = client.get("/api/search", params={"q": "spirit", "limit": 10})
    checks.append((safe(f"GET /api/search?q=spirit -> total={r.json().get('total', 0) if r.status_code == 200 else 'n/a'}"), r.status_code == 200 and r.json().get("total", 0) >= 1, ""))

    r = client.get("/")
    checks.append((safe(f"GET / -> status={r.status_code}"), r.status_code == 200, ""))

    r = client.get("/api/videos", params={"limit": 20})
    total = r.json().get("total", 0) if r.status_code == 200 else 0
    checks.append((safe(f"GET /api/videos -> status={r.status_code} total={total}"), r.status_code == 200 and total >= 11, ""))

    for name, ok, detail in checks:
        log(f"  [{'PASS' if ok else 'FAIL'}] {name} {safe(detail)}")
except Exception as e:
    log(f"  [FAIL] Website sanity exception: {safe(str(e))}")
log("")

log("## C3. DB Clone Protection Check")
import hashlib
db_path = REPO / "data" / "dawah_knowledge.db"
bak_path = REPO / "data" / "dawah_knowledge.db.bak"
h = hashlib.md5(db_path.read_bytes()).hexdigest() if db_path.exists() else "N/A"
log(f"  data/dawah_knowledge.db md5: {h}")
log(f"  data/dawah_knowledge.db.bak exists: {bak_path.exists()}")
log("")

(REPO / "reports" / "phase_c_report.md").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[+] Report saved to {REPO / 'reports' / 'phase_c_report.md'}")
