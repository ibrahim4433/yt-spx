import sys
import io
import time
import json
import threading
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Ensure utf-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import uvicorn
from web.app import app

def run():
    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8005, log_level="error")

    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(2)

    # 1. Test /api/stats
    with urllib.request.urlopen("http://127.0.0.1:8005/api/stats") as resp:
        stats = json.loads(resp.read().decode())
        print("[+] Stats endpoint:", stats)
        assert stats["total_videos"] > 0
        assert stats["total_qa"] > 0

    # 2. Test /api/search with English query
    with urllib.request.urlopen("http://127.0.0.1:8005/api/search?q=cause") as resp:
        res = json.loads(resp.read().decode())
        print(f"[+] Search 'cause' returned {res['count']} results.")
        assert res["count"] > 0

    # 3. Test /api/chat with Arabic query "من خلق الله"
    url = "http://127.0.0.1:8005/api/chat?q=%D9%85%D9%86+%D8%AE%D9%84%D9%82+%D8%A7%D9%84%D9%84%D9%87"
    with urllib.request.urlopen(url) as resp:
        chat_res = json.loads(resp.read().decode())
        pkg = chat_res.get("response_package")
        print("[+] Chat bot matched:", pkg["title"])
        print("[+] Timestamp link:", pkg["timestamp_url"])
        assert len(pkg["title"]) > 0
        assert "youtube.com" in pkg["timestamp_url"]

    # 4. Test HTML Index page
    with urllib.request.urlopen("http://127.0.0.1:8005/") as resp:
        html = resp.read().decode()
        assert "Dawah Dialectics" in html
        print("[+] HTML index page served successfully (Length:", len(html), "bytes)")

    print("[SUCCESS] All endpoints verified cleanly!")

if __name__ == "__main__":
    run()
