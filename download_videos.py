#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from pathlib import Path

# ==============================================================================
# 1. إعداد المسارات المحلية
# ==============================================================================
# المجلد الحالي الذي يُشغل منه السكربت
BASE_DIR = Path(__file__).resolve().parent

# مجلد حفظ الفيديوهات المحملة
DOWNLOAD_DIR = BASE_DIR / "Downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ملف الأرشيف لحفظ سجل الفيديوهات التي اكتمل تنزيلها
ARCHIVE_FILE = DOWNLOAD_DIR / "archive.txt"

# ==============================================================================
# 2. البحث عن ملف المصدر واستخراج الروابط النقية
# ==============================================================================
print("[1/3] البحث عن ملف الروابط وتنقيته...")

source_file = None
possible_names = ["remaining_urls.txt", "urls.txt", "links.txt", "urls.csv"]

# فحص الأسماء الشائعة في المجلد الحالي
for name in possible_names:
    candidate = BASE_DIR / name
    if candidate.is_file():
        source_file = candidate
        break

# إذا لم يُعثر عليه بالأسماء المحددة، فحص أول ملف txt أو csv في المجلد
if not source_file:
    for f in BASE_DIR.glob("*"):
        if f.is_file() and f.suffix.lower() in [".txt", ".csv"] and f.name != "clean_urls.txt":
            source_file = f
            break

if not source_file:
    print("[-] خطأ: لم يتم العثور على أي ملف نصي أو CSV يحتوي على الروابط.")
    print(f"    يرجى وضع ملف الروابط داخل المجلد: {BASE_DIR}")
    sys.exit(1)

print(f"  -> تم العثور على ملف المصدر: {source_file.name}")

with open(source_file, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# استخراج الروابط العادية و Shorts بدقة
raw_urls = re.findall(
    r"https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[a-zA-Z0-9_-]+",
    content,
)

# إزالة التكرار مع الحفاظ على الترتيب
clean_urls = []
seen = set()
for url in raw_urls:
    if url not in seen:
        seen.add(url)
        clean_urls.append(url)

if not clean_urls:
    print(f"[-] خطأ: لم يتم العثور على أي روابط يوتيوب صالحة داخل {source_file.name}")
    sys.exit(1)

clean_batch_file = BASE_DIR / "clean_urls.txt"
with open(clean_batch_file, "w", encoding="utf-8") as f:
    for url in clean_urls:
        f.write(f"{url}\n")

print(f"  -> تم استخراج {len(clean_urls)} رابط صالح.")
print(f"  -> حُفظت القائمة في: {clean_batch_file.name}")

# ==============================================================================
# 3. تنظيف ملفات الترجمة غير المرغوبة (إن وُجدت)
# ==============================================================================
print("\n[2/3] فحص ملفات الترجمة...")
ALLOWED_LANGUAGES = {"ja", "jpn", "ar", "ara", "en", "eng"}
SUB_EXTENSIONS = {".srt", ".vtt", ".ass", ".sub", ".smi"}


def clean_unwanted_subtitles(directory: Path, allowed_langs: set) -> int:
    removed = 0
    for f in directory.rglob("*"):
        if f.is_file() and f.suffix.lower() in SUB_EXTENSIONS:
            parts = f.name.split(".")
            if len(parts) >= 3:
                lang = parts[-2].lower()
                if len(lang) in (2, 3) and lang.isalpha():
                    if lang not in allowed_langs:
                        try:
                            f.unlink()
                            removed += 1
                        except OSError:
                            pass
    return removed


deleted_subs = clean_unwanted_subtitles(DOWNLOAD_DIR, ALLOWED_LANGUAGES)
print(f"  -> تم حذف {deleted_subs} ملف ترجمة للغات غير مطلوبة.")

# ==============================================================================
# 4. تشغيل عملية التنزيل عبر yt-dlp
# ==============================================================================
print("\n[3/3] بدء تنزيل الفيديوهات...")

output_template = str(DOWNLOAD_DIR / "%(title)s.%(ext)s")

ytdlp_cmd = [
    "yt-dlp",
    "--skip-download",           
    "--write-sub",               
    "--write-auto-sub",          
    "--sub-lang", "en,en-US",    
    "--sub-format", "srt/vtt",   
    "--continue",
    "--no-overwrites",
    "--download-archive", str(ARCHIVE_FILE),
    "--force-write-archive",     
    "--batch-file", str(clean_batch_file),
    "-o", output_template,
    "--retries", "10",
    "--ignore-errors",           
    
    # الكوكيز والانتحال ومحرك جافاسكربت
    "--cookies", "cookies.txt", 
    "--impersonate", "chrome", 
    "--js-runtimes", "node",
    
    # ------------------ تعديلات الأمان لتجنب الحظر ------------------
    # 1. زيادة التأخير الزمني (ينتظر بين 5 إلى 15 ثانية بشكل عشوائي بين كل فيديو)
    "--sleep-interval", "5",     
    "--max-sleep-interval", "15", 
    
    # 2. تحديد عدد الطلبات المسموح بها لتجنب الإرهاق
    "--sleep-requests", "1", # تأخير بسيط بين الطلبات الداخلية لنفس الفيديو
]

try:
    process = subprocess.Popen(
        ytdlp_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end="")

    process.stdout.close()
    exit_code = process.wait()
    print(f"\n[+] اكتمل التشغيل بكود خروج: {exit_code}")

except KeyboardInterrupt:
    print("\n[!] تم إيقاف التنزيل يدوياً. يمكنك إعادة تشغيل السكربت للاستكمال من حيث توقف.")
    sys.exit(0)
