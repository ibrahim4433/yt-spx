import sqlite3
import json
import urllib.request
import urllib.parse
import time
import io
import sys
import os
import pathlib

# Ensure project root is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ISLAMIC_TERM_FIXES = {
    "التكية": "التقية",
    "التكيه": "التقية",
    "كمessenger": "كرسول",
    "كمرسول": "كرسول لله",
    "خرافه": "رعيته",
}

def clean_arabic_terms(text: str) -> str:
    if not text:
        return ""
    for wrong, right in ISLAMIC_TERM_FIXES.items():
        text = text.replace(wrong, right)
    return text.strip()

def is_none_text(text: str) -> bool:
    if not text:
        return True
    return text.strip().lower() in ["none", "none used", "none used.", "none.", "n/a", "no", "not applicable", "null"]

def translate_single(text: str, max_retries: int = 2) -> str:
    if not text or is_none_text(text):
        return ""
    url = 'https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=ar&dt=t&q=' + urllib.parse.quote(text)
    for _ in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                res = ''.join([part[0] for part in data[0] if part and len(part) > 0]).strip()
                return clean_arabic_terms(res)
        except Exception:
            time.sleep(0.3)
    return ""

def batch_translate_to_ar(texts, max_retries: int = 2):
    """Translates a list of strings using a numeric token delimiter that Google never translates."""
    if not texts:
        return []
    
    clean_texts = [str(t).strip() if t else "" for t in texts]
    if not any(clean_texts):
        return ["" for _ in texts]

    delimiter = "\n[[[998877]]]\n"
    combined = delimiter.join(clean_texts)
    
    url = 'https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=ar&dt=t&q=' + urllib.parse.quote(combined)
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode())
                translated_combined = ''.join([part[0] for part in data[0] if part and len(part) > 0])
                results = translated_combined.split("[[[998877]]]")
                
                # Check for exact match
                if len(results) == len(texts):
                    return [clean_arabic_terms(r) for r in results]
                # If delimiter failed, fallback to translating individually
                break
        except Exception:
            time.sleep(0.4)
            
    # Fallback to 1-by-1 translation if batch delimiter failed
    fallback_results = []
    for t in texts:
        fallback_results.append(translate_single(t))
    return fallback_results

def run_enrichment(db_path=None):
    if db_path is None:
        from db.database import DB_PATH
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find rows contaminated with broken delimiters or missing Arabic fields
    rows = conn.execute("""
        SELECT * FROM qa_units 
        WHERE logical_premises_ar_json LIKE '%تقسيم%'
           OR logical_premises_ar_json LIKE '%SPLIT%'
           OR logical_premises_ar_json LIKE '%""%'
           OR analogy_or_metaphor_ar LIKE '%تقسيم%'
           OR analogy_or_metaphor_ar LIKE '%SPLIT%'
           OR socratic_question_ar LIKE '%تقسيم%'
           OR socratic_question_ar LIKE '%SPLIT%'
           OR logical_premises_ar_json = '[]'
           OR logical_premises_ar_json IS NULL
    """).fetchall()

    if not rows:
        print("   ✅ All records in database are clean and 100% Arabic!")
        conn.close()
        return

    print(f"\n   🌐 Found {len(rows)} records needing repair and clean Arabic translation...")
    
    try:
        from tqdm.auto import tqdm
        pbar = tqdm(rows, desc="   ⚡ Clean Arabic Repair")
    except ImportError:
        pbar = rows

    updated_count = 0
    try:
        for r in pbar:
            row_id = r["id"]

            premises_en = json.loads(r["logical_premises_json"] or "[]")
            analogy_en = (r["analogy_or_metaphor"] or "").strip()
            socratic_en = (r["socratic_question"] or "").strip()
            quote_en = (r["verbatim_quote"] or "").strip()

            # Clean "none" values in english
            if is_none_text(analogy_en):
                analogy_en = ""
            if is_none_text(socratic_en):
                socratic_en = ""

            # Prepare items to translate cleanly from the authentic English text
            need_trans = []
            keys = []

            for idx, p in enumerate(premises_en):
                if p and p.strip():
                    need_trans.append(p.strip())
                    keys.append(("premise", idx))

            if analogy_en:
                need_trans.append(analogy_en)
                keys.append(("analogy", 0))

            if socratic_en:
                need_trans.append(socratic_en)
                keys.append(("socratic", 0))

            if quote_en:
                need_trans.append(quote_en[:300])
                keys.append(("quote", 0))

            premises_ar = []
            analogy_ar = ""
            socratic_ar = ""
            quote_ar = ""

            if need_trans:
                translated_parts = batch_translate_to_ar(need_trans)
                premise_map = {}
                for (k_type, k_idx), trans_val in zip(keys, translated_parts):
                    if k_type == "premise":
                        premise_map[k_idx] = trans_val
                    elif k_type == "analogy":
                        analogy_ar = trans_val
                    elif k_type == "socratic":
                        socratic_ar = trans_val
                    elif k_type == "quote":
                        quote_ar = trans_val

                premises_ar = [premise_map.get(i, "").strip() for i in range(len(premises_en))]
                # Filter out any accidental empty premise strings
                premises_ar = [p for p in premises_ar if p]

            with conn:
                conn.execute("""
                    UPDATE qa_units SET
                        logical_premises_json = ?,
                        logical_premises_ar_json = ?,
                        analogy_or_metaphor = ?,
                        analogy_or_metaphor_ar = ?,
                        socratic_question = ?,
                        socratic_question_ar = ?,
                        verbatim_quote = ?,
                        verbatim_quote_ar = ?
                    WHERE id = ?
                """, (
                    json.dumps(premises_en, ensure_ascii=False),
                    json.dumps(premises_ar, ensure_ascii=False),
                    analogy_en,
                    analogy_ar,
                    socratic_en,
                    socratic_ar,
                    quote_en,
                    quote_ar,
                    row_id
                ))
            updated_count += 1
    except KeyboardInterrupt:
        print("\n   [!] Repair paused by user. Progress safely saved.")
    finally:
        print(f"   🎉 Cleaned and re-translated {updated_count} records cleanly!")
        conn.close()

if __name__ == "__main__":
    run_enrichment()
