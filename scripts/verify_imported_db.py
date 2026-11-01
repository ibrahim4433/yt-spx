import sqlite3
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

conn = sqlite3.connect('data/dawah_knowledge.db')
c = conn.cursor()

c.execute("SELECT count(1) FROM qa_units")
total = c.fetchone()[0]
print(f"📊 Total QA units in database: {total}")

c.execute("SELECT count(distinct video_id) FROM qa_units")
videos = c.fetchone()[0]
print(f"🎬 Distinct YouTube videos covered: {videos}")

# Check dummy URLs
c.execute("SELECT id, video_id, timestamp_url FROM qa_units WHERE timestamp_url LIKE '%dQw4w9WgXcQ%' OR video_id LIKE 'pilot_%'")
dummies = c.fetchall()
print(f"⚠️  Dummy link records: {len(dummies)}")
for d in dummies:
    print("  ", d)


# Check real extracted units
c.execute("""
    SELECT count(1) FROM qa_units 
    WHERE id NOT IN ('theology_001', 'theology_002', 'evil_001', 'morality_001', 
                     'science_001', 'preservation_001', 'free_will_001', 'prophethood_001')
      AND timestamp_url NOT LIKE '%dQw4w9WgXcQ%'
""")
real_count = c.fetchone()[0]
print(f"✅ Real extracted dialogue units from videos: {real_count}")

# Check Arabic enrichment completeness
c.execute("""
    SELECT count(1) FROM qa_units 
    WHERE (logical_premises_ar_json IS NULL OR logical_premises_ar_json = '' OR logical_premises_ar_json = '[]')
       OR (analogy_or_metaphor_ar IS NULL OR analogy_or_metaphor_ar = '' OR analogy_or_metaphor_ar LIKE 'None%')
""")
missing_ar = c.fetchone()[0]
print(f"🔍 Units missing Arabic premises/analogies: {missing_ar} / {total}")

# Show sample of real extracted unit
c.execute("""
    SELECT id, video_title_ar, question_ar, core_answer_summary_ar, 
           logical_premises_ar_json, analogy_or_metaphor_ar, socratic_question_ar, timestamp_url
    FROM qa_units 
    WHERE video_id NOT LIKE 'pilot_%' AND timestamp_url NOT LIKE '%dQw4w9WgXcQ%'
    LIMIT 2
""")
samples = c.fetchall()
print("\n" + "="*60)
print("SAMPLE REAL EXTRACTED UNITS:")
print("="*60)
for s in samples:
    print(f"🆔 ID: {s[0]}")
    print(f"🎬 Video: {s[1]}")
    print(f"❓ Question: {s[2]}")
    print(f"💡 Core Answer: {s[3]}")
    print(f"🧠 Premises: {s[4]}")
    print(f"🌟 Analogy: {s[5]}")
    print(f"❓ Socratic: {s[6]}")
    print(f"🔗 Video URL: {s[7]}")
    print("-"*60)

conn.close()
