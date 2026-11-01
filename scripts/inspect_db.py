import sqlite3
import json
import io
import sys

# Ensure utf-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

conn = sqlite3.connect('data/dawah_knowledge.db')
conn.row_factory = sqlite3.Row

total_qa = conn.execute('SELECT count(*) FROM qa_units').fetchone()[0]
total_videos = conn.execute('SELECT count(*) FROM videos').fetchone()[0]
print(f"==================================================")
print(f"📊 Total Extracted Q&A Units: {total_qa}")
print(f"📹 Total Videos Cataloged: {total_videos}")
print(f"==================================================")

print("\n--- 📁 Category Breakdown ---")
for row in conn.execute('SELECT topic_category, count(*) as c FROM qa_units GROUP BY topic_category ORDER BY c DESC'):
    print(f"  • {row['topic_category']}: {row['c']}")

print("\n--- 👥 Interlocutor Breakdown ---")
for row in conn.execute('SELECT interlocutor_type, count(*) as c FROM qa_units GROUP BY interlocutor_type ORDER BY c DESC'):
    print(f"  • {row['interlocutor_type']}: {row['c']}")

print("\n--- 🎥 Processed Videos Summary ---")
query = '''
    SELECT q.video_id, COALESCE(v.title_ar, q.video_title_ar) as title, count(q.id) as qa_count 
    FROM qa_units q
    LEFT JOIN videos v ON q.video_id = v.video_id
    GROUP BY q.video_id
    ORDER BY qa_count DESC
'''
for row in conn.execute(query):
    print(f"  • [{row['qa_count']:02d} Q&As] {row['video_id']} : {row['title']}")

print("\n--- 🔍 Detailed Sample of Dialectical Units ---")
sample_query = '''
    SELECT id, video_id, topic_category, interlocutor_type, question_ar, question_en,
           core_answer_summary_ar, analogy_or_metaphor_ar, socratic_question_ar,
           timestamp_url, timestamp_formatted
    FROM qa_units 
    WHERE analogy_or_metaphor_ar IS NOT NULL AND analogy_or_metaphor_ar != ''
    LIMIT 3
'''
for i, r in enumerate(conn.execute(sample_query)):
    print(f"\n[Analogy Sample {i+1}] {r['topic_category']} ({r['timestamp_formatted']})")
    print(f"  س: {r['question_ar']}")
    print(f"  ج: {r['core_answer_summary_ar']}")
    print(f"  💡 قياس / تمثيل: {r['analogy_or_metaphor_ar']}")
    if r['socratic_question_ar']:
        print(f"  ❓ سؤال سقراطي: {r['socratic_question_ar']}")
    print(f"  🔗 رابط التوقيت المباشر: {r['timestamp_url']}")
