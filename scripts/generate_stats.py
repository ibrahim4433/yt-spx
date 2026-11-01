import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
conn = sqlite3.connect('data/dawah_knowledge.db')
c = conn.cursor()

# 1. Total QA & videos
total_qa = c.execute('SELECT count(1) FROM qa_units').fetchone()[0]
total_videos = c.execute('SELECT count(distinct video_id) FROM qa_units').fetchone()[0]

# 2. QA per video distribution
video_counts = c.execute('SELECT video_id, count(1) as cnt, video_title_ar FROM qa_units GROUP BY video_id ORDER BY cnt DESC').fetchall()
counts_list = [v[1] for v in video_counts]
avg_qa = sum(counts_list) / len(counts_list) if counts_list else 0
max_qa = video_counts[0] if video_counts else (None, 0, '')
min_qa = video_counts[-1] if video_counts else (None, 0, '')

# 3. Categories breakdown
cat_counts = c.execute('SELECT topic_category, count(1) as cnt FROM qa_units GROUP BY topic_category ORDER BY cnt DESC').fetchall()

# 4. Interlocutor breakdown
interlocutor_counts = c.execute('SELECT interlocutor_type, count(1) as cnt FROM qa_units GROUP BY interlocutor_type ORDER BY cnt DESC').fetchall()

# 5. Dialectical components
premises_count = c.execute('SELECT count(1) FROM qa_units WHERE logical_premises_ar_json != "[]" AND logical_premises_ar_json IS NOT NULL').fetchone()[0]
analogies_count = c.execute('SELECT count(1) FROM qa_units WHERE analogy_or_metaphor_ar != "" AND analogy_or_metaphor_ar IS NOT NULL AND analogy_or_metaphor_ar NOT LIKE "None%"').fetchone()[0]
socratic_count = c.execute('SELECT count(1) FROM qa_units WHERE socratic_question_ar != "" AND socratic_question_ar IS NOT NULL AND socratic_question_ar NOT LIKE "None%"').fetchone()[0]
quotes_count = c.execute('SELECT count(1) FROM qa_units WHERE verbatim_quote_ar != "" AND verbatim_quote_ar IS NOT NULL').fetchone()[0]

print('TOTAL_QA:', total_qa)
print('TOTAL_VIDEOS:', total_videos)
print('AVG_QA:', round(avg_qa, 1))
print('MAX_QA:', max_qa)
print('MIN_QA:', min_qa)
print('\nCATEGORIES:')
for cat, cnt in cat_counts:
    pct = round((cnt / total_qa) * 100, 1)
    print(f"  - {cat}: {cnt} ({pct}%)")

print('\nINTERLOCUTORS:')
for inl, cnt in interlocutor_counts:
    pct = round((cnt / total_qa) * 100, 1)
    print(f"  - {inl}: {cnt} ({pct}%)")

print('\nDIALECTICS:')
print(f"  - Premises: {premises_count} / {total_qa} ({round(premises_count/total_qa*100, 1)}%)")
print(f"  - Analogies: {analogies_count} / {total_qa} ({round(analogies_count/total_qa*100, 1)}%)")
print(f"  - Socratic: {socratic_count} / {total_qa} ({round(socratic_count/total_qa*100, 1)}%)")
print(f"  - Verbatim Quotes: {quotes_count} / {total_qa} ({round(quotes_count/total_qa*100, 1)}%)")

print('\nTOP 8 VIDEOS WITH MOST EXTRACTED UNITS:')
for v in video_counts[:8]:
    print(f"  - {v[1]} وحدة: {v[2]} (ID: {v[0]})")

conn.close()
