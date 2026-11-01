import sqlite3
import json
import re
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Exact dictionary of full-field and substring replacements for all CJK/Korean/English splices found
EXACT_FIELD_OVERRIDES = {
    ("f7Jz-oRuLEY_013", "core_answer_summary_ar"): "لا توجد تناقضات في القرآن؛ فالتناقضات الظاهرية تنشأ فقط من سوء التفسير أو عدم فهم السياق الكامل للآيات.",
    ("xmxuhX-t0vg_007", "core_answer_summary_ar"): "القرآن يؤكد أنه وحي منزل من عند الله، وطبيعته المعجزة تثبت مصدره الإلهي بخلاف الكتابات البشرية.",
    ("ZQuttV4Uoa0_002", "question_ar"): "لماذا تتبع شخصاً أمر أتباعه بشراء السيوف واستخدامها في القتال؟",
    ("nWZs52jTvZQ_005", "core_answer_summary_ar"): "يشرح المتحدث أن نمط الحياة المادي المعاصر فارغ لأنه يفتقر إلى التأمل الحقيقي والإشباع الروحي، وأن الإنسان يبحث عن ذاته الحقيقية من خلال التأمل والتفاعل الإنساني الصادق.",
    ("bvq51UNgFkA_006", "core_answer_summary_ar"): "إن نظام العقوبات الإسلامي أكثر فعالية في ردع الجرائم ومنعها بسبب حزمه وقوة الردع فيه، كما يثبت ذلك التطبيق التاريخي للإسلام حيث كانت معدلات السرقة في أدنى مستوياتها.",
    ("_wAhT_JJZWU_001", "question_ar"): "هل انتشار الإسلام عبر الفتوحات العسكرية أمر جيد أم سيئ إذا كان دعوةً بالحجة وليس مجرد حملة عسكرية؟",
    ("nulm1c1d-NA_003", "question_ar"): "لماذا تعتقد أن كل الشرور والمصائب التي تحدث في العالم هي من الشيطان وليست من الله؟",
    ("ZrUCph8OR80_001", "question_ar"): "هل يوجد أي شيء في القرآن والسنة يمكن وصفه بأنه همجي أو متوحش؟",
    ("ZrUCph8OR80_001", "core_answer_summary_ar"): "يرد المتحدث بأن ما يصفه البعض بالقسوة في التشريع الجنائي إنما هو أحكام رادعة محددة بشروط صارمة جداً ولا تطبق إلا في حالات استثنائية لحماية المجتمع.",
    ("ZrUCph8OR80_019", "core_answer_summary_ar"): "الأخلاق موضوعية بالنسبة لمن يؤمن بمصدرها الإلهي، وبالنسبة لنا كمسلمين فهي حقيقة موضوعية مطلقة مصدرها الله.",
    ("VXj5nrLjAr8_004", "core_answer_summary_ar"): "يؤكد المتحدث أن ادعاء لورنس كراوس قابل للدحض وقد ثبت بطلانه علمياً وفلسفياً، بخلاف الادعاءات غير القابلة للفحص.",
    ("tngZKqTUN8o_004", "verbatim_quote_ar"): "في الإسلام يعني تحريك المياه الراكدة. نعم هذا صحيح، هناك جهود كبيرة يبذلها المجتمع المسلم في جنوب أفريقيا، ولكن للأسف هناك تقصير في الاعتناء بدعوة غير المسلمين إلى الإسلام، وليس عندهم فقط بل على مستوى العالم.",
    ("tngZKqTUN8o_073", "core_answer_summary_ar"): "يرد الداعية بأن كسب العقل عبر الأدلة والبراهين المنطقية أهم وأبقى من مجرد مخاطبة العواطف.",
    ("HSZdX5ZDCow_002", "core_answer_summary_ar"): "عادةً ما ينتقم القادة السياسيون من خصومهم عند التمكن من السلطة، ولكن النبي محمد ﷺ علّم الرحمة والعفو الشامل عن الجميع.",
    ("HSZdX5ZDCow_002", "logical_premises_ar_json"): json.dumps([
        "المقدمة الأولى: القادة السياسيون عادةً ما ينتقمون من خصومهم عند الوصول للسلطة.",
        "المقدمة الثانية: النبي محمد ﷺ ضرب أروع الأمثلة في الرحمة والعفو العام عن الجميع عند فتح مكة.",
        "النتيجة: ينبغي للقادة اتباع نموذج الرحمة والعفو النبوي."
    ], ensure_ascii=False),
    ("HSZdX5ZDCow_014", "core_answer_summary_ar"): "أوضح الداعية أنه حتى النبي محمد ﷺ لم يستطع إجبار عمه على دخول الإسلام، مما يؤكد أن الإيمان اختيار شخصي وهداية من الله.",
    ("HSZdX5ZDCow_014", "logical_premises_ar_json"): json.dumps([
        "المقدمة الأولى: لم يستطع النبي محمد ﷺ إقناع عمه أبي طالب باعتناق الإسلام رغماً عنه.",
        "المقدمة الثانية: الدخول في الدين هو اختيار شخصي وهداية من الله وحده.",
        "النتيجة: لا إكراه في الدين ولا يمكن فرض الإيمان بالقوة."
    ], ensure_ascii=False),
    ("HSZdX5ZDCow_014", "fallacy_or_misconception_exposed_ar"): "سوء فهم حرية الإرادة والهداية الإلهية",
    ("P4h2HVQet9E_007", "socratic_question_ar"): "كيف يُعتبر أشخاص مثل الملك آرثر أبطالاً في التاريخ عندما غزوا أراضي الدول الأخرى؟",
    ("P4h2HVQet9E_008", "question_ar"): "لماذا يُصنّف قادة مثل الملك آرثر كأبطال عندما غزوا بلداناً أخرى؟",
    ("P4h2HVQet9E_020", "verbatim_quote_ar"): "لا، هل هذا يعني أنها لم تقم بعلاقة زوجية؟ لا، وبالتالي لم تكن هناك علاقة قسرية.",
    ("P4h2HVQet9E_024", "socratic_question_ar"): "ما الذي تقصده بمصطلح «حياة محتملة»؟",
    ("o8vChP5OkJU_011", "core_answer_summary_ar"): "يرد الداعية بأن النص يستخدم اللغة المجازية والرمزية للإشارة إلى النبي القادم وليس إلى سليمان نفسه.",
    ("H2aVOoEvMMk_018", "core_answer_summary_ar"): "يرى كثير من الناس أن المثلية الجنسية أمر غير طبيعي، ولكن مع تغير المعايير الوضعية قد يعتبرها المجتمع العلماني طبيعية بعد مئة عام.",
    ("H2aVOoEvMMk_018", "logical_premises_ar_json"): json.dumps([
        "كثير من الناس يعتبرون المثلية الجنسية سلوكاً غير طبيعي.",
        "في المجتمعات المتغيرة التي لا تستند لوحي ثابت، قد يُنظر إلى ما هو مرفوض اليوم كأمر طبيعي بعد قرن من الزمان."
    ], ensure_ascii=False),
    ("H2aVOoEvMMk_018", "verbatim_quote_ar"): "لا، ولكنني لا أعتقد أن ذلك سيكون طبيعياً، وكذلك المثلية الجنسية.",
    ("local_753519_049", "core_answer_summary_ar"): "الأرجاس والقبائح تعني الذنوب والأعمال الشريرة، والأمر الإلهي هنا موجه للقضاء والحكم بالتطهير.",
    ("local_753519_049", "logical_premises_ar_json"): json.dumps([
        "جميع الأرجاس والقبائح تعني الذنوب والأعمال الشريرة.",
        "أمر الحكم والتطهير صادر من الله.",
        "الأمر موجه للقاضي الأرضي ليحكم وفقاً للعدل الإلهي."
    ], ensure_ascii=False),
    ("local_753519_050", "logical_premises_ar_json"): json.dumps([
        "هناك أمر للقاضي ليحكم بالعدل الإلهي.",
        "الأمر يهدف للحكم والتطهير من الفساد.",
        "الأمر موجه للقاضي الأرضي للتنفيذ وفقاً للشريعة."
    ], ensure_ascii=False),
}

SUBSTRING_REPLACEMENTS = [
    ("따라서،", "لذلك،"),
    ("따라서", "لذلك"),
    ("여전히", "لا تزال"),
    ("행동اته", "أفعاله"),
    ("예수", "يسوع"),
    ("자حات", "المؤثرات"),
    ("ال명령", "الأمر"),
    ("ي\ufffd行", "يحكم"),
    ("بprocesseات", "بعمليات"),
    ("لproceses", "لعمليات"),
    ("المiracles", "المعجزات"),
    ("فالمachines", "فالآلات"),
    ("كمessenger", "كرسول"),
    ("للمonotheism", "للتوحيد"),
    ("تeachings", "تعاليمه"),
    ("تeaching of", "تعاليم"),
    ("نear-sacrifice", "فداء"),
    ("مechanism", "آلية"),
    ("مutually exclusive", "متنافيين"),
    ("بال Apostasy", "بالردة"),
    ("الرansom", "الفدية"),
    ("رansom", "فدية"),
    ("لل compartion", "للمشاركة"),
    ("Giving names to creation that belong uniquely to الله", "إعطاء أسماء وصفات خاصة بالله للمخلوقات"),
    ("Giving مثل هذه الألقاب للأشياء", "إعطاء مثل هذه الألقاب للمخلوقات"),
    ("يself-introduce", "يعرّف نفسه"),
    ("المессيا", "المسيح"),
    ("المессias", "المسيح"),
    ("إنجيل لوثر", "إنجيل لوقا"),
    ("يرمنيا", "إرميا"),
    ("Jeremiah", "إرميا"),
    ("Revelations", "سفر الرؤيا"),
    ("السcribes", "النساخ"),
    ("للسcribes", "للنساخ"),
    ("فيgeneral", "بشكل عام"),
    ("ceased،", "توقفت،"),
    ("مبرمjàجة", "همجية"),
    ("مبرمjàج", "همجي"),
    ("غيرiable", "غير موثوق"),
    ("برеляiability", "بموثوقية"),
    ("ل muslim", "للمسلم"),
    ("بال muslim", "بالمسلم"),
    ("الMuslims", "المسلمون"),
    ("sometimes some people find this difficult.", "أحياناً يجد بعض الناس صعوبة في ذلك."),
    ("أكثر aptitude", "أكثر قابلية واستعداداً"),
    ("دonors", "المتبرعين"),
    ("donors", "المتبرعون"),
    ("بال modesty", "بالحشمة والحياء"),
    ("ا Donate charity", "تصدق"),
    ("وpractices", "ومناسكه"),
    ("م Caller", "كداعية"),
    ("يexists", "يوجد"),
    ("أن ي존", "أن يكون واجب الوجود"),
    ("علىTEGRITY OF THE MESSAGE AND ITS IMPACT", "على نقاء الرسالة وتأثيرها"),
    ("منTEGRITY OF THE MESSAGE AND ITS IMPACT", "من نقاء الرسالة وتأثيرها"),
    ("سœur", "أخت"),
    ("TURN OFF", "مغلق"),
    ("OFF", "مغلق"),
    ("ي Treatmentهن", "يتم معاملتهن"),
    ("من ي hypocritical", "يتصف بالنفاق والازدواجية"),
    ("ال hypocritical", "النفاق وازدواجية المعايير"),
    ("هلISIS", "هل داعش"),
    ("ISIS", "داعش"),
    ("outcomes جنسي", "علاقات جنسية"),
    ("outcomes", "علاقة جنسية"),
    ("Pakistan has it.", "باكستان تمتلكه."),
    ("—that's terrible", "—هذا أمر فظيع"),
    ("غير Sahte", "غير ممنوع قانوناً"),
    ("ليس Sahte", "ليس ممنوعاً قانوناً"),
    ("وم mortally", "وفانٍ"),
    ("مست stressing", "مؤكداً"),
    ("وي Curse", "ويشتم بشكل"),
    ("أناalking to you", "أنا أتحدث إليك"),
    ("التطهيرracial أو العرقي", "التطهير العرقي والإبادة"),
    ("التطهير racial أو العرقي", "التطهير العرقي والإبادة"),
    ("سياسة جنocيدية", "سياسة إبادة جماعية"),
    ("أوردانج زبruitضَع", "أورنكزيب وظّف"),
    ("للMadhhab الهنافي", "للمذهب الحنفي"),
    ("لا أ Agree", "لا أتفق"),
    ("بال systems", "بالأنظمة"),
    ("معياركmoral", "معيارك الأخلاقي"),
    ("السمotoxico", "السم"),
    ("جين جنسي Homo", "جين المثلية الجنسية"),
    ("بهomo جنسياً", "بميول مثلية"),
    ("هو homo جنسي", "ذو ميول مثلية"),
    ("طريقةalking", "طريقة المشي أو الكلام"),
    ("كنت نihilist", "كنت عدمياً"),
    ("الخصم يIdentify as nihilist.", "الخصم يعرّف نفسه بأنه عدمي."),
    ("النihilists", "العدميون"),
    ("ي Describes", "يوصف"),
    ("و Justice هي", "والعدالة"),
    ("لماذا ي puff إلى وجود", "لماذا يظهر الابن فجأة إلى الوجود"),
    ("ب labeling", "بالتسمية"),
    ("مع\ufffdضلقة", "مع اللحم"),
    ("قد\ufffdعترفنا", "قد اعترفنا"),
    ("إффективية", "فعالية"),
    ("و_statements_ حول أنه ابن الإنسان ومجيء ب núبوات_", "وعباراته حول كونه ابن الإنسان ومجيئه مع السحب"),
    ("تعني الإعجاز وليس terroism الحديث", "تعني الردع والإعداد وليس الإرهاب بمفهومه الحديث"),
    ("بني قريش", "بني قريظة"),
    ("مكتبةك", "كتابك المقدس"),
    ("دعم\ufffd الحجج", "دعم الحجج"),
    ("أبطل\ufffd حجتك", "أبطلت حجتك"),
]

ar_fields = [
    'question_ar', 'core_answer_summary_ar', 'logical_premises_ar_json',
    'fallacy_or_misconception_exposed_ar', 'analogy_or_metaphor_ar',
    'socratic_question_ar', 'verbatim_quote_ar'
]

def clean_db_glitches(target_db_path):
    conn = sqlite3.connect(target_db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM qa_units").fetchall()
    updated_count = 0

    with conn:
        for r in rows:
            rec_id = r['id']
            updates = {}
            for f in ar_fields:
                val = r[f] or ''
                orig_val = val
                if (rec_id, f) in EXACT_FIELD_OVERRIDES:
                    val = EXACT_FIELD_OVERRIDES[(rec_id, f)]
                else:
                    for old_s, new_s in SUBSTRING_REPLACEMENTS:
                        if old_s in val:
                            val = val.replace(old_s, new_s)
                    val = re.sub(r'[\u4e00-\u9fff\uac00-\ud7af\ufffd]+', '', val)
                if val != orig_val:
                    updates[f] = val
            if updates:
                set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
                params = list(updates.values()) + [rec_id]
                conn.execute(f"UPDATE qa_units SET {set_clause} WHERE id = ?", params)
                updated_count += 1

        conn.execute("INSERT INTO qa_fts(qa_fts) VALUES('rebuild')")
    conn.close()
    return updated_count


if __name__ == "__main__":
    src_dir = Path(r"data-20261031T140241Z-1-001\data")
    dst_dir = Path("data")
    if src_dir.exists():
        for fname in ["dawah_knowledge.db", "extracted_qa.jsonl", "extraction_checkpoint.json", "extraction_live.log"]:
            s = src_dir / fname
            d = dst_dir / fname
            if s.exists():
                shutil.copy2(s, d)
                print(f"[+] Synced {fname} ({s.stat().st_size} bytes)")

    db_path = dst_dir / "dawah_knowledge.db"
    updated_count = clean_db_glitches(db_path)
    print(f"[+] Successfully repaired {updated_count} records and rebuilt FTS5 index!")
