#!/usr/bin/env python3
"""
core/smart_bot.py
Intelligent, zero-LLM automated dialogue assistant.
Uses deterministic NLP, bilingual intent classification, token intersection,
and pre-analyzed dialectical answer keys to provide instant, persuasive answers
with verbatim evidence and exact YouTube timestamp links.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from db.database import search_qa, get_db_connection

# Category translations
CATEGORY_TRANSLATIONS = {
    "Theology & God's Existence": "العقيدة ووجود الله",
    "Problem of Evil & Suffering": "مشكلة الشر والمعاناة",
    "Comparative Religion & Christianity": "مقارنة الأديان والنصرانية",
    "Objective Morality & Ethics": "الأخلاق الموضوعية والإلحاد",
    "Epistemology & Science": "المعرفة والعلم التجريبي",
    "Prophethood & Quran": "النبوة والقرآن الكريم",
    "Shariah & Social Inquiries": "الشريعة والشبهات الاجتماعية"
}

INTERLOCUTOR_TRANSLATIONS = {
    "Atheist / Materialist": "ملحد / مادي",
    "Materialist": "مادي",
    "Christian / Trinitarian": "مسيحي / نصراني",
    "Agnostic / Skeptic": "لاأدري / متشكك",
    "Secular Humanist / Christian": "علماني / نصراني",
    "Secular Humanist": "علماني إنسانوي",
    "Inquiring Seeker": "باحث عن الحق",
    "Muslim with Doubts": "مسلم يطلب دليلاً"
}

# Bilingual Intent Concept Graph
THEOLOGICAL_INTENT_MAP = [
    {
        "category": "Theology & God's Existence",
        "subcategory": "First Cause & Who Created God",
        "keywords_en": ["who created god", "first cause", "beginning of universe", "infinite regress", "category mistake", "contingency", "kalam"],
        "keywords_ar": ["من خلق الله", "السبب الاول", "بدايه الكون", "تسلسل لا نهائي", "خالق الله", "الواجب الوجود", "السببيه", "علة الوجود"],
        "default_question_id": "theology_001"
    },
    {
        "category": "Problem of Evil & Suffering",
        "subcategory": "Divine Wisdom & Purpose of Suffering",
        "keywords_en": ["problem of evil", "why suffering", "innocent children", "cancer", "evil and good", "bad things happen", "god allows evil"],
        "keywords_ar": ["مشكله الشر", "لماذا الالم", "معاناه الاطفال", "الشر والخير", "لماذا يسمح الله بالشر", "المصائب", "حكمه الشر"],
        "default_question_id": "evil_001"
    },
    {
        "category": "Comparative Religion & Christianity",
        "subcategory": "Trinity Contradictions & Jesus in Islam",
        "keywords_en": ["trinity", "jesus is god", "son of god", "father son", "three in one", "crucifixion", "god died"],
        "keywords_ar": ["التثليث", "يسوع هو الله", "ابن الله", "الثالوث", "ثلاثه في واحد", "صلب المسيح", "موت الله", "الوهيه المسيح"],
        "default_question_id": "trinity_001"
    },
    {
        "category": "Comparative Religion & Christianity",
        "subcategory": "Textual Preservation: Quran vs Bible",
        "keywords_en": ["bible changed", "quran preserved", "manuscripts", "bible corruption", "anonymous authors", "council of nicaea"],
        "keywords_ar": ["تحريف الانجيل", "حفظ القران", "مخطوطات الكتاب المقدس", "التحريف", "مجمع نيقيه", "اسانيد القران"],
        "default_question_id": "preservation_001"
    },
    {
        "category": "Objective Morality & Ethics",
        "subcategory": "Grounding Morality Without God",
        "keywords_en": ["objective morality", "moral without god", "atheist ethics", "subjective morals", "good and bad", "right and wrong"],
        "keywords_ar": ["مصدر الاخلاق", "اخلاق الملحد", "الاخلاق بدون اله", "الاخلاق الموضوعيه", "النسبيه الاخلاقيه"],
        "default_question_id": "morality_001"
    },
    {
        "category": "Epistemology & Science",
        "subcategory": "Scientism vs Metaphysics",
        "keywords_en": ["science proves", "scientism", "empirical proof", "scientific method", "only believe in science", "evolution vs creation"],
        "keywords_ar": ["العلمويه", "اثبات علمي", "المنهج التجريبي", "التطور والخلق", "هل العلم يفسر كل شيء"],
        "default_question_id": "epist_001"
    },
    {
        "category": "Shariah & Social Inquiries",
        "subcategory": "Women in Islam & Social Jurisprudence",
        "keywords_en": ["women in islam", "hijab", "four wives", "polygamy", "inheritance", "female rights", "oppression"],
        "keywords_ar": ["المراه في الاسلام", "الحجاب", "تعدد الزوجات", "حقوق المراه", "الميراث", "اضطهاد المراه"],
        "default_question_id": "women_001"
    }
]


ARABIC_STOPWORDS = {
    "لماذا", "كيف", "هل", "ما", "ماذا", "من", "اين", "متى", "كم", "اي", "اية",
    "في", "على", "الى", "عن", "مع", "بين", "حتى", "منذ", "مذ",
    "ان", "انما", "كان", "يكون", "تكون", "اصبح", "ليس",
    "هو", "هي", "هم", "هما", "هن", "انت", "انتم", "انا", "نحن",
    "هذا", "هذه", "هؤلاء", "ذلك", "تلك", "اولئك",
    "الذي", "التي", "الذين", "اللاتي", "اللواتي",
    "كل", "بعض", "غير", "سوى", "الا", "او", "ام", "بل", "لكن", "ثم", "و", "ف", "يا"
}

ENGLISH_STOPWORDS = {
    "why", "how", "what", "where", "when", "who", "which", "whom", "whose",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "having", "can", "could", "should", "would", "will",
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "until",
    "in", "on", "at", "to", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "don", "t"
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    t = re.sub(r"[\u064B-\u065F\u0670]", "", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ء", "")
    t = t.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def detect_language(text: str) -> str:
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    return "ar" if arabic_chars > latin_chars else "en"


def compute_match_score(query_norm: str, item: Dict[str, Any], query_lang: str) -> float:
    stopwords = ARABIC_STOPWORDS if query_lang == "ar" else ENGLISH_STOPWORDS
    raw_tokens = query_norm.split()
    content_tokens = [t for t in raw_tokens if t not in stopwords and len(t) > 1]

    search_tokens = content_tokens if content_tokens else raw_tokens
    if not search_tokens:
        return 0.0

    target_q = normalize_text(item.get("question_ar" if query_lang == "ar" else "question_en", ""))
    target_tokens = set(target_q.split())
    target_content_tokens = {t for t in target_tokens if t not in stopwords}

    if query_norm in target_q:
        return 95.0

    # Overlap MUST be evaluated on content tokens
    overlap = len(set(search_tokens).intersection(target_content_tokens))
    if overlap == 0:
        return 0.0

    score = (overlap / len(search_tokens)) * 70.0

    summary_text = normalize_text(
        (item.get("core_answer_summary_ar") if query_lang == "ar" else item.get("core_answer_summary_en")) or ""
    )
    summary_tokens = {t for t in summary_text.split() if t not in stopwords}
    sum_overlap = len(set(search_tokens).intersection(summary_tokens))
    score += (sum_overlap / len(search_tokens)) * 20.0

    tags = [normalize_text(t) for t in item.get("tags", [])]
    for tag in tags:
        if any(tok in tag for tok in search_tokens):
            score += 10.0
            break

    return min(score, 100.0)


class SmartDialecticBot:
    def answer_query(self, user_query: str) -> Dict[str, Any]:
        user_query = user_query.strip()
        if not user_query:
            return {
                "status": "empty",
                "message": "Please enter a question or topic to search."
            }

        query_lang = detect_language(user_query)
        query_norm = normalize_text(user_query)

        candidates = search_qa(query=user_query, limit=20)
        if not candidates:
            candidates = search_qa(query="", limit=50)

        intent_category = None
        for intent in THEOLOGICAL_INTENT_MAP:
            kw_list = intent["keywords_ar"] if query_lang == "ar" else intent["keywords_en"]
            for kw in kw_list:
                norm_kw = normalize_text(kw)
                if norm_kw in query_norm or all(part in query_norm for part in norm_kw.split()):
                    intent_category = intent["category"]
                    break
            if intent_category:
                break

        scored_candidates = []
        for c in candidates:
            sc = compute_match_score(query_norm, c, query_lang)
            if intent_category and c.get("topic_category") == intent_category:
                sc += 15.0
            scored_candidates.append((sc, c))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_score = scored_candidates[0][0] if scored_candidates else 0.0

        if scored_candidates and top_score >= 35.0:
            best_match = scored_candidates[0][1]
            related = [c[1] for c in scored_candidates[1:4] if c[0] >= 15.0]

            return {
                "status": "success",
                "query": user_query,
                "language": query_lang,
                "confidence_score": round(top_score, 2),
                "matched_qa": best_match,
                "response_package": self._format_response_package(best_match, query_lang),
                "related_questions": [
                    {
                        "id": r["id"],
                        "question": r["question_ar"] if query_lang == "ar" else r["question_en"],
                        "category": CATEGORY_TRANSLATIONS.get(r["topic_category"], r["topic_category"]) if query_lang == "ar" else r["topic_category"]
                    } for r in related
                ]
            }
        elif scored_candidates and top_score >= 15.0:
            best_match = scored_candidates[0][1]
            return {
                "status": "partial_match",
                "query": user_query,
                "language": query_lang,
                "confidence_score": round(top_score, 2),
                "message": (
                    "⚠️ لم نجد إجابة مباشرة متطابقة حرفياً مع سؤالك في الحلقات المستخرجة حتى الآن، ولكن هذه أقرب محاورة في سياق عام:"
                    if query_lang == "ar" else
                    "⚠️ No direct exact match was found in the extracted corpus yet, but here is the closest related dialogue:"
                ),
                "matched_qa": best_match,
                "response_package": self._format_response_package(best_match, query_lang),
                "related_questions": []
            }
        else:
            return {
                "status": "no_match",
                "query": user_query,
                "language": query_lang,
                "confidence_score": 0.0,
                "message": (
                    "عذراً، لم نعثر حتى الآن على حجة مسجلة تتناول موضوع سؤالك في الحلقات المستخرجة حالياً. "
                    "نظراً لأن عملية استخراج الأرشيف ما زالت جارية، يمكنك تجربة أحد الموضوعات المتوفرة حالياً مثل:\n"
                    "• وجود الله وقانون السببية\n"
                    "• الرد على شبهات تحريف القرآن ومقارنة المخطوطات\n"
                    "• ألوهية المسيح ومناظرات التثليث\n"
                    "• حدود المنهج العلمي ونقد المادية"
                    if query_lang == "ar" else
                    "Sorry, no specific dialectical argument has been extracted for this exact topic yet in the current archive. "
                    "Try searching for available topics such as God's existence, the Trinity, textual preservation, or scientism."
                ),
                "matched_qa": None,
                "response_package": None,
                "related_questions": [
                    {
                        "id": c["id"],
                        "question": c["question_ar"] if query_lang == "ar" else c["question_en"],
                        "category": CATEGORY_TRANSLATIONS.get(c["topic_category"], c["topic_category"]) if query_lang == "ar" else c["topic_category"]
                    } for c in candidates[:3]
                ] if candidates else []
            }

    def _format_response_package(self, qa: Dict[str, Any], lang: str) -> Dict[str, Any]:
        is_ar = (lang == "ar")
        raw_premises = qa.get("logical_premises_ar") if is_ar and qa.get("logical_premises_ar") else qa.get("logical_premises", [])
        premises = [str(p).strip() for p in (raw_premises or []) if str(p).strip()]
        analogy = qa.get("analogy_or_metaphor_ar") if is_ar and qa.get("analogy_or_metaphor_ar") else qa.get("analogy_or_metaphor")
        socratic = qa.get("socratic_question_ar") if is_ar and qa.get("socratic_question_ar") else qa.get("socratic_question")
        verbatim = qa.get("verbatim_quote_ar") if is_ar and qa.get("verbatim_quote_ar") else qa.get("verbatim_quote")

        if analogy and str(analogy).strip().lower() in ["none", "none used", "none used.", "none.", "n/a", ""]:
            analogy = ""
        if socratic and str(socratic).strip().lower() in ["none", "none used", "none used.", "none.", "n/a", ""]:
            socratic = ""
        cat = CATEGORY_TRANSLATIONS.get(qa.get("topic_category"), qa.get("topic_category")) if is_ar else qa.get("topic_category")
        interlocutor = INTERLOCUTOR_TRANSLATIONS.get(qa.get("interlocutor_type"), qa.get("interlocutor_type")) if is_ar else qa.get("interlocutor_type")

        return {
            "title": qa.get("question_ar" if is_ar else "question_en"),
            "interlocutor": interlocutor,
            "category": cat,
            "summary": qa.get("core_answer_summary_ar" if is_ar else "core_answer_summary_en"),
            "logical_premises": premises,
            "fallacy_exposed": qa.get("fallacy_or_misconception_exposed_ar" if is_ar else "fallacy_or_misconception_exposed"),
            "analogy": analogy,
            "socratic_question": socratic,
            "verbatim_quote": verbatim,
            "video_title": qa.get("video_title_ar"),
            "timestamp": qa.get("timestamp_formatted"),
            "timestamp_url": qa.get("timestamp_url")
        }


smart_bot = SmartDialecticBot()
