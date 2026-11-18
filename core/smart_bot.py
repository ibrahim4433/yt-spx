#!/usr/bin/env python3
"""
core/smart_bot.py
Zero-LLM deterministic dialectical assistant for Dawah Dialectics V2 (yt-spx).

Uses bilingual intent classification, Arabic morphological normalization, token
intersection, and V2 speaker-diarized answer dossiers (including Da'ee voiceprint
attribution, interlocutor verbatim quotes, dual-speaker exchange timelines, and
Stage 1 scenario turn fallback).
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from db.database import (
    BILINGUAL_TOPIC_DOMAINS,
    get_category_arabic_name,
    search_qa,
    search_scenario_turns,
)

CATEGORY_TRANSLATIONS = {k: v["ar"] for k, v in BILINGUAL_TOPIC_DOMAINS.items()}

INTERLOCUTOR_TRANSLATIONS = {
    "Atheist / Materialist": "ملحد / مادي",
    "Materialist": "مادي",
    "Christian / Trinitarian": "مسيحي / نصراني",
    "Agnostic / Skeptic": "لاأدري / متشكك",
    "Secular Humanist / Christian": "علماني / نصراني",
    "Secular Humanist": "علماني إنسانوي",
    "Inquiring Seeker": "باحث عن الحق",
    "Muslim with Doubts": "مسلم يطلب دليلاً",
    "Jewish / Rabbinic": "يهودي / حاخامي",
    "Hindu / Eastern": "هندوسي / شرق آسيوي",
    "Buddhist / Eastern": "بوذي / شرق آسيوي",
    "Shiite": "شيعي",
    "Shia": "شيعي",
}

_UBIQUITOUS_TOKENS = {
    "god", "allah", "islam", "islamic", "muslim", "muslims", "quran",
    "jesus", "christ", "christian", "christians", "bible", "biblical", "religion",
    "people", "believe", "allow", "mean", "question", "does",
    "الله", "الاسلام", "مسلم", "مسلمين", "قران", "القران", "الدين", "ديني",
}


_AR_CREATION_STEM_FORMS = frozenset({
    "خلق", "خالق", "مخلوق", "مخلوقه", "مخلوقات", "مخلوقون", "مخلوقين",
    "يخلق", "تخلق", "نخلق", "اخلق", "خلقت", "خلقوا", "خالقه", "خالقهم",
    "خلقه", "خلقها", "خلقهم", "خلقك", "خلقكم", "خلقني", "مخلوقاته",
})


def _stem_token(tok: str, lang: str) -> str:
    if lang == "ar":
        if tok in _AR_CREATION_STEM_FORMS:
            return "خلق"
        stemmed = tok
        for pref in ("وال", "فال", "بال", "كال", "لال", "لل", "ال"):
            if stemmed.startswith(pref) and len(stemmed) - len(pref) >= 2:
                stemmed = stemmed[len(pref):]
                break
        if stemmed in _AR_CREATION_STEM_FORMS:
            return "خلق"
        if len(stemmed) >= 4 and stemmed[0] in ("و", "ف", "ب", "ل") and stemmed[1:] in _AR_CREATION_STEM_FORMS:
            return "خلق"
        for suf in ("ات", "ون", "ين", "هم", "ها", "كم"):
            if stemmed.endswith(suf) and len(stemmed) - len(suf) >= 3:
                candidate = stemmed[:-len(suf)]
                if candidate in _AR_CREATION_STEM_FORMS:
                    return "خلق"
                stemmed = candidate
                break
        if len(stemmed) >= 5 and stemmed.endswith("ه") and stemmed[:-1] in _AR_CREATION_STEM_FORMS:
            return "خلق"
        return "خلق" if stemmed in _AR_CREATION_STEM_FORMS else stemmed
    if len(tok) > 4:
        if tok.endswith("ies"):
            return tok[:-3] + "y"
        if tok.endswith("ing") and len(tok) > 5:
            return tok[:-3]
        if tok.endswith("ed") and len(tok) > 4:
            return tok[:-2]
        if tok.endswith("es") and len(tok) > 4:
            return tok[:-2]
        if tok.endswith("s") and not tok.endswith("ss"):
            return tok[:-1]
    return tok


def _tokenize_with_stems(text: str, stopwords: set, lang: str) -> tuple[set, set]:
    norm = normalize_text(text)
    toks = {t for t in norm.split() if t and t not in stopwords and len(t) > 1}
    stems = {_stem_token(t, lang) for t in toks}
    return toks, stems

THEOLOGICAL_INTENT_MAP = [
    {
        "category": "Theology & God's Existence",
        "subcategory": "First Cause & Who Created God",
        "keywords_en": [
            "who created god", "first cause", "beginning of universe",
            "infinite regress", "category mistake", "contingency", "kalam",
        ],
        "keywords_ar": [
            "من خلق الله", "السبب الاول", "بدايه الكون", "تسلسل لا نهائي",
            "خالق الله", "الواجب الوجود", "السببيه", "علة الوجود",
        ],
    },
    {
        "category": "Problem of Evil & Suffering",
        "subcategory": "Divine Wisdom & Purpose of Suffering",
        "keywords_en": [
            "problem of evil", "why suffering", "innocent children",
            "cancer", "evil and good", "bad things happen", "god allows evil",
        ],
        "keywords_ar": [
            "مشكله الشر", "لماذا الالم", "معاناه الاطفال", "الشر والخير",
            "لماذا يسمح الله بالشر", "المصائب", "حكمه الشر",
        ],
    },
    {
        "category": "Comparative Religion & Christianity",
        "subcategory": "Trinity Contradictions & Jesus in Islam",
        "keywords_en": [
            "trinity", "jesus is god", "son of god", "father son",
            "three in one", "crucifixion", "god died",
        ],
        "keywords_ar": [
            "التثليث", "يسوع هو الله", "ابن الله", "الثالوث",
            "ثلاثه في واحد", "صلب المسيح", "موت الله", "الوهيه المسيح",
        ],
    },
    {
        "category": "Textual Preservation & Manuscripts",
        "subcategory": "Textual Preservation: Quran vs Bible",
        "keywords_en": [
            "bible changed", "quran preserved", "manuscripts",
            "bible corruption", "anonymous authors", "council of nicaea",
        ],
        "keywords_ar": [
            "تحريف الانجيل", "حفظ القران", "مخطوطات الكتاب المقدس",
            "التحريف", "مجمع نيقيه", "اسانيد القران",
        ],
    },
    {
        "category": "Objective Morality & Ethics",
        "subcategory": "Grounding Morality Without God",
        "keywords_en": [
            "objective morality", "moral without god", "atheist ethics",
            "subjective morals", "good and bad", "right and wrong",
        ],
        "keywords_ar": [
            "مصدر الاخلاق", "اخلاق الملحد", "الاخلاق بدون اله",
            "الاخلاق الموضوعيه", "النسبيه الاخلاقيه",
        ],
    },
    {
        "category": "Epistemology & Science",
        "subcategory": "Scientism vs Metaphysics",
        "keywords_en": [
            "science proves", "scientism", "empirical proof",
            "scientific method", "only believe in science", "evolution vs creation",
        ],
        "keywords_ar": [
            "العلمويه", "اثبات علمي", "المنهج التجريبي",
            "التطور والخلق", "هل العلم يفسر كل شيء",
        ],
    },
    {
        "category": "Shariah & Social Inquiries",
        "subcategory": "Women in Islam & Social Jurisprudence",
        "keywords_en": [
            "women in islam", "hijab", "four wives", "polygamy",
            "inheritance", "female rights", "oppression",
        ],
        "keywords_ar": [
            "المراه في الاسلام", "الحجاب", "تعدد الزوجات",
            "حقوق المراه", "الميراث", "اضطهاد المراه",
        ],
    },
]

ARABIC_STOPWORDS = {
    "لماذا", "كيف", "هل", "ما", "ماذا", "من", "اين", "متى", "كم", "اي", "اية",
    "في", "على", "الى", "عن", "مع", "بين", "حتى", "منذ", "مذ",
    "ان", "انما", "كان", "يكون", "تكون", "اصبح", "ليس",
    "هو", "هي", "هم", "هما", "هن", "انت", "انتم", "انا", "نحن",
    "هذا", "هذه", "هؤلاء", "ذلك", "تلك", "اولئك",
    "الذي", "التي", "الذين", "اللاتي", "اللواتي",
    "كل", "بعض", "غير", "سوى", "الا", "او", "ام", "بل", "لكن", "ثم", "و", "ف", "يا",
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
    "than", "too", "very", "just", "don", "t",
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    t = re.sub(r"[\u064B-\u065F\u0670]", "", t)
    t = t.replace("\u0640", "")
    t = re.sub(r"[\u060C\u061B\u061F\u066A-\u066D\u06D4]+", " ", t)
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

    search_stems = [_stem_token(t, query_lang) for t in search_tokens]
    token_weights = [
        (0.25 if (t in _UBIQUITOUS_TOKENS or _stem_token(t, query_lang) in _UBIQUITOUS_TOKENS) else 1.0)
        for t in search_tokens
    ]
    total_weight = sum(token_weights) or 1.0

    def _weighted_overlap(field_text: str) -> tuple[float, int]:
        toks, stems = _tokenize_with_stems(field_text, stopwords, query_lang)
        if not toks:
            return 0.0, 0
        matched_w = 0.0
        specific_hits = 0
        for tok, st, w in zip(search_tokens, search_stems, token_weights):
            if tok in toks or st in stems:
                matched_w += w
                if w >= 1.0:
                    specific_hits += 1
        return matched_w / total_weight, specific_hits

    target_q = normalize_text(item.get("question_ar" if query_lang == "ar" else "question_en", ""))
    if query_norm and query_norm in target_q:
        return 95.0

    q_ratio, q_spec = _weighted_overlap(item.get("question_ar" if query_lang == "ar" else "question_en", "") or "")
    inter_ratio, inter_spec = _weighted_overlap(
        (item.get("interlocutor_verbatim_quote_ar") if query_lang == "ar" else item.get("interlocutor_verbatim_quote")) or ""
    )
    subcat_text = f"{item.get('topic_subcategory') or ''} {' '.join(str(t) for t in (item.get('tags') or []))}"
    tag_ratio, tag_spec = _weighted_overlap(subcat_text)

    premises_list = item.get("logical_premises_ar" if query_lang == "ar" else "logical_premises") or []
    summary_raw = (
        f"{(item.get('core_answer_summary_ar') if query_lang == 'ar' else item.get('core_answer_summary_en')) or ''} "
        f"{' '.join(str(p) for p in premises_list)} "
        f"{(item.get('fallacy_or_misconception_exposed_ar') if query_lang == 'ar' else item.get('fallacy_or_misconception_exposed')) or ''}"
    )
    sum_ratio, sum_spec = _weighted_overlap(summary_raw)

    total_specific_hits = max(q_spec, inter_spec, tag_spec, sum_spec)
    score = q_ratio * 65.0 + inter_ratio * 25.0 + sum_ratio * 22.0 + tag_ratio * 18.0

    # Prevent ubiquitous single words ("god", "islam", "allow") from hijacking unrelated multi-word questions
    specific_query_tokens = sum(1 for w in token_weights if w >= 1.0)
    if specific_query_tokens >= 1 and total_specific_hits == 0:
        score = min(score, 12.0)

    return min(score, 100.0)


class SmartDialecticBot:
    def answer_query(self, user_query: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
        user_query = (user_query or "").strip()
        if not user_query:
            return {
                "status": "empty",
                "message": "Please enter a question or topic to search.",
            }

        query_lang = detect_language(user_query)
        query_norm = normalize_text(user_query)

        candidates = search_qa(query=user_query, limit=200, db_path=db_path)
        if not candidates:
            candidates = search_qa(query="", limit=250, db_path=db_path)

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

        stopwords = ARABIC_STOPWORDS if query_lang == "ar" else ENGLISH_STOPWORDS
        q_toks, _ = _tokenize_with_stems(user_query, stopwords, query_lang)
        specific_query_stems = {
            _stem_token(t, query_lang)
            for t in q_toks
            if t not in _UBIQUITOUS_TOKENS and _stem_token(t, query_lang) not in _UBIQUITOUS_TOKENS
        }

        def _has_specific_token_overlap(cand: Dict[str, Any]) -> bool:
            if not specific_query_stems:
                return True
            cand_text = " ".join([
                cand.get("question_ar" if query_lang == "ar" else "question_en") or "",
                cand.get("core_answer_summary_ar" if query_lang == "ar" else "core_answer_summary_en") or "",
                cand.get("topic_subcategory") or "",
                " ".join(str(tg) for tg in (cand.get("tags") or [])),
            ])
            _, cand_stems = _tokenize_with_stems(cand_text, stopwords, query_lang)
            return bool(specific_query_stems & cand_stems)

        scored_candidates = []
        for c in candidates:
            sc = compute_match_score(query_norm, c, query_lang)
            if (
                intent_category
                and c.get("topic_category") == intent_category
                and sc >= 10.0
                and _has_specific_token_overlap(c)
            ):
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
                        "category": (
                            CATEGORY_TRANSLATIONS.get(r["topic_category"], get_category_arabic_name(r["topic_category"]))
                            if query_lang == "ar"
                            else r["topic_category"]
                        ),
                    }
                    for r in related
                ],
            }

        if scored_candidates and top_score >= 20.0:
            best_match = scored_candidates[0][1]
            return {
                "status": "partial_match",
                "query": user_query,
                "language": query_lang,
                "confidence_score": round(top_score, 2),
                "message": (
                    "لم نعثر على تطابق حرفي تام مع سؤالك في الحجج المستخرجة حتى الآن، ولكن هذه أقرب محاورة موثقة ذات صلة:"
                    if query_lang == "ar"
                    else "No exact wording match was found in the extracted dossiers yet, so here is the closest documented exchange:"
                ),
                "matched_qa": best_match,
                "response_package": self._format_response_package(best_match, query_lang),
                "related_questions": [],
            }

        # Fallback check: search diarized Stage 1 scenario turns if Stage 2 Q&As don't cover it yet
        scen_hits = search_scenario_turns(query=user_query, limit=4, db_path=db_path).get("turns", [])

        if not candidates:
            msg = (
                "قاعدة البيانات مهيأة حالياً للجيل الثاني (V2) المبني على عزل الأصوات وتمييز المتحدثين وفصل تداخل الكلام لـ 2,640 فيديو. "
                "شغّل دفعات المعالجة الذكية على Google Colab ثم زامن ملف dawah_knowledge.db لتظهر الحجج والنصوص الكاملة هنا."
                if query_lang == "ar"
                else "The database is initialized on the clean V2 multi-stage speaker-diarization schema (2,640 videos cataloged). "
                "Run Google Colab smart batches and sync dawah_knowledge.db to populate verified Claim-and-Proof dossiers."
            )
        else:
            msg = (
                "لم نعثر حتى الآن على ملف حجة مستخرج يتناول هذا السؤال تحديداً في الدفعات المنجزة حالياً. "
                "يمكنك البحث في تبويب «نصوص المناظرات الكاملة» أو تجربة موضوعات مثل: وجود الله والسببية، حفظ القرآن، أو مناقشة التثليث."
                if query_lang == "ar"
                else "No extracted dossier matches this exact question in the completed batches yet. "
                "Try searching the Scenario Transcripts tab or exploring topics such as God's existence, textual preservation, or the Trinity."
            )

        related_list = (
            [
                {
                    "id": c["id"],
                    "question": c["question_ar"] if query_lang == "ar" else c["question_en"],
                    "category": (
                        CATEGORY_TRANSLATIONS.get(c["topic_category"], get_category_arabic_name(c["topic_category"]))
                        if query_lang == "ar"
                        else c["topic_category"]
                    ),
                }
                for sc, c in scored_candidates[:3]
                if sc >= 12.0
            ]
            if scored_candidates
            else []
        )

        return {
            "status": "no_match",
            "query": user_query,
            "language": query_lang,
            "confidence_score": round(top_score, 2) if related_list else 0.0,
            "message": msg,
            "matched_qa": None,
            "response_package": None,
            "scenario_matches": scen_hits,
            "related_questions": related_list,
        }

    def _format_response_package(self, qa: Dict[str, Any], lang: str) -> Dict[str, Any]:
        is_ar = lang == "ar"
        raw_premises_en = [str(p).strip() for p in (qa.get("logical_premises") or []) if str(p).strip()]
        raw_premises_ar = [str(p).strip() for p in (qa.get("logical_premises_ar") or []) if str(p).strip()]
        premises = raw_premises_ar if (is_ar and raw_premises_ar) else raw_premises_en

        empty_sentinels = {"none", "none used", "none used.", "none.", "n/a", ""}

        def _clean_opt(val: Any) -> str:
            s = str(val or "").strip()
            return "" if s.lower() in empty_sentinels else s

        analogy_en = _clean_opt(qa.get("analogy_or_metaphor"))
        analogy_ar = _clean_opt(qa.get("analogy_or_metaphor_ar"))
        socratic_en = _clean_opt(qa.get("socratic_question"))
        socratic_ar = _clean_opt(qa.get("socratic_question_ar"))
        verbatim_en = _clean_opt(qa.get("verbatim_quote"))
        verbatim_ar = _clean_opt(qa.get("verbatim_quote_ar"))
        inter_verbatim_en = _clean_opt(qa.get("interlocutor_verbatim_quote"))
        inter_verbatim_ar = _clean_opt(qa.get("interlocutor_verbatim_quote_ar"))
        fallacy_en = _clean_opt(qa.get("fallacy_or_misconception_exposed"))
        fallacy_ar = _clean_opt(qa.get("fallacy_or_misconception_exposed_ar"))

        analogy = analogy_ar if (is_ar and analogy_ar) else (analogy_en or analogy_ar)
        socratic = socratic_ar if (is_ar and socratic_ar) else (socratic_en or socratic_ar)
        verbatim = verbatim_ar if (is_ar and verbatim_ar) else (verbatim_en or verbatim_ar)
        inter_verbatim = inter_verbatim_ar if (is_ar and inter_verbatim_ar) else (inter_verbatim_en or inter_verbatim_ar)

        raw_cat = qa.get("topic_category") or ""
        cat_ar = CATEGORY_TRANSLATIONS.get(raw_cat, get_category_arabic_name(raw_cat))
        cat = cat_ar if is_ar else raw_cat
        raw_inter = qa.get("interlocutor_type") or ""
        inter_ar = INTERLOCUTOR_TRANSLATIONS.get(raw_inter, raw_inter)
        interlocutor = inter_ar if is_ar else raw_inter
        daee_name = (
            qa.get("daee_name_ar") or qa.get("daee_name_en")
            if is_ar
            else qa.get("daee_name_en") or qa.get("daee_name_ar")
        )

        return {
            "id": qa.get("id"),
            "video_id": qa.get("video_id"),
            "episode_index": qa.get("episode_index", 1),
            "title": qa.get("question_ar" if is_ar else "question_en"),
            "title_en": qa.get("question_en") or "",
            "title_ar": qa.get("question_ar") or "",
            "daee_name": daee_name or ("الداعية المحاور" if is_ar else "Main Da'ee"),
            "daee_name_en": qa.get("daee_name_en") or "",
            "daee_name_ar": qa.get("daee_name_ar") or "",
            "interlocutor": interlocutor,
            "interlocutor_en": raw_inter,
            "interlocutor_ar": inter_ar,
            "category": cat,
            "category_en": raw_cat,
            "category_ar": cat_ar,
            "summary": qa.get("core_answer_summary_ar" if is_ar else "core_answer_summary_en"),
            "summary_en": qa.get("core_answer_summary_en") or "",
            "summary_ar": qa.get("core_answer_summary_ar") or "",
            "logical_premises": premises,
            "logical_premises_en": raw_premises_en,
            "logical_premises_ar": raw_premises_ar,
            "fallacy_exposed": fallacy_ar if (is_ar and fallacy_ar) else (fallacy_en or fallacy_ar),
            "fallacy_exposed_en": fallacy_en,
            "fallacy_exposed_ar": fallacy_ar,
            "analogy": analogy,
            "analogy_en": analogy_en,
            "analogy_ar": analogy_ar,
            "socratic_question": socratic,
            "socratic_question_en": socratic_en,
            "socratic_question_ar": socratic_ar,
            "interlocutor_verbatim_quote": inter_verbatim,
            "interlocutor_verbatim_quote_en": inter_verbatim_en,
            "interlocutor_verbatim_quote_ar": inter_verbatim_ar,
            "verbatim_quote": verbatim,
            "verbatim_quote_en": verbatim_en,
            "verbatim_quote_ar": verbatim_ar,
            "scriptural_citations": qa.get("scriptural_citations") or [],
            "dialogue_exchange": qa.get("dialogue_exchange") or [],
            "source_turn_ids": qa.get("source_turn_ids") or [],
            "video_title": qa.get("video_title_ar"),
            "timestamp": qa.get("timestamp_formatted"),
            "timestamp_seconds": qa.get("timestamp_seconds") or 0,
            "answer_timestamp": qa.get("answer_timestamp_formatted") or qa.get("timestamp_formatted"),
            "answer_timestamp_seconds": (
                qa.get("answer_timestamp_seconds")
                if qa.get("answer_timestamp_seconds") is not None
                else (qa.get("timestamp_seconds") or 0)
            ),
            "timestamp_url": qa.get("timestamp_url"),
        }


smart_bot = SmartDialecticBot()
