/**
 * web/static/js/app.js
 * Dawah Dialectics — Bilingual Claim-and-Proof Dossier & Smart Assistant Engine
 */

const PAGE_SIZE = 20;

let currentLang = "en";
let isParallelMode = true;
let currentCategory = "all";
let currentInterlocutor = "all";
let currentSort = "default";
let currentOffset = 0;
let currentTotal = 0;
let loadedCards = [];
let cardParallelOverrides = new Set();
let categoriesData = [];
let interlocutorsData = [];
let corpusStats = { total_qa: 607, total_videos: 2348, categories_count: 10 };
let searchDebounceTimeout = null;

const CATEGORY_MAP = {
  "Prophethood & Quran": { en: "Prophethood & Quran", ar: "النبوة والقرآن الكريم" },
  "Comparative Religion & Christianity": { en: "Comparative Religion & Trinity", ar: "مقارنة الأديان والتثليث" },
  "Theology & God's Existence": { en: "Theology & God's Existence", ar: "العقيدة ووجود الله" },
  "Epistemology & Science": { en: "Epistemology & Science", ar: "المعرفة والعلم التجريبي" },
  "Shariah & Social Inquiries": { en: "Shariah & Social Inquiries", ar: "الشريعة والشبهات الاجتماعية" },
  "Ethics & Morality": { en: "Ethics & Morality", ar: "الأخلاق والقيم" },
  "Objective Morality & Ethics": { en: "Objective Morality & Ethics", ar: "الأخلاق الموضوعية والإلحاد" },
  "Problem of Evil & Suffering": { en: "Problem of Evil & Suffering", ar: "مشكلة الشر والمعاناة" },
  "Dawah & Social Inquiries": { en: "Dawah & Social Inquiries", ar: "الدعوة والحوار الاجتماعي" },
  "Miscellaneous": { en: "General Philosophical Inquiries", ar: "مباحث فكرية عامة" }
};

const INTERLOCUTOR_MAP = {
  "Inquiring Seeker": { en: "Inquiring Seeker", ar: "باحث عن الحق" },
  "Christian / Trinitarian": { en: "Christian / Trinitarian", ar: "مسيحي / تثليثي" },
  "Atheist / Materialist": { en: "Atheist / Materialist", ar: "ملحد / مادي" },
  "Secular Humanist": { en: "Secular Humanist", ar: "علماني إنسانوي" },
  "Materialist": { en: "Materialist", ar: "مادي طبيعي" },
  "Secular Humanist / Christian": { en: "Secular / Christian", ar: "علماني / مسيحي" },
  "Agnostic / Skeptic": { en: "Agnostic / Skeptic", ar: "لاأدري / متشكك" },
  "Muslim with Doubts": { en: "Muslim Seeking Proof", ar: "مسلم يطلب البرهان" }
};

const SVG_ICONS = {
  fallacy: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/></svg>`,
  socratic: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>`,
  quoteMark: `<svg class="socratic-quote-glyph" width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M10 7H6a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h2v2a2 2 0 0 1-2 2H5v2h1a4 4 0 0 0 4-4V7zm10 0h-4a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h2v2a2 2 0 0 1-2 2h-1v2h1a4 4 0 0 0 4-4V7z"/></svg>`,
  proof: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>`,
  analogy: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>`,
  verbatim: `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>`,
  play: `<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><polygon points="6 4 20 12 6 20 6 4"/></svg>`,
  copy: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`,
  bot: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`,
  parallel: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16"/></svg>`
};

const translations = {
  en: {
    navTitle: "Dawah Dialectics",
    navSubtitle: "جدليات الدعوة الإسلامية",
    betaBadgeLabel: "BETA CORPUS",
    betaStatusDesc: "Active distillation: {qa} verified dialectical Q&A units indexed from {videos} street-debate transcripts (5.95M spoken words).",
    filterTopicLabel: "Theological Pillars",
    filterInterlocutorLabel: "Interlocutor Archetype",
    resetLabel: "Reset",
    allTopicsLabel: "All Extracted Units",
    allInterlocutorsLabel: "All Interlocutors",
    tabSearch: "Claim & Proof Archive",
    tabChat: "Smart Dialectical Assistant",
    tabStats: "Corpus & Beta Ledger",
    mobileFilterLabel: "Filters",
    searchHeading: "Claim & Proof Dialectical Dossier",
    searchSubheading: "Every theological doubt paired directly with its exposed fallacy, Socratic counter-question, deductive syllogism, and timestamped street-debate evidence.",
    searchPlaceholder: "Search 607 doubts, premises, fallacies (e.g., 'Who created God', 'من خلق الله', 'Trinity')...",
    sortLabel: "Order:",
    sortDefault: "Corpus Index (QA-001 → QA-607)",
    sortTimeAsc: "Debate Timestamp (Earliest)",
    sortTimeDesc: "Debate Timestamp (Deepest)",
    sortCat: "Theological Pillar",
    showingRange: "Showing {start}–{end} of {total} dialectical units",
    noResultsTitle: "No matching dialectical dossiers found",
    noResultsSub: "Try clearing a filter pill or searching with broader theological keywords in English or Arabic.",
    loadMoreLabel: "Load Next 20 Dossiers",
    loadedStatus: "Loaded {loaded} of {total} matching dossiers",
    assistantTitle: "Smart Dialectical Assistant",
    assistantDesc: "Zero-LLM deterministic reasoning engine: retrieves verified syllogisms, exposed fallacies, analogies, Socratic counter-questions, and exact YouTube debate timestamps with zero hallucination.",
    chipHint: "Common Objections & Doubts",
    welcomeText: "Ask any theological objection, philosophical doubt, or inquiry about Islam in English or Arabic. The engine will decompose the objection into its exposed fallacy, Socratic counter-question, numbered deductive premises, concrete analogy, and exact video timestamp.",
    chatPlaceholder: "Type a question or objection in English or Arabic (e.g., 'Who created God?', 'من خلق الله؟')...",
    sendLabel: "Analyze Objection",
    premisesLabel: "Deductive Syllogism & Rational Proof",
    fallacyLabel: "Fallacy Exposed • المغالطة المكشوفة",
    analogyLabel: "Concrete Analogy • المثال التوضيحي",
    socraticLabel: "Socratic Counter-Question • السؤال السقراطي",
    verbatimLabel: "Verbatim Street Dialogue Excerpt",
    watchBtn: "Watch Video Proof at",
    askBotBtn: "Inspect in Assistant",
    copyBtn: "Copy",
    testBotBtn: "Test in Bot",
    parallelBtnLabel: "EN ⇄ عربي",
    copiedToast: "Copied Socratic counter-question to clipboard",
    footerText: "Grounded in deductive reason, textual verification, and timestamped street dialogues.",
    statsTitle: "Corpus Metadata & Beta Extraction Ledger",
    statsSubtitle: "Complete methodological transparency: tracking 2,348 cataloged SCDawah Arabic street-debate transcripts (~5.95M spoken words) through automated dialectical distillation.",
    statQaLabel: "Extracted Q&A Dossiers",
    statVideosLabel: "Cataloged Debate Videos",
    statWordsLabel: "Spoken Words Preserved",
    statPillarsLabel: "Theological Pillars Indexed",
    ledgerCatHeading: "Distribution by Theological Pillar",
    ledgerCatSub: "Click any pillar row to filter the Claim & Proof Archive directly.",
    ledgerInterHeading: "Distribution by Interlocutor Archetype",
    ledgerInterSub: "Real street-debate interlocutor profiles across extracted units.",
    colabTitle: "Open-Source Google Colab Extraction Pipeline",
    colabDesc: "This repository is in active Beta extraction. Run checkpointed batch distillation across all 2,348 subtitle files using dawah_knowledge_extractor_colab.ipynb or run_colab_all_in_one.py with zero data loss.",
    chips: [
      { text: "If God created the universe, who created God?", query: "Who created God?" },
      { text: "Why does God allow innocent suffering & evil?", query: "Why does God allow innocent children to suffer?" },
      { text: "How do we logically examine the Trinity?", query: "How to refute the Trinity?" },
      { text: "Can atheism ground objective moral duties?", query: "Can atheists have objective morality?" },
      { text: "Is scientism self-refuting as a theory of truth?", query: "Why believe in what science cannot observe?" },
      { text: "Textual preservation of the Quran vs. Bible", query: "How do we know the Quran was preserved?" },
      { text: "Women's rights and social laws in Islam", query: "Does Islam oppress women?" }
    ]
  },
  ar: {
    navTitle: "جدليات الدعوة الإسلامية",
    navSubtitle: "Dawah Dialectics Archive",
    betaBadgeLabel: "أرشيف بيتا",
    betaStatusDesc: "استخراج نشط: تم تحرير وفهرسة {qa} وحدة حوارية برهانية من أصل {videos} مناظرة مسجلة (5.95 مليون كلمة منطوقة).",
    filterTopicLabel: "الأبواب الفكرية والعقدية",
    filterInterlocutorLabel: "نمط المحاور في الميدان",
    resetLabel: "إعادة ضبط",
    allTopicsLabel: "كافة الوحدات المستخرجة",
    allInterlocutorsLabel: "جميع المحاورين",
    tabSearch: "ديوان الشبهات والبراهين",
    tabChat: "المساعد الجدلي الذكي",
    tabStats: "سجل الأرشيف والإحصائيات",
    mobileFilterLabel: "التصنيفات",
    searchHeading: "ملفات تفكيك الشبهات والبرهان العقلي",
    searchSubheading: "كل شبهة أو سؤال فلسفي مقترن مباشرة بكشف المغالطة المنطقية، والسؤال السقراطي المعاكس، والقياس العقلي، وتوثيق الفيديو بالثانية.",
    searchPlaceholder: "ابحث في 607 وحدة حوارية (مثال: 'من خلق الله'، 'مشكلة الشر'، 'التثليث'، 'حفظ القرآن')...",
    sortLabel: "الترتيب:",
    sortDefault: "التسلسل المرجعي (QA-001 ← QA-607)",
    sortTimeAsc: "توقيت المناظرة (الأبكر)",
    sortTimeDesc: "توقيت المناظرة (الأعمق)",
    sortCat: "الباب الموضوعي",
    showingRange: "عرض {start}–{end} من أصل {total} وحدة حوارية",
    noResultsTitle: "لم يتم العثور على ملفات حوارية مطابقة",
    noResultsSub: "جرّب إلغاء بعض الفلاتر النشطة أو البحث بكلمات مفتاحية أوسع بالعربية أو الإنجليزية.",
    loadMoreLabel: "تحميل 20 ملفاً إضافياً",
    loadedStatus: "تم عرض {loaded} من أصل {total} وحدة مطابقة",
    assistantTitle: "المساعد الحواري الذكي الفوري",
    assistantDesc: "محرك استدلالي فوري (بدون هلوسة): يستخرج الأقيسة المنطقية، والمغالطات المكشوفة، والأمثلة التوضيحية، والأسئلة السقراطية، وروابط الفيديو الدقيقة.",
    chipHint: "أبرز الشبهات والأسئلة المتكررة",
    welcomeText: "اطرح أي شبهة فلسفية أو تساؤل عقدي بالعربية أو الإنجليزية؛ سيقوم المحرك بتفكيك الشبهة إلى المغالطة الكامنة، والسؤال السقراطي المعاكس، والمقدمات المنطقية المتسلسلة، والمثال الواقعي مع رابط اللحظة الدقيقة في المناظرة.",
    chatPlaceholder: "اكتب سؤالك أو الشبهة هنا بالعربية أو الإنجليزية (مثال: 'من خلق الله؟'، 'التثليث')...",
    sendLabel: "تحليل الشبهة",
    premisesLabel: "التسلسل البرهاني الاستنتاجي • Deductive Syllogism",
    fallacyLabel: "المغالطة المنطقية المكشوفة • Fallacy Exposed",
    analogyLabel: "المثال التوضيحي والقياس • Concrete Analogy",
    socraticLabel: "السؤال السقراطي المعاكس • Socratic Turn",
    verbatimLabel: "نص الحوار الحرفي من الميدان",
    watchBtn: "شاهد البرهان في الدقيقة",
    askBotBtn: "تحليل في المساعد",
    copyBtn: "نسخ",
    testBotBtn: "اختبر في المساعد",
    parallelBtnLabel: "عربي ⇄ EN",
    copiedToast: "تم نسخ السؤال السقراطي إلى الحافظة",
    footerText: "قائم على البرهان العقلي، والتوثيق النصي، والمناظرات الميدانية الموثقة باللحظة.",
    statsTitle: "سجل بيانات الأرشيف والاستخراج المرحلي (Beta)",
    statsSubtitle: "شفافية منهجية كاملة: فهرسة وتتبع 2,348 ملف ترجمة لمناظرات ركن المتحدثين (~5.95 مليون كلمة) عبر خط استخراج برهاني مهيكل.",
    statQaLabel: "ملف حواري محرر ومفهرس",
    statVideosLabel: "فيديو مناظرة في الأرشيف",
    statWordsLabel: "كلمة منطوقة محفوظة",
    statPillarsLabel: "أبواب فكرية وعقدية",
    ledgerCatHeading: "توزيع الوحدات حسب الأبواب الفكرية",
    ledgerCatSub: "اضغط على أي باب للانتقال المباشر إلى ملفاته في ديوان البراهين.",
    ledgerInterHeading: "توزيع الوحدات حسب خلفية المحاور",
    ledgerInterSub: "أنماط المحاورين الفعلية في مناظرات الشارع المستخرجة.",
    colabTitle: "خط الاستخراج المفتوح عبر Google Colab",
    colabDesc: "الأرشيف في مرحلة استخراج نشطة (Beta). يمكنك تشغيل الاستخراج الآلي مع الحفظ التلقائي عبر dawah_knowledge_extractor_colab.ipynb أو run_colab_all_in_one.py.",
    chips: [
      { text: "إذا كان الله قد خلق الكون، فمن خلق الله؟", query: "من خلق الله؟" },
      { text: "لماذا يسمح الله بالشر ومعاناة الأطفال؟", query: "لماذا يسمح الله بمعاناة الأطفال؟" },
      { text: "كيف نناقش عقيدة التثليث عقلياً؟", query: "كيف نرد على عقيدة التثليث؟" },
      { text: "هل يمكن تأسيس أخلاق موضوعية بدون الله؟", query: "هل يستطيع الملحد تأسيس أخلاق موضوعية؟" },
      { text: "الرد على حصر الحقيقة في العلم التجريبي", query: "لماذا نؤمن بما لا يثبته العلم التجريبي؟" },
      { text: "حفظ القرآن الكريم مقارنة بالمخطوطات", query: "تحريف الانجيل وحفظ القران" },
      { text: "الشبهات حول حقوق المرأة في الإسلام", query: "هل يظلم الإسلام المرأة؟" }
    ]
  }
};

document.addEventListener("DOMContentLoaded", () => {
  initLanguagePreference();
  setupKeyboardShortcuts();
  renderLanguageUI();
  loadCorpusMetadata();
  performSearch(true);
});

function initLanguagePreference() {
  const savedLang = localStorage.getItem("dawah_lang");
  const savedParallel = localStorage.getItem("dawah_parallel_mode");

  if (savedLang === "ar" || savedLang === "en") {
    currentLang = savedLang;
  } else {
    const browserLang = (navigator.language || navigator.userLanguage || "en").toLowerCase();
    currentLang = browserLang.startsWith("ar") ? "ar" : "en";
  }

  if (savedParallel !== null) {
    isParallelMode = savedParallel === "true";
  } else {
    isParallelMode = true;
  }

  applyDocumentDirection();
}

function applyDocumentDirection() {
  document.documentElement.setAttribute("dir", currentLang === "ar" ? "rtl" : "ltr");
  document.documentElement.setAttribute("lang", currentLang);
}

function setLanguageMode(lang, parallel) {
  if (lang) {
    currentLang = lang;
    localStorage.setItem("dawah_lang", currentLang);
  }
  isParallelMode = Boolean(parallel);
  localStorage.setItem("dawah_parallel_mode", String(isParallelMode));
  cardParallelOverrides.clear();

  applyDocumentDirection();
  renderLanguageUI();
  renderFacetRails();
  renderActiveFilterChips();
  renderCards(loadedCards);
  renderStatsTables();
}

function updateSegmentedButtons() {
  const btnEn = document.getElementById("mode-btn-en");
  const btnAr = document.getElementById("mode-btn-ar");
  const btnParallel = document.getElementById("mode-btn-parallel");
  if (!btnEn || !btnAr || !btnParallel) return;

  btnEn.classList.toggle("active", !isParallelMode && currentLang === "en");
  btnAr.classList.toggle("active", !isParallelMode && currentLang === "ar");
  btnParallel.classList.toggle("active", isParallelMode);
}

function setupKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    if ((e.key === "/" || (e.ctrlKey && e.key.toLowerCase() === "k")) &&
        document.activeElement.tagName !== "INPUT" &&
        document.activeElement.tagName !== "TEXTAREA") {
      e.preventDefault();
      switchTab("search-tab");
      const input = document.getElementById("search-input");
      if (input) input.focus();
    }
  });
}

function renderLanguageUI() {
  const t = translations[currentLang];
  updateSegmentedButtons();

  setText("nav-title", t.navTitle);
  setText("nav-subtitle", t.navSubtitle);
  setText("beta-badge-label", t.betaBadgeLabel);

  const betaDescEl = document.getElementById("beta-status-desc");
  if (betaDescEl) {
    betaDescEl.innerText = t.betaStatusDesc
      .replace("{qa}", corpusStats.total_qa.toLocaleString())
      .replace("{videos}", corpusStats.total_videos.toLocaleString());
  }

  setText("filter-topic-label", t.filterTopicLabel);
  setText("filter-interlocutor-label", t.filterInterlocutorLabel);
  setText("reset-filters-btn", t.resetLabel);
  setText("tab-search-label", t.tabSearch);
  setText("tab-chat-label", t.tabChat);
  setText("tab-stats-label", t.tabStats);
  setText("mobile-filter-label", t.mobileFilterLabel);
  setText("search-heading", t.searchHeading);
  setText("search-subheading", t.searchSubheading);

  const searchInput = document.getElementById("search-input");
  if (searchInput) searchInput.placeholder = t.searchPlaceholder;

  setText("sort-label", t.sortLabel);
  setText("sort-opt-default", t.sortDefault);
  setText("sort-opt-time-asc", t.sortTimeAsc);
  setText("sort-opt-time-desc", t.sortTimeDesc);
  setText("sort-opt-cat", t.sortCat);
  setText("load-more-label", t.loadMoreLabel);

  setText("assistant-title", t.assistantTitle);
  setText("assistant-desc", t.assistantDesc);
  setText("chip-hint", t.chipHint);
  setText("welcome-text", t.welcomeText);

  const chatInput = document.getElementById("chat-input");
  if (chatInput) chatInput.placeholder = t.chatPlaceholder;

  setText("send-label", t.sendLabel);
  setText("footer-text", t.footerText);

  setText("stats-title", t.statsTitle);
  setText("stats-subtitle", t.statsSubtitle);
  setText("stat-qa-label", t.statQaLabel);
  setText("stat-videos-label", t.statVideosLabel);
  setText("stat-words-label", t.statWordsLabel);
  setText("stat-pillars-label", t.statPillarsLabel);
  setText("ledger-cat-heading", t.ledgerCatHeading);
  setText("ledger-cat-sub", t.ledgerCatSub);
  setText("ledger-inter-heading", t.ledgerInterHeading);
  setText("ledger-inter-sub", t.ledgerInterSub);
  setText("colab-title", t.colabTitle);
  setText("colab-desc", t.colabDesc);

  const chipsContainer = document.getElementById("prompt-chips-container");
  if (chipsContainer) {
    chipsContainer.innerHTML = t.chips.map(chip => `
      <button type="button" class="starter-chip-btn" onclick="askAssistant('${escapeJsStr(chip.query)}')">
        ${escapeHtml(chip.text)}
      </button>
    `).join("");
  }

  updateResultsCounterText();
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.innerText = val;
}

function loadCorpusMetadata() {
  Promise.all([
    fetch("/api/stats").then(r => r.json()),
    fetch("/api/categories").then(r => r.json()),
    fetch("/api/interlocutors").then(r => r.json())
  ])
    .then(([stats, catData, interData]) => {
      corpusStats = stats || corpusStats;
      categoriesData = (catData && catData.categories) || [];
      interlocutorsData = (interData && interData.interlocutors) || [];

      setText("stat-videos", (corpusStats.total_videos || 2348).toLocaleString());
      setText("stat-qa", (corpusStats.total_qa || 607).toLocaleString());
      setText("stat-pillars", (corpusStats.categories_count || 10).toLocaleString());
      setText("nav-qa-count", (corpusStats.total_qa || 607).toLocaleString());
      setText("beta-ratio-num", `${(corpusStats.total_qa || 607).toLocaleString()} / ${(corpusStats.total_videos || 2348).toLocaleString()} (26%)`);

      renderLanguageUI();
      renderFacetRails();
      renderStatsTables();
    })
    .catch(err => console.error("Metadata load error:", err));
}

function getLocalizedCategory(catKey, lang = currentLang) {
  if (!catKey) return "";
  return (CATEGORY_MAP[catKey] && CATEGORY_MAP[catKey][lang]) || catKey;
}

function getLocalizedInterlocutor(interKey, lang = currentLang) {
  if (!interKey) return "";
  return (INTERLOCUTOR_MAP[interKey] && INTERLOCUTOR_MAP[interKey][lang]) || interKey;
}

function renderFacetRails() {
  const t = translations[currentLang];
  const catContainer = document.getElementById("category-pills");
  if (catContainer) {
    const totalQa = corpusStats.total_qa || 607;
    catContainer.innerHTML = `
      <button type="button" class="facet-btn ${currentCategory === 'all' ? 'active' : ''}" onclick="setCategoryFilter('all')">
        <span>${escapeHtml(t.allTopicsLabel)}</span>
        <span class="facet-count">${totalQa}</span>
      </button>
      ${categoriesData.map(item => `
        <button type="button" class="facet-btn ${currentCategory === item.topic_category ? 'active' : ''}" onclick="setCategoryFilter('${escapeJsStr(item.topic_category)}')">
          <span>${escapeHtml(getLocalizedCategory(item.topic_category))}</span>
          <span class="facet-count">${item.count}</span>
        </button>
      `).join("")}
    `;
  }

  const interContainer = document.getElementById("interlocutor-pills");
  if (interContainer) {
    const totalQa = corpusStats.total_qa || 607;
    interContainer.innerHTML = `
      <button type="button" class="facet-btn ${currentInterlocutor === 'all' ? 'active' : ''}" onclick="setInterlocutorFilter('all')">
        <span>${escapeHtml(t.allInterlocutorsLabel)}</span>
        <span class="facet-count">${totalQa}</span>
      </button>
      ${interlocutorsData.map(item => `
        <button type="button" class="facet-btn ${currentInterlocutor === item.interlocutor_type ? 'active' : ''}" onclick="setInterlocutorFilter('${escapeJsStr(item.interlocutor_type)}')">
          <span>${escapeHtml(getLocalizedInterlocutor(item.interlocutor_type))}</span>
          <span class="facet-count">${item.count}</span>
        </button>
      `).join("")}
    `;
  }
}

function renderStatsTables() {
  const catTable = document.getElementById("stats-categories-table");
  const totalQa = Math.max(corpusStats.total_qa || 607, 1);

  if (catTable && categoriesData.length) {
    catTable.innerHTML = categoriesData.map(item => {
      const pct = Math.max(Math.round((item.count / totalQa) * 100), 2);
      return `
        <button type="button" class="arch-bar-row" onclick="jumpToCategoryFromStats('${escapeJsStr(item.topic_category)}')">
          <div class="arch-bar-top">
            <span>${escapeHtml(getLocalizedCategory(item.topic_category))}</span>
            <span class="arch-bar-count">${item.count} (${pct}%)</span>
          </div>
          <div class="arch-bar-track">
            <div class="arch-bar-fill" style="width: ${pct}%;"></div>
          </div>
        </button>
      `;
    }).join("");
  }

  const interTable = document.getElementById("stats-interlocutors-table");
  if (interTable && interlocutorsData.length) {
    interTable.innerHTML = interlocutorsData.map(item => {
      const pct = Math.max(Math.round((item.count / totalQa) * 100), 2);
      return `
        <button type="button" class="arch-bar-row" onclick="jumpToInterlocutorFromStats('${escapeJsStr(item.interlocutor_type)}')">
          <div class="arch-bar-top">
            <span>${escapeHtml(getLocalizedInterlocutor(item.interlocutor_type))}</span>
            <span class="arch-bar-count">${item.count} (${pct}%)</span>
          </div>
          <div class="arch-bar-track">
            <div class="arch-bar-fill" style="width: ${pct}%;"></div>
          </div>
        </button>
      `;
    }).join("");
  }
}

function jumpToCategoryFromStats(cat) {
  currentCategory = cat;
  switchTab("search-tab");
  renderFacetRails();
  performSearch(true);
}

function jumpToInterlocutorFromStats(inter) {
  currentInterlocutor = inter;
  switchTab("search-tab");
  renderFacetRails();
  performSearch(true);
}

function switchTab(tabId) {
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));

  const pane = document.getElementById(tabId);
  if (pane) pane.classList.add("active");

  const activeBtn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
  if (activeBtn) activeBtn.classList.add("active");
}

function toggleMobileRail(open) {
  const rail = document.getElementById("facet-rail");
  if (rail) rail.classList.toggle("open", Boolean(open));
}

function setCategoryFilter(cat) {
  currentCategory = cat;
  renderFacetRails();
  toggleMobileRail(false);
  switchTab("search-tab");
  performSearch(true);
}

function setInterlocutorFilter(inter) {
  currentInterlocutor = inter;
  renderFacetRails();
  toggleMobileRail(false);
  switchTab("search-tab");
  performSearch(true);
}

function resetAllFilters() {
  currentCategory = "all";
  currentInterlocutor = "all";
  currentSort = "default";
  const searchInput = document.getElementById("search-input");
  if (searchInput) searchInput.value = "";
  const sortSelect = document.getElementById("sort-select");
  if (sortSelect) sortSelect.value = "default";
  const clearBtn = document.getElementById("clear-search-btn");
  if (clearBtn) clearBtn.hidden = true;

  renderFacetRails();
  performSearch(true);
}

function clearSearchInput() {
  const searchInput = document.getElementById("search-input");
  if (searchInput) {
    searchInput.value = "";
    searchInput.focus();
  }
  const clearBtn = document.getElementById("clear-search-btn");
  if (clearBtn) clearBtn.hidden = true;
  performSearch(true);
}

function handleSearchInput() {
  const searchInput = document.getElementById("search-input");
  const clearBtn = document.getElementById("clear-search-btn");
  if (clearBtn && searchInput) {
    clearBtn.hidden = searchInput.value.trim().length === 0;
  }

  switchTab("search-tab");
  clearTimeout(searchDebounceTimeout);
  searchDebounceTimeout = setTimeout(() => {
    performSearch(true);
  }, 180);
}

function handleSortChange() {
  const sortSelect = document.getElementById("sort-select");
  if (sortSelect) {
    currentSort = sortSelect.value;
    performSearch(true);
  }
}

function performSearch(resetOffset = true) {
  if (resetOffset) {
    currentOffset = 0;
    loadedCards = [];
  }

  const query = (document.getElementById("search-input")?.value || "").trim();
  const url = `/api/search?q=${encodeURIComponent(query)}&category=${encodeURIComponent(currentCategory)}&interlocutor=${encodeURIComponent(currentInterlocutor)}&sort=${encodeURIComponent(currentSort)}&limit=${PAGE_SIZE}&offset=${currentOffset}`;

  fetch(url)
    .then(r => r.json())
    .then(data => {
      const batch = data.results || [];
      if (resetOffset && query && batch.length === 0 && (currentCategory !== "all" || currentInterlocutor !== "all")) {
        currentCategory = "all";
        currentInterlocutor = "all";
        renderFacetRails();
        performSearch(true);
        return;
      }
      currentTotal = typeof data.total === "number" ? data.total : batch.length;
      if (resetOffset) {
        loadedCards = batch;
      } else {
        loadedCards = loadedCards.concat(batch);
      }
      renderActiveFilterChips();
      updateResultsCounterText();
      renderCards(loadedCards);
    })
    .catch(err => console.error("Search error:", err));
}

function loadMoreResults() {
  currentOffset += PAGE_SIZE;
  performSearch(false);
}

function updateResultsCounterText() {
  const t = translations[currentLang];
  const counterEl = document.getElementById("results-count-text");
  if (counterEl) {
    const start = loadedCards.length ? 1 : 0;
    const end = loadedCards.length;
    counterEl.innerText = t.showingRange
      .replace("{start}", start.toLocaleString())
      .replace("{end}", end.toLocaleString())
      .replace("{total}", currentTotal.toLocaleString());
  }

  const pagBar = document.getElementById("pagination-bar");
  const pagStatus = document.getElementById("pagination-status-text");
  const loadMoreBtn = document.getElementById("load-more-btn");

  if (pagBar && pagStatus && loadMoreBtn) {
    if (currentTotal > 0) {
      pagBar.hidden = false;
      pagStatus.innerText = t.loadedStatus
        .replace("{loaded}", loadedCards.length.toLocaleString())
        .replace("{total}", currentTotal.toLocaleString());
      loadMoreBtn.hidden = loadedCards.length >= currentTotal;
    } else {
      pagBar.hidden = true;
    }
  }
}

function renderActiveFilterChips() {
  const container = document.getElementById("active-filter-chips");
  if (!container) return;

  const chips = [];
  if (currentCategory !== "all") {
    chips.push(`
      <button type="button" class="filter-tag-chip" onclick="setCategoryFilter('all')">
        <span>${escapeHtml(getLocalizedCategory(currentCategory))}</span>
        <span aria-hidden="true">×</span>
      </button>
    `);
  }
  if (currentInterlocutor !== "all") {
    chips.push(`
      <button type="button" class="filter-tag-chip" onclick="setInterlocutorFilter('all')">
        <span>${escapeHtml(getLocalizedInterlocutor(currentInterlocutor))}</span>
        <span aria-hidden="true">×</span>
      </button>
    `);
  }
  const query = (document.getElementById("search-input")?.value || "").trim();
  if (query) {
    chips.push(`
      <button type="button" class="filter-tag-chip" onclick="clearSearchInput()">
        <span>"${escapeHtml(query)}"</span>
        <span aria-hidden="true">×</span>
      </button>
    `);
  }
  container.innerHTML = chips.join("");
}

function isNoneText(str) {
  if (!str) return true;
  const s = String(str).trim().toLowerCase();
  return ["none", "none used", "none used.", "none.", "n/a", "no", "not applicable", "لا يوجد"].includes(s);
}

function toggleCardParallel(cardId) {
  if (cardParallelOverrides.has(cardId)) {
    cardParallelOverrides.delete(cardId);
  } else {
    cardParallelOverrides.add(cardId);
  }
  renderCards(loadedCards);
}

function copySocraticText(text) {
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const toast = document.getElementById("copy-toast");
    if (toast) {
      toast.innerText = translations[currentLang].copiedToast;
      toast.hidden = false;
      setTimeout(() => {
        toast.hidden = true;
      }, 2200);
    }
  });
}

function filterByTag(tagText) {
  const searchInput = document.getElementById("search-input");
  if (searchInput) {
    searchInput.value = tagText;
    const clearBtn = document.getElementById("clear-search-btn");
    if (clearBtn) clearBtn.hidden = false;
    switchTab("search-tab");
    performSearch(true);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

function renderCards(cards) {
  const container = document.getElementById("qa-cards-container");
  if (!container) return;
  container.innerHTML = "";

  const t = translations[currentLang];
  const isAr = currentLang === "ar";

  if (!cards.length) {
    container.innerHTML = `
      <div class="empty-dossier-state">
        <h3>${escapeHtml(t.noResultsTitle)}</h3>
        <p>${escapeHtml(t.noResultsSub)}</p>
        <button type="button" class="btn-primary-emerald" onclick="resetAllFilters()">${escapeHtml(t.resetLabel)}</button>
      </div>
    `;
    return;
  }

  cards.forEach((card, index) => {
    const cardId = card.id || `qa_${index + 1}`;
    const serialCode = `QA-${String(card.serial_number || index + 1).padStart(3, "0")}`;
    const showParallel = isParallelMode ? !cardParallelOverrides.has(cardId) : cardParallelOverrides.has(cardId);

    const qEn = card.question_en || card.question_ar || "";
    const qAr = card.question_ar || card.question_en || "";
    const sumEn = card.core_answer_summary_en || card.core_answer_summary_ar || "";
    const sumAr = card.core_answer_summary_ar || card.core_answer_summary_en || "";

    const primaryQuestion = isAr ? qAr : qEn;
    const primarySummary = isAr ? sumAr : sumEn;

    const categoryName = getLocalizedCategory(card.topic_category);
    const interlocutorName = getLocalizedInterlocutor(card.interlocutor_type);
    const subcategoryText = card.topic_subcategory && !isNoneText(card.topic_subcategory) ? card.topic_subcategory : "";

    // Question Header HTML (Parallel vs Single)
    const questionBannerHtml = showParallel
      ? `
        <div class="question-parallel-grid">
          <div class="q-col-en">
            <h3>${escapeHtml(qEn)}</h3>
          </div>
          <div class="q-col-ar">
            <h3>${escapeHtml(qAr)}</h3>
          </div>
        </div>
      `
      : `
        <div class="question-single">
          <h3>${escapeHtml(primaryQuestion)}</h3>
        </div>
      `;

    // Core Thesis Summary HTML (Parallel vs Single)
    const thesisSummaryHtml = showParallel
      ? `
        <div class="core-thesis-parallel">
          <p class="thesis-en">${escapeHtml(sumEn)}</p>
          <p class="thesis-ar">${escapeHtml(sumAr)}</p>
        </div>
      `
      : `<p class="core-thesis-block">${escapeHtml(primarySummary)}</p>`;

    // Fallacy Exposed
    const fallacyEn = card.fallacy_or_misconception_exposed;
    const fallacyAr = card.fallacy_or_misconception_exposed_ar;
    const activeFallacy = isAr ? (fallacyAr || fallacyEn) : (fallacyEn || fallacyAr);
    let fallacyHtml = "";
    if (activeFallacy && !isNoneText(activeFallacy)) {
      const fallacyBody = showParallel && fallacyEn && fallacyAr && fallacyEn !== fallacyAr
        ? `<div>${escapeHtml(fallacyEn)}</div><div class="ar-font" dir="rtl" style="margin-top:0.35rem; padding-top:0.35rem; border-top:1px dashed rgba(166,43,29,0.2);">${escapeHtml(fallacyAr)}</div>`
        : escapeHtml(activeFallacy);
      fallacyHtml = `
        <div class="fallacy-panel">
          <div class="fallacy-panel-bar">
            ${SVG_ICONS.fallacy}
            <span>${escapeHtml(t.fallacyLabel)}</span>
          </div>
          <div class="fallacy-panel-body">${fallacyBody}</div>
        </div>
      `;
    }

    // Redesigned Socratic Counter-Question ("The Socratic Turn")
    const socraticEn = card.socratic_question;
    const socraticAr = card.socratic_question_ar;
    const activeSocratic = isAr ? (socraticAr || socraticEn) : (socraticEn || socraticAr);
    let socraticHtml = "";
    if (activeSocratic && !isNoneText(activeSocratic)) {
      const socraticBody = showParallel && socraticEn && socraticAr && socraticEn !== socraticAr
        ? `
          <div>“${escapeHtml(socraticEn)}”</div>
          <div class="ar-font" dir="rtl" style="margin-top:0.4rem; padding-top:0.4rem; border-top:1px dashed rgba(138,75,8,0.22); font-style:normal;">«${escapeHtml(socraticAr)}»</div>
        `
        : `“${escapeHtml(activeSocratic)}”`;

      socraticHtml = `
        <div class="socratic-turn-panel">
          <div class="socratic-turn-bar">
            <span class="socratic-badge-title">
              ${SVG_ICONS.socratic}
              <span>${escapeHtml(t.socraticLabel)}</span>
            </span>
            <div class="socratic-micro-actions">
              <button type="button" class="socratic-action-btn" onclick="copySocraticText('${escapeJsStr(activeSocratic)}')">
                ${SVG_ICONS.copy}
                <span>${escapeHtml(t.copyBtn)}</span>
              </button>
              <button type="button" class="socratic-action-btn" onclick="askAssistant('${escapeJsStr(primaryQuestion)}')">
                ${SVG_ICONS.bot}
                <span>${escapeHtml(t.testBotBtn)}</span>
              </button>
            </div>
          </div>
          <div class="socratic-turn-body">
            ${SVG_ICONS.quoteMark}
            <div class="socratic-question-text">${socraticBody}</div>
          </div>
        </div>
      `;
    }

    // Verbatim Street Dialogue Excerpt
    const quoteText = isAr ? (card.verbatim_quote_ar || card.verbatim_quote) : (card.verbatim_quote || card.verbatim_quote_ar);
    let quoteHtml = "";
    if (quoteText && !isNoneText(quoteText)) {
      quoteHtml = `
        <div class="verbatim-strip">
          <div class="verbatim-strip-label">
            ${SVG_ICONS.verbatim}
            <span>${escapeHtml(t.verbatimLabel)} [${escapeHtml(card.timestamp_formatted || "00:00")}]</span>
          </div>
          <div>“${escapeHtml(quoteText)}”</div>
        </div>
      `;
    }

    // Deductive Syllogism Premises
    const cleanPremiseStr = (str) => String(str || "")
      .trim()
      .replace(/^(Premise\s*\d+\s*[:.-]?\s*|Conclusion\s*[:.-]?\s*|المقدمة\s*(الأولى|الثانية|الثالثة|الرابعة|الخامسة|\d+)\s*[:.-]?\s*|النتيجة\s*[:.-]?\s*|الاستنتاج\s*[:.-]?\s*)/i, "");

    const rawPremisesEn = (card.logical_premises || []).map(cleanPremiseStr).filter(p => p.length > 0);
    const rawPremisesAr = (card.logical_premises_ar || []).map(cleanPremiseStr).filter(p => p.length > 0);
    const activePremises = isAr && rawPremisesAr.length ? rawPremisesAr : rawPremisesEn;

    let premisesHtml = "";
    if (activePremises.length) {
      const stepsCountLabel = isAr ? `${activePremises.length} خطوات منطقية` : `${activePremises.length} Logical Steps`;
      premisesHtml = `
        <div>
          <div class="proof-section-bar">
            <span>${SVG_ICONS.proof} ${escapeHtml(t.premisesLabel)}</span>
            <span class="proof-steps-pill">${escapeHtml(stepsCountLabel)}</span>
          </div>
          <ol class="syllogism-list" style="margin-top:0.5rem;">
            ${activePremises.map((p, idx) => {
              const isLast = idx === activePremises.length - 1 && activePremises.length > 1;
              const badgeCode = isLast ? (isAr ? "النتيجة" : "Conclusion") : `${idx + 1}`;
              const badgeClass = isLast ? "step-badge step-badge-conclusion" : "step-badge";
              const pEnItem = rawPremisesEn[idx] || p;
              const pArItem = rawPremisesAr[idx] || "";
              const stepContent = showParallel && pArItem && pEnItem !== pArItem
                ? `<div>${escapeHtml(pEnItem)}</div><div class="ar-font" dir="rtl" style="margin-top:0.3rem; padding-top:0.3rem; border-top:1px dashed var(--border-hairline); color:var(--ink-secondary);">${escapeHtml(pArItem)}</div>`
                : escapeHtml(p);
              return `
                <li class="syllogism-step">
                  <span class="${badgeClass}">${badgeCode}</span>
                  <div>${stepContent}</div>
                </li>
              `;
            }).join("")}
          </ol>
        </div>
      `;
    }

    // Concrete Analogy
    const analogyEn = card.analogy_or_metaphor;
    const analogyAr = card.analogy_or_metaphor_ar;
    const activeAnalogy = isAr ? (analogyAr || analogyEn) : (analogyEn || analogyAr);
    let analogyHtml = "";
    if (activeAnalogy && !isNoneText(activeAnalogy)) {
      const analogyBody = showParallel && analogyEn && analogyAr && analogyEn !== analogyAr
        ? `<div>${escapeHtml(analogyEn)}</div><div class="ar-font" dir="rtl" style="margin-top:0.35rem; padding-top:0.35rem; border-top:1px dashed rgba(15,82,62,0.2);">${escapeHtml(analogyAr)}</div>`
        : escapeHtml(activeAnalogy);
      analogyHtml = `
        <div class="analogy-panel">
          <div class="analogy-bar">
            ${SVG_ICONS.analogy}
            <span>${escapeHtml(t.analogyLabel)}</span>
          </div>
          <div class="analogy-body">${analogyBody}</div>
        </div>
      `;
    }

    // Concept Tags
    const tags = Array.isArray(card.tags) ? card.tags.slice(0, 5) : [];
    const tagsHtml = tags.length
      ? `
        <div class="concept-tags-row">
          ${tags.map(tag => `
            <button type="button" class="concept-tag" onclick="filterByTag('${escapeJsStr(tag)}')">#${escapeHtml(tag)}</button>
          `).join("")}
        </div>
      `
      : "";

    const watchTime = card.timestamp_formatted || "00:00";
    const videoUrl = card.timestamp_url || card.youtube_url || "#";

    const article = document.createElement("article");
    article.className = "dossier-card";
    article.innerHTML = `
      <div class="dossier-banner">
        <div class="dossier-meta-bar">
          <div class="meta-badges-left">
            <span class="badge-serial">${escapeHtml(serialCode)}</span>
            <button type="button" class="badge-pillar" onclick="setCategoryFilter('${escapeJsStr(card.topic_category)}')">
              ${escapeHtml(categoryName)}
            </button>
            ${subcategoryText ? `<span class="badge-subcat">· ${escapeHtml(subcategoryText)}</span>` : ""}
            <button type="button" class="badge-interlocutor" onclick="setInterlocutorFilter('${escapeJsStr(card.interlocutor_type)}')">
              ${escapeHtml(interlocutorName)}
            </button>
          </div>
          <button type="button" class="card-lang-peek-btn ${showParallel ? 'active' : ''}" onclick="toggleCardParallel('${escapeJsStr(cardId)}')">
            ${SVG_ICONS.parallel}
            <span>${escapeHtml(t.parallelBtnLabel)}</span>
          </button>
        </div>
        ${questionBannerHtml}
      </div>

      <div class="dossier-split-body">
        <div class="dossier-col-claim">
          ${thesisSummaryHtml}
          ${fallacyHtml}
          ${socraticHtml}
          ${quoteHtml}
        </div>

        <div class="dossier-col-proof">
          ${premisesHtml}
          ${analogyHtml}
          <div class="proof-footer-area">
            ${tagsHtml}
            <div class="video-proof-bar">
              <div class="source-video-meta">
                <span class="timestamp-chip">${escapeHtml(watchTime)}</span>
                <span class="source-video-title" title="${escapeHtml(card.video_title_ar || "")}">${escapeHtml(card.video_title_ar || "SCDawah Arabic Archive")}</span>
              </div>
              <div class="proof-actions">
                <button type="button" class="btn-outline-action" onclick="askAssistant('${escapeJsStr(primaryQuestion)}')">
                  ${SVG_ICONS.bot}
                  <span>${escapeHtml(t.askBotBtn)}</span>
                </button>
                <a href="${escapeHtml(videoUrl)}" target="_blank" rel="noopener" class="btn-primary-emerald">
                  ${SVG_ICONS.play}
                  <span>${escapeHtml(t.watchBtn)} ${escapeHtml(watchTime)}</span>
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
    container.appendChild(article);
  });
}

function askAssistant(promptText) {
  switchTab("chat-tab");
  const input = document.getElementById("chat-input");
  if (input) {
    input.value = promptText;
    sendChatMessage();
  }
}

function sendChatMessage() {
  const input = document.getElementById("chat-input");
  const query = (input?.value || "").trim();
  if (!query) return;

  const messagesBox = document.getElementById("chat-messages");
  if (!messagesBox) return;

  const userMsgEl = document.createElement("div");
  userMsgEl.className = "message user-msg";
  userMsgEl.innerHTML = `<div class="msg-content"><p>${escapeHtml(query)}</p></div>`;
  messagesBox.appendChild(userMsgEl);
  input.value = "";
  messagesBox.scrollTop = messagesBox.scrollHeight;

  fetch(`/api/chat?q=${encodeURIComponent(query)}`)
    .then(r => r.json())
    .then(data => {
      const isAr = data.language === "ar" || currentLang === "ar";
      const t = translations[isAr ? "ar" : "en"];
      const pkg = data.response_package;

      let botResponseHtml = "";
      if (pkg) {
        const noticeHtml = data.status === "partial_match" && data.message
          ? `<div class="socratic-turn-panel" style="margin-bottom:0.85rem; padding:0.65rem 0.85rem; font-size:0.84rem; font-weight:600; color:var(--socratic-ochre);">${escapeHtml(data.message)}</div>`
          : "";

        const cleanPremises = (pkg.logical_premises || []).map(p => String(p || "").trim()).filter(p => p.length > 0);
        const premisesHtml = cleanPremises.length
          ? `
            <div style="margin-top:0.85rem;">
              <div class="proof-section-bar">
                <span>${SVG_ICONS.proof} ${escapeHtml(t.premisesLabel)}</span>
                <span>CONFIDENCE: ${escapeHtml(String(data.confidence_score || 95))}%</span>
              </div>
              <ol class="syllogism-list" style="margin-top:0.45rem;">
                ${cleanPremises.map((p, idx) => {
                  const isLast = idx === cleanPremises.length - 1 && cleanPremises.length > 1;
                  const badgeCode = isLast ? (isAr ? "النتيجة" : "Conclusion") : `${idx + 1}`;
                  const badgeClass = isLast ? "step-badge step-badge-conclusion" : "step-badge";
                  return `
                  <li class="syllogism-step">
                    <span class="${badgeClass}">${badgeCode}</span>
                    <span>${escapeHtml(p)}</span>
                  </li>
                  `;
                }).join("")}
              </ol>
            </div>
          `
          : "";

        const fallacyHtml = pkg.fallacy_exposed && !isNoneText(pkg.fallacy_exposed)
          ? `
            <div class="fallacy-panel" style="margin-top:0.75rem;">
              <div class="fallacy-panel-bar">${SVG_ICONS.fallacy} <span>${escapeHtml(t.fallacyLabel)}</span></div>
              <div class="fallacy-panel-body">${escapeHtml(pkg.fallacy_exposed)}</div>
            </div>
          `
          : "";

        const socraticHtml = pkg.socratic_question && !isNoneText(pkg.socratic_question)
          ? `
            <div class="socratic-turn-panel" style="margin-top:0.75rem;">
              <div class="socratic-turn-bar">
                <span class="socratic-badge-title">${SVG_ICONS.socratic} <span>${escapeHtml(t.socraticLabel)}</span></span>
                <button type="button" class="socratic-action-btn" onclick="copySocraticText('${escapeJsStr(pkg.socratic_question)}')">
                  ${SVG_ICONS.copy} <span>${escapeHtml(t.copyBtn)}</span>
                </button>
              </div>
              <div class="socratic-turn-body">
                ${SVG_ICONS.quoteMark}
                <div class="socratic-question-text">“${escapeHtml(pkg.socratic_question)}”</div>
              </div>
            </div>
          `
          : "";

        const analogyHtml = pkg.analogy && !isNoneText(pkg.analogy)
          ? `
            <div class="analogy-panel" style="margin-top:0.75rem;">
              <div class="analogy-bar">${SVG_ICONS.analogy} <span>${escapeHtml(t.analogyLabel)}</span></div>
              <div class="analogy-body">${escapeHtml(pkg.analogy)}</div>
            </div>
          `
          : "";

        const relatedHtml = data.related_questions && data.related_questions.length
          ? `
            <div style="margin-top:0.9rem; padding-top:0.75rem; border-top:1px dashed var(--border-hairline);">
              <div style="font-family:var(--font-mono); font-size:0.75rem; font-weight:700; text-transform:uppercase; color:var(--ink-muted); margin-bottom:0.4rem;">
                ${isAr ? "أسئلة ذات صلة في الأرشيف:" : "Related Dossiers in Corpus:"}
              </div>
              <div style="display:flex; flex-wrap:wrap; gap:0.4rem;">
                ${data.related_questions.map(rq => `
                  <button type="button" class="btn-outline-action" onclick="askAssistant('${escapeJsStr(rq.question)}')">
                    ${escapeHtml(rq.question)}
                  </button>
                `).join("")}
              </div>
            </div>
          `
          : "";

        botResponseHtml = `
          ${noticeHtml}
          <div class="bot-msg-header">
            <div style="display:flex; gap:0.4rem; flex-wrap:wrap;">
              <span class="badge-pillar">${escapeHtml(pkg.category || "")}</span>
              <span class="badge-interlocutor">${escapeHtml(pkg.interlocutor || "")}</span>
            </div>
            <span class="timestamp-chip">${escapeHtml(pkg.timestamp || "00:00")}</span>
          </div>
          <h3 style="font-size:1.18rem; font-weight:800; color:var(--ink-primary); margin-bottom:0.45rem;">${escapeHtml(pkg.title || "")}</h3>
          <p style="font-size:0.91rem; color:var(--ink-primary); line-height:1.6;">${escapeHtml(pkg.summary || "")}</p>
          ${fallacyHtml}
          ${socraticHtml}
          ${premisesHtml}
          ${analogyHtml}
          <div class="video-proof-bar" style="margin-top:0.9rem; padding-top:0.75rem; border-top:1px solid var(--border-hairline);">
            <span class="source-video-title">${escapeHtml(pkg.video_title || "")}</span>
            <a href="${escapeHtml(pkg.timestamp_url || "#")}" target="_blank" rel="noopener" class="btn-primary-emerald">
              ${SVG_ICONS.play}
              <span>${escapeHtml(t.watchBtn)} ${escapeHtml(pkg.timestamp || "00:00")}</span>
            </a>
          </div>
          ${relatedHtml}
        `;
      } else {
        const suggestionsHtml = data.related_questions && data.related_questions.length
          ? `
            <div style="margin-top:0.85rem; display:flex; flex-direction:column; gap:0.4rem;">
              ${data.related_questions.map(q => `
                <button type="button" class="starter-chip-btn" onclick="askAssistant('${escapeJsStr(q.question)}')">
                  ${escapeHtml(q.question)}
                </button>
              `).join("")}
            </div>
          `
          : "";
        botResponseHtml = `
          <p style="font-size:0.9rem; color:var(--ink-primary); white-space:pre-line;">${escapeHtml(data.message || "")}</p>
          ${suggestionsHtml}
        `;
      }

      const botMsgEl = document.createElement("div");
      botMsgEl.className = "message assistant-msg";
      botMsgEl.innerHTML = `<div class="msg-content">${botResponseHtml}</div>`;
      messagesBox.appendChild(botMsgEl);
      messagesBox.scrollTop = messagesBox.scrollHeight;
    })
    .catch(err => console.error("Chat error:", err));
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeJsStr(str) {
  if (!str) return "";
  return String(str)
    .replace(/\\/g, "\\\\")
    .replace(/'/g, "\\'")
    .replace(/"/g, "&quot;")
    .replace(/\n/g, " ");
}
