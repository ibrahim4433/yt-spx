/**
 * web/static/js/app.js
 * Dawah Dialectics V2 (yt-spx) — Bilingual Diarized Scenario Studio,
 * Multi-Speaker Claim-and-Proof Dossiers & Zero-LLM Smart Assistant
 */

const PAGE_SIZE = 20;
const SCENARIOS_PAGE_SIZE = 30;

let currentLang = "en";
let isParallelMode = true;
let currentCategory = "all";
let currentInterlocutor = "all";
let currentSpeaker = "all";
let currentVideoId = "all";
let currentVideoLabel = "";
let currentSort = "default";
let currentOffset = 0;
let currentTotal = 0;
let loadedCards = [];
let cardParallelOverrides = new Set();

let categoriesData = [];
let interlocutorsData = [];
let speakersData = [];
let corpusStats = {
  total_qa: 0,
  total_videos: 2640,
  stage1_completed: 0,
  stage2_completed: 0,
  categories_count: 0,
  cleaned_hours: 0,
  overlaps_resolved: 0,
  pedestrian_isolated: 0,
  speakers_identified: 0,
  total_spoken_words: 0
};

let scenariosOffset = 0;
let scenariosTotal = 0;
let loadedScenarios = [];
let scenarioSearchDebounce = null;
let searchDebounceTimeout = null;

// Interactive Scenario Studio State
let activeScenarioPayload = null;
let activeStudioEpisode = "all";
let activeTargetTurnId = "";
let studioYtApiReady = false;
let studioYtApiLoading = false;
let studioPlayer = null;
let studioSyncTimer = null;
let lastHighlightedTurnId = "";

const CATEGORY_MAP = {
  "Prophethood & Quran": { en: "Prophethood & Quran", ar: "النبوة والقرآن الكريم" },
  "Comparative Religion & Christianity": { en: "Comparative Religion & Christianity", ar: "مقارنة الأديان والمسيحية" },
  "Theology & God's Existence": { en: "Theology & God's Existence", ar: "العقيدة ووجود الله" },
  "Epistemology & Science": { en: "Epistemology & Science", ar: "المعرفة والعلم التجريبي" },
  "Shariah & Social Inquiries": { en: "Shariah & Social Inquiries", ar: "الشريعة والشبهات الاجتماعية" },
  "Ethics & Morality": { en: "Ethics & Morality", ar: "الأخلاق والقيم" },
  "Objective Morality & Ethics": { en: "Objective Morality & Ethics", ar: "الأخلاق الموضوعية والقيم" },
  "Problem of Evil & Suffering": { en: "Problem of Evil & Suffering", ar: "معضلة الشر والمعاناة" },
  "Dawah & Social Inquiries": { en: "Dawah & Social Inquiries", ar: "الدعوة والحوار الاجتماعي" },
  "Comparative Religion & Shariah": { en: "Comparative Religion & Shariah", ar: "مقارنة الأديان والتشريع" },
  "Comparative Religion & History": { en: "Comparative Religion & History", ar: "مقارنة الأديان والتاريخ" },
  "Comparative Religion & Ethics": { en: "Comparative Religion & Ethics", ar: "مقارنة الأديان والأخلاق" },
  "Social Inquiries & Feminism": { en: "Social Inquiries & Feminism", ar: "الشريعة والنسوية والقضايا المجتمعية" },
  "Textual Preservation & Reliability": { en: "Textual Preservation & Reliability", ar: "مقارنة الأديان وحفظ النصوص" },
  "Textual Preservation & Manuscript Evidence": { en: "Manuscript Evidence & Preservation", ar: "مقارنة الأديان والمخطوطات" },
  "Textual Preservation & Biblical Reliability": { en: "Biblical Reliability & Manuscripts", ar: "مقارنة الأديان وموثوقية الكتاب المقدس" },
  "Textual Preservation & Biblical Manuscripts": { en: "Biblical Manuscripts & Preservation", ar: "مقارنة الأديان ومخطوطات الكتاب المقدس" },
  "Textual Preservation & Biblical Inerrancy": { en: "Biblical Inerrancy & Criticism", ar: "مقارنة الأديان ونقد دعوى عصمة الإنجيل" },
  "Textual Preservation & Biblical Criticism": { en: "Biblical Criticism & Preservation", ar: "مقارنة الأديان والنقد النصي" },
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
  "Muslim with Doubts": { en: "Muslim Seeking Proof", ar: "مسلم يطلب البرهان" },
  "Shiite": { en: "Shiite", ar: "شيعي" },
  "Shia": { en: "Shia", ar: "شيعي" }
};

const SVG_ICONS = {
  fallacy: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6M9 9l6 6"/></svg>`,
  socratic: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>`,
  quoteMark: `<svg class="socratic-quote-glyph" width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M10 7H6a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h2v2a2 2 0 0 1-2 2H5v2h1a4 4 0 0 0 4-4V7zm10 0h-4a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h2v2a2 2 0 0 1-2 2h-1v2h1a4 4 0 0 0 4-4V7z"/></svg>`,
  proof: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>`,
  analogy: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>`,
  verbatim: `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>`,
  speaker: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M19 8v6M22 11h-6"/></svg>`,
  scenario: `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 6h16M4 12h10M4 18h14"/><circle cx="19" cy="12" r="2"/></svg>`,
  overlap: `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12h4l3-7 4 14 3-7h6"/></svg>`,
  play: `<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><polygon points="6 4 20 12 6 20 6 4"/></svg>`,
  copy: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`,
  bot: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`,
  parallel: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16"/></svg>`
};

const translations = {
  en: {
    navTitle: "Dawah Dialectics",
    navSubtitle: "جدليات الدعوة الإسلامية",
    betaBadgeLabel: "V2 DIARIZED",
    filterTopicLabel: "Theological Pillars",
    filterSpeakerLabel: "Da'ee Voiceprint",
    filterInterlocutorLabel: "Interlocutor Archetype",
    resetLabel: "Reset",
    allTopicsLabel: "All Extracted Dossiers",
    allSpeakersLabel: "All Identified Da'ees",
    allInterlocutorsLabel: "All Interlocutors",
    tabSearch: "Claim & Proof Dossiers",
    tabScenarios: "Scenario Transcripts",
    tabChat: "Smart Assistant",
    tabStats: "Pipeline Ledger",
    mobileFilterLabel: "Filters",
    searchHeading: "Claim & Proof Dialectical Dossier",
    searchSubheading: "Speaker-verified street-debate objections paired with exposed fallacies, Socratic counter-questions, deductive syllogisms, and dual-speaker dialogue exchanges.",
    searchPlaceholder: "Search doubts, premises, fallacies, or diarized quotes (EN / العربية)...",
    sortLabel: "Order:",
    sortDefault: "Corpus Index",
    sortTimeAsc: "Debate Timestamp (Earliest)",
    sortTimeDesc: "Debate Timestamp (Deepest)",
    sortCat: "Theological Pillar",
    showingRange: "Showing {start}–{end} of {total} dialectical dossiers",
    showingZero: "0 extracted dossiers matching current view",
    noResultsTitle: "No matching dialectical dossiers found",
    noResultsSub: "Try clearing an active filter pill or searching with broader theological keywords in English or Arabic.",
    emptyV2Title: "V2 Clean-Slate Corpus Initialized (2,640 Videos Queued)",
    emptyV2Sub: "All noisy V1 auto-captions have been purged. Run dawah_dialectics_v2_colab.ipynb on Google Colab Free T4 GPUs and sync dawah_knowledge.db to populate verified multi-speaker dossiers, or browse the 2,640 cataloged videos in Scenario Transcripts.",
    browseCatalogBtn: "Browse 2,640 Videos in Scenario Studio",
    loadMoreLabel: "Load Next 20 Dossiers",
    loadedStatus: "Loaded {loaded} of {total} matching dossiers",
    scenariosHeading: "Multi-Speaker Scenario Transcripts",
    scenariosSubheading: "Full chronological street-debate transcripts built with Demucs vocal cleaning, NeMo MSDD speaker diarization, MossFormer2 2-stream overlap separation, and pedestrian interruption filtering.",
    scenarioSearchPlaceholder: "Search video titles or spoken transcript turns across speakers...",
    roleAll: "All Speaker Roles",
    roleDaee: "Main Da'ee Only",
    roleInter: "Interlocutors Only",
    rolePed: "Pedestrian Interruptions",
    onlyDiarizedLabel: "Completed Stage 1 Scenarios Only",
    turnHitsTitle: "Direct Spoken Turn Matches (FTS5)",
    openStudioBtn: "Open Scenario Studio",
    filterVideoQaBtn: "View Dossiers",
    scenariosLoadMore: "Load Next 30 Videos",
    assistantTitle: "Smart Dialectical Assistant",
    assistantDesc: "Zero-LLM deterministic reasoning engine: retrieves speaker-verified syllogisms, exposed fallacies, analogies, Socratic counter-questions, and exact YouTube debate timestamps with zero hallucination.",
    chipHint: "Common Objections & Doubts",
    welcomeText: "Ask any theological objection, philosophical doubt, or inquiry about Islam in English or Arabic. The engine retrieves speaker-verified syllogisms, Socratic counter-questions, dual-speaker exchanges, and direct links to the Scenario Transcript Studio.",
    chatPlaceholder: "Type a question or objection in English or Arabic (e.g., 'Who created God?', 'من خلق الله؟')...",
    sendLabel: "Analyze Objection",
    premisesLabel: "Deductive Syllogism & Rational Proof",
    fallacyLabel: "Fallacy Exposed • المغالطة المكشوفة",
    analogyLabel: "Concrete Analogy • المثال التوضيحي",
    socraticLabel: "Socratic Counter-Question • السؤال السقراطي",
    interVerbatimLabel: "Interlocutor Verbatim Objection",
    verbatimLabel: "Da'ee Verbatim Evidence",
    exchangeLabel: "Chronological Speaker Exchange",
    watchBtn: "Watch at",
    scenarioJumpBtn: "Scenario Studio",
    askBotBtn: "Inspect in Assistant",
    copyBtn: "Copy",
    testBotBtn: "Test in Bot",
    parallelBtnLabel: "EN ⇄ عربي",
    copiedToast: "Copied Socratic counter-question to clipboard",
    footerText: "Grounded in vocal isolation, neural speaker diarization, overlap separation, and timestamped street dialogues.",
    statsTitle: "V2 Acoustic Diarization & Extraction Ledger",
    statsSubtitle: "Methodological transparency across 2,640 cataloged street-debate videos processed through Demucs vocal cleaning, NeMo MSDD diarization, MossFormer2 overlap separation, and Qwen 2.5 14B distillation.",
    statQaLabel: "Extracted Q&A Dossiers",
    statStage1Label: "Diarized Scenarios (Stage 1)",
    statHoursLabel: "Clean Vocals Preserved",
    statOverlapsLabel: "Overlaps Unmixed (2-Stream)",
    statPedestriansLabel: "Pedestrian Turns Isolated",
    statSpeakersLabel: "Debate Speakers Indexed",
    ledgerCatHeading: "Distribution by Theological Pillar",
    ledgerCatSub: "Click any pillar row to filter the Claim & Proof Archive directly.",
    ledgerSpeakerHeading: "Identified Da'ee Voiceprints",
    ledgerSpeakerSub: "Cross-video TitaNet-Large 192-D speaker profiles and catalog voiceprints.",
    ledgerInterHeading: "Distribution by Interlocutor Archetype",
    ledgerInterSub: "Real street-debate interlocutor profiles across extracted units.",
    colabTitle: "Multi-Account Google Colab Free T4 + 5 TB Drive Pipeline",
    colabDesc: "Every micro-step (M1 audio fetch, M2 chunked Demucs vocals, M3 NeMo MSDD diarization, M4 MossFormer2 overlap separation, M5 CTC scenario JSON, and M6 Qwen 2.5 14B episode extraction) saves atomically to /content/drive/MyDrive/yt-spx.",
    studioSpeakersHeading: "Diarized Speaker Roster",
    studioQaHeading: "Extracted Dossiers in This Video",
    studioFilterPlaceholder: "Filter turns in this video...",
    hidePedestriansBase: "Hide Pedestrian Interruptions",
    autoFollowLabel: "Auto-Follow Video",
    overlapBannerLabel: "Simultaneous Cross-Talk Unmixed • 2-Stream Separation",
    allEpisodesLabel: "All Episodes",
    episodePrefix: "Episode",
    queuedScenarioNotice: "This video is cataloged at M0_QUEUED. Once processed by Stage 1 on Google Colab T4, its full speaker-diarized & overlap-resolved transcript will appear here alongside the synced video.",
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
    navSubtitle: "Dawah Dialectics V2",
    betaBadgeLabel: "الجيل الثاني V2",
    filterTopicLabel: "الأبواب الفكرية والعقدية",
    filterSpeakerLabel: "البصمة الصوتية للداعية",
    filterInterlocutorLabel: "نمط المحاور في الميدان",
    resetLabel: "إعادة ضبط",
    allTopicsLabel: "كافة الملفات المستخرجة",
    allSpeakersLabel: "جميع الدعاة المحاورين",
    allInterlocutorsLabel: "جميع المحاورين",
    tabSearch: "ديوان الشبهات والبراهين",
    tabScenarios: "نصوص المناظرات الكاملة",
    tabChat: "المساعد الجدلي",
    tabStats: "سجل المعالجة V2",
    mobileFilterLabel: "الفلاتر",
    searchHeading: "ملفات تفكيك الشبهات والبرهان العقلي",
    searchSubheading: "كل شبهة ميدانية موثقة بالمتحدث الفعلي ومقترنة بكشف المغالطة المنطقية، والسؤال السقراطي المعاكس، والقياس العقلي، وتسلسل الحوار الثنائي.",
    searchPlaceholder: "ابحث في الشبهات أو المقدمات أو الاقتباسات الموثقة بالمتحدثين...",
    sortLabel: "الترتيب:",
    sortDefault: "التسلسل المرجعي",
    sortTimeAsc: "توقيت المناظرة (الأبكر)",
    sortTimeDesc: "توقيت المناظرة (الأعمق)",
    sortCat: "الباب الموضوعي",
    showingRange: "عرض {start}–{end} من أصل {total} ملف حواري",
    showingZero: "0 ملفات حوارية مطابقة للعرض الحالي",
    noResultsTitle: "لم يتم العثور على ملفات حوارية مطابقة",
    noResultsSub: "جرّب إلغاء بعض الفلاتر النشطة أو البحث بكلمات مفتاحية أوسع بالعربية أو الإنجليزية.",
    emptyV2Title: "تم تهيئة قاعدة بيانات الجيل الثاني النقية (2,640 فيديو مفهرس)",
    emptyV2Sub: "تم مسح جميع سجلات الترجمة التلقائية القديمة غير الدقيقة. شغّل دفعات المعالجة الصوتية عبر dawah_dialectics_v2_colab.ipynb على Google Colab T4 وزامن ملف dawah_knowledge.db، أو استعرض قائمة الفيديوهات في تبويب نصوص المناظرات الكاملة.",
    browseCatalogBtn: "استعرض 2,640 فيديو في استوديو المناظرات",
    loadMoreLabel: "تحميل 20 ملفاً إضافياً",
    loadedStatus: "تم عرض {loaded} من أصل {total} وحدة مطابقة",
    scenariosHeading: "نصوص المناظرات الكاملة المميزة بالمتحدثين",
    scenariosSubheading: "نصوص حوارية زمنية كاملة مبنية عبر عزل الضوضاء (Demucs)، وتمييز المتحدثين (NeMo MSDD)، وفصل تداخل الأصوات (MossFormer2)، وعزل مقاطعات المارة.",
    scenarioSearchPlaceholder: "ابحث في عناوين المناظرات أو داخل كلام المتحدثين مباشرة...",
    roleAll: "جميع أدوار المتحدثين",
    roleDaee: "الداعية المحاور فقط",
    roleInter: "الضيف / المحاور فقط",
    rolePed: "مقاطعات المارة فقط",
    onlyDiarizedLabel: "المناظرات المكتملة صوتياً فقط (Stage 1)",
    turnHitsTitle: "نتائج مطابقة مباشرة من كلام المتحدثين (FTS5)",
    openStudioBtn: "فتح استوديو المناظرة",
    filterVideoQaBtn: "عرض الحجج المستخرجة",
    scenariosLoadMore: "تحميل 30 فيديو إضافياً",
    assistantTitle: "المساعد الحواري الذكي الفوري",
    assistantDesc: "محرك استدلالي فوري (بدون هلوسة): يستخرج الأقيسة المنطقية، والمغالطات المكشوفة، والأمثلة التوضيحية، والأسئلة السقراطية، وروابط الفيديو الدقيقة.",
    chipHint: "أبرز الشبهات والأسئلة المتكررة",
    welcomeText: "اطرح أي شبهة فلسفية أو تساؤل عقدي بالعربية أو الإنجليزية؛ سيقوم المحرك باسترجاع القياس العقلي، والسؤال السقراطي المعاكس، وتسلسل الحوار الثنائي مع ربط مباشر باستوديو النص الكامل للمناظرة.",
    chatPlaceholder: "اكتب سؤالك أو الشبهة هنا بالعربية أو الإنجليزية (مثال: 'من خلق الله؟'، 'التثليث')...",
    sendLabel: "تحليل الشبهة",
    premisesLabel: "التسلسل البرهاني الاستنتاجي • Deductive Syllogism",
    fallacyLabel: "المغالطة المنطقية المكشوفة • Fallacy Exposed",
    analogyLabel: "المثال التوضيحي والقياس • Concrete Analogy",
    socraticLabel: "السؤال السقراطي المعاكس • Socratic Turn",
    interVerbatimLabel: "نص اعتراض المحاور الحرفي",
    verbatimLabel: "نص برهان الداعية الحرفي",
    exchangeLabel: "التسلسل الزمني للحوار الثنائي",
    watchBtn: "شاهد عند",
    scenarioJumpBtn: "استوديو النص الكامل",
    askBotBtn: "تحليل في المساعد",
    copyBtn: "نسخ",
    testBotBtn: "اختبر في المساعد",
    parallelBtnLabel: "عربي ⇄ EN",
    copiedToast: "تم نسخ السؤال السقراطي إلى الحافظة",
    footerText: "قائم على عزل الضوضاء، وتمييز البصمة الصوتية للمتحدثين، وفصل تداخل الكلام، والتوثيق الزمني الدقيق.",
    statsTitle: "سجل المعالجة الصوتية وتمييز المتحدثين (V2)",
    statsSubtitle: "شفافية منهجية شاملة لـ 2,640 مناظرة عبر عزل الأصوات (Demucs)، وتمييز المتحدثين (NeMo MSDD)، وفصل تداخل الكلام (MossFormer2)، واستخراج البراهين عبر Qwen 2.5 14B.",
    statQaLabel: "ملف حجة وبرهان مستخرج",
    statStage1Label: "سيناريو مناظرة مكتمل صوتياً",
    statHoursLabel: "ساعات صوت نقي معزول",
    statOverlapsLabel: "تداخلات كلام تم فصلها",
    statPedestriansLabel: "مقاطعات مارة تم عزلها",
    statSpeakersLabel: "متحدثون تم تمييزهم صوتياً",
    ledgerCatHeading: "توزيع الحجج حسب الأبواب الفكرية",
    ledgerCatSub: "اضغط على أي باب للانتقال المباشر إلى ملفاته في ديوان البراهين.",
    ledgerSpeakerHeading: "البصمات الصوتية للدعاة المحاورين",
    ledgerSpeakerSub: "تمييز صوتي عابر للفيديوهات عبر متجهات TitaNet-Large 192-D.",
    ledgerInterHeading: "توزيع الحجج حسب خلفية المحاور",
    ledgerInterSub: "أنماط المحاورين الفعلية في مناظرات الشارع المستخرجة.",
    colabTitle: "خط المعالجة المتوازي عبر Google Colab T4 + مساحة 5 تيرابايت",
    colabDesc: "كل خطوة جزئية (M1 تنزيل الصوت، M2 عزل الضوضاء، M3 تمييز المتحدثين، M4 فصل التداخل، M5 بناء السيناريو، M6 الاستخراج المنطقي) تُحفظ ذرياً على /content/drive/MyDrive/yt-spx.",
    studioSpeakersHeading: "قائمة المتحدثين المميزين صوتياً",
    studioQaHeading: "الحجج المستخرجة من هذه المناظرة",
    studioFilterPlaceholder: "ابحث داخل مداخلات هذه المناظرة...",
    hidePedestriansBase: "إخفاء مقاطعات المارة العشوائية",
    autoFollowLabel: "متابعة تلقائية مع الفيديو",
    overlapBannerLabel: "تداخل كلام متزامن تم فصله صوتياً إلى مسارين مستقلين",
    allEpisodesLabel: "كل الجولات الحوارية",
    episodePrefix: "جولة حوارية",
    queuedScenarioNotice: "هذا الفيديو مدرج في الفهرس بحالة الانتظار (M0_QUEUED). بمجرد إتمام معالجته الصوتية في المرحلة الأولى على Google Colab T4 سيظهر هنا النص الحواري الكامل مقسماً حسب المتحدثين ومتزامناً مع الفيديو.",
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
  initScenarioStudioDialog();
  setupKeyboardShortcuts();
  renderLanguageUI();
  loadCorpusMetadata();
  performSearch(true);
  loadScenarioCatalog(true);
});

function initScenarioStudioDialog() {
  const dialog = document.getElementById("scenario-studio-dialog");
  if (!dialog) return;

  // Light-dismiss fallback for browsers without native closedby="any" support
  if (!("closedBy" in HTMLDialogElement.prototype)) {
    dialog.addEventListener("click", (e) => {
      if (e.target === dialog) {
        dialog.close();
      }
    });
  }

  dialog.addEventListener("close", () => {
    stopStudioSyncLoop();
    if (studioPlayer && typeof studioPlayer.pauseVideo === "function") {
      try {
        studioPlayer.pauseVideo();
      } catch (_) {}
    } else {
      const wrap = document.getElementById("studio-player-wrap");
      if (wrap) wrap.innerHTML = `<div id="studio-yt-player"></div>`;
    }
  });
}

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
  renderScenarioCatalogCards(loadedScenarios);
  renderStatsTables();
  if (activeScenarioPayload) {
    renderStudioContents(activeScenarioPayload);
  }
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
    const dialog = document.getElementById("scenario-studio-dialog");
    if (dialog && dialog.open) return;

    if (
      (e.key === "/" || (e.ctrlKey && e.key.toLowerCase() === "k")) &&
      document.activeElement.tagName !== "INPUT" &&
      document.activeElement.tagName !== "TEXTAREA"
    ) {
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

  setText("filter-topic-label", t.filterTopicLabel);
  setText("filter-speaker-label", t.filterSpeakerLabel);
  setText("filter-interlocutor-label", t.filterInterlocutorLabel);
  setText("reset-filters-btn", t.resetLabel);

  setText("tab-search-label", t.tabSearch);
  setText("tab-scenarios-label", t.tabScenarios);
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

  setText("scenarios-heading", t.scenariosHeading);
  setText("scenarios-subheading", t.scenariosSubheading);
  const scenInput = document.getElementById("scenario-search-input");
  if (scenInput) scenInput.placeholder = t.scenarioSearchPlaceholder;
  setText("role-opt-all", t.roleAll);
  setText("role-opt-daee", t.roleDaee);
  setText("role-opt-inter", t.roleInter);
  setText("role-opt-ped", t.rolePed);
  setText("only-diarized-label", t.onlyDiarizedLabel);
  setText("turn-hits-title", t.turnHitsTitle);
  setText("scenarios-load-more-label", t.scenariosLoadMore);

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
  setText("stat-stage1-label", t.statStage1Label);
  setText("stat-hours-label", t.statHoursLabel);
  setText("stat-overlaps-label", t.statOverlapsLabel);
  setText("stat-pedestrians-label", t.statPedestriansLabel);
  setText("stat-speakers-label", t.statSpeakersLabel);
  setText("ledger-cat-heading", t.ledgerCatHeading);
  setText("ledger-cat-sub", t.ledgerCatSub);
  setText("ledger-speaker-heading", t.ledgerSpeakerHeading);
  setText("ledger-speaker-sub", t.ledgerSpeakerSub);
  setText("ledger-inter-heading", t.ledgerInterHeading);
  setText("ledger-inter-sub", t.ledgerInterSub);
  setText("colab-title", t.colabTitle);
  setText("colab-desc", t.colabDesc);

  setText("studio-speakers-heading", t.studioSpeakersHeading);
  setText("studio-qa-heading", t.studioQaHeading);
  const turnFilter = document.getElementById("studio-turn-filter");
  if (turnFilter) turnFilter.placeholder = t.studioFilterPlaceholder;
  setText("auto-follow-label", t.autoFollowLabel);

  const chipsContainer = document.getElementById("prompt-chips-container");
  if (chipsContainer) {
    chipsContainer.innerHTML = t.chips
      .map(
        (chip) => `
      <button type="button" class="starter-chip-btn" onclick="askAssistant('${escapeJsStr(chip.query)}')">
        ${escapeHtml(chip.text)}
      </button>
    `
      )
      .join("");
  }

  updateTelemetryStripUI();
  updateResultsCounterText();
  updateScenariosCounterText();
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.innerText = val;
}

function loadCorpusMetadata() {
  Promise.all([
    fetch("/api/stats").then((r) => r.json()),
    fetch("/api/categories").then((r) => r.json()),
    fetch("/api/interlocutors").then((r) => r.json()),
    fetch("/api/speakers").then((r) => r.json())
  ])
    .then(([stats, catData, interData, spkData]) => {
      corpusStats = stats || corpusStats;
      categoriesData = (catData && catData.categories) || [];
      interlocutorsData = (interData && interData.interlocutors) || [];
      speakersData = (spkData && spkData.speakers) || [];

      updateTelemetryStripUI();
      renderFacetRails();
      renderStatsTables();
      // Re-render empty state if total_qa updated
      if (loadedCards.length === 0) {
        renderCards([]);
      }
    })
    .catch((err) => console.error("Metadata load error:", err));
}

function updateTelemetryStripUI() {
  const totalQa = Number(corpusStats.total_qa || 0);
  const totalVids = Number(corpusStats.total_videos || 2640);
  const s1Done = Number(corpusStats.stage1_completed || 0);
  const cleanHrs = Number(corpusStats.cleaned_hours || 0);
  const overlaps = Number(corpusStats.overlaps_resolved || 0);
  const peds = Number(corpusStats.pedestrian_isolated || 0);
  const spks = Number(corpusStats.speakers_identified || 0);

  setText("stat-qa", totalQa.toLocaleString());
  setText("stat-stage1", `${s1Done.toLocaleString()} / ${totalVids.toLocaleString()}`);
  setText("stat-hours", `${cleanHrs.toFixed(1)}h`);
  setText("stat-overlaps", overlaps.toLocaleString());
  setText("stat-pedestrians", peds.toLocaleString());
  setText("stat-speakers", spks.toLocaleString());

  setText("nav-qa-count", totalQa.toLocaleString());
  setText("nav-scenarios-count", s1Done > 0 ? s1Done.toLocaleString() : totalVids.toLocaleString());

  const pct = totalVids > 0 ? Math.min(100, Math.round((s1Done / totalVids) * 100)) : 0;
  setText(
    "beta-ratio-num",
    totalQa > 0
      ? `${totalQa.toLocaleString()} Q&A · ${s1Done}/${totalVids}`
      : `${s1Done.toLocaleString()} / ${totalVids.toLocaleString()} (${pct}%)`
  );

  const fillEl = document.getElementById("beta-progress-fill");
  const trackEl = document.getElementById("beta-progress-track");
  if (fillEl) fillEl.style.width = `${pct}%`;
  if (trackEl) trackEl.setAttribute("aria-valuenow", String(pct));

  const botBadge = document.getElementById("welcome-bot-badge");
  if (botBadge) {
    botBadge.innerText =
      currentLang === "ar"
        ? `محرك الجيل الثاني جاهز • ${totalQa.toLocaleString()} حجة / ${s1Done.toLocaleString()} مناظرة مميزة صوتياً`
        : `V2 ENGINE READY • ${totalQa.toLocaleString()} DOSSIERS / ${s1Done.toLocaleString()} SCENARIOS`;
  }
}

function getLocalizedCategory(catKey, lang = currentLang) {
  if (!catKey) return "";
  if (CATEGORY_MAP[catKey] && CATEGORY_MAP[catKey][lang]) {
    return CATEGORY_MAP[catKey][lang];
  }
  if (lang === "ar") {
    if (catKey.includes("Textual Preservation") || catKey.includes("Manuscript") || catKey.includes("Biblical")) {
      return "مقارنة الأديان والمخطوطات";
    }
    if (catKey.includes("Feminism") || catKey.includes("Social Inquiries")) {
      return "الشريعة والقضايا المجتمعية";
    }
    if (catKey.includes("Comparative Religion")) {
      return "مقارنة الأديان والمسيحية";
    }
    if (catKey.includes("Theology") || catKey.includes("God")) {
      return "العقيدة ووجود الله";
    }
    if (catKey.includes("Morality") || catKey.includes("Ethics")) {
      return "الأخلاق الموضوعية والقيم";
    }
    if (catKey.includes("Prophethood") || catKey.includes("Quran")) {
      return "النبوة والقرآن الكريم";
    }
    if (catKey.includes("Evil") || catKey.includes("Suffering")) {
      return "معضلة الشر والمعاناة";
    }
    if (catKey.includes("Epistemology") || catKey.includes("Science")) {
      return "المعرفة والعلم التجريبي";
    }
  }
  return catKey;
}

function getLocalizedInterlocutor(interKey, lang = currentLang) {
  if (!interKey) return "";
  return (INTERLOCUTOR_MAP[interKey] && INTERLOCUTOR_MAP[interKey][lang]) || interKey;
}

function getLocalizedSpeaker(speakerKey) {
  if (!speakerKey || speakerKey === "all") return "";
  const found = speakersData.find((s) => s.speaker_key === speakerKey || s.name_en === speakerKey);
  if (!found) return speakerKey;
  return currentLang === "ar" ? found.name_ar || found.name_en : found.name_en || found.name_ar;
}

function renderFacetRails() {
  const t = translations[currentLang];
  const totalQa = Number(corpusStats.total_qa || 0);

  const catContainer = document.getElementById("category-pills");
  if (catContainer) {
    catContainer.innerHTML = `
      <button type="button" class="facet-btn ${currentCategory === "all" ? "active" : ""}" onclick="setCategoryFilter('all')">
        <span>${escapeHtml(t.allTopicsLabel)}</span>
        <span class="facet-count">${totalQa}</span>
      </button>
      ${categoriesData
        .map(
          (item) => {
            const catLabel = currentLang === "ar"
              ? (item.topic_category_ar || getLocalizedCategory(item.topic_category, "ar"))
              : (item.topic_category || getLocalizedCategory(item.topic_category, "en"));
            return `
        <button type="button" class="facet-btn ${currentCategory === item.topic_category ? "active" : ""}" onclick="setCategoryFilter('${escapeJsStr(item.topic_category)}')">
          <span>${escapeHtml(catLabel)}</span>
          <span class="facet-count">${item.count}</span>
        </button>
      `;
          }
        )
        .join("")}
    `;
  }

  const spkContainer = document.getElementById("speaker-pills");
  if (spkContainer) {
    spkContainer.innerHTML = `
      <button type="button" class="facet-btn ${currentSpeaker === "all" ? "active" : ""}" onclick="setSpeakerFilter('all')">
        <span>${escapeHtml(t.allSpeakersLabel)}</span>
        <span class="facet-count">${speakersData.length}</span>
      </button>
      ${speakersData
        .map((spk) => {
          const disp = currentLang === "ar" ? spk.name_ar || spk.name_en : spk.name_en || spk.name_ar;
          const cnt = spk.qa_count > 0 ? spk.qa_count : `${spk.videos_count}v`;
          return `
          <button type="button" class="facet-btn ${currentSpeaker === spk.speaker_key ? "active" : ""}" onclick="setSpeakerFilter('${escapeJsStr(spk.speaker_key)}')">
            <span>${escapeHtml(disp)}</span>
            <span class="facet-count">${escapeHtml(String(cnt))}</span>
          </button>
        `;
        })
        .join("")}
    `;
  }

  const interContainer = document.getElementById("interlocutor-pills");
  if (interContainer) {
    interContainer.innerHTML = `
      <button type="button" class="facet-btn ${currentInterlocutor === "all" ? "active" : ""}" onclick="setInterlocutorFilter('all')">
        <span>${escapeHtml(t.allInterlocutorsLabel)}</span>
        <span class="facet-count">${totalQa}</span>
      </button>
      ${interlocutorsData
        .map(
          (item) => `
        <button type="button" class="facet-btn ${currentInterlocutor === item.interlocutor_type ? "active" : ""}" onclick="setInterlocutorFilter('${escapeJsStr(item.interlocutor_type)}')">
          <span>${escapeHtml(getLocalizedInterlocutor(item.interlocutor_type))}</span>
          <span class="facet-count">${item.count}</span>
        </button>
      `
        )
        .join("")}
    `;
  }
}

function renderStatsTables() {
  const isAr = currentLang === "ar";
  const emptyLedgerMsg = isAr
    ? "سيظهر التوزيع التفصيلي تلقائياً فور مزامنة أول دفعة مستخرجة من Google Colab."
    : "Detailed breakdown bars populate automatically once the first Google Colab V2 shard batch is synced.";

  const totalQa = Math.max(Number(corpusStats.total_qa || 0), 1);

  const catTable = document.getElementById("stats-categories-table");
  if (catTable) {
    if (categoriesData.length) {
      catTable.innerHTML = categoriesData
        .map((item) => {
          const pct = Math.max(Math.round((item.count / totalQa) * 100), 2);
          const catLabel = isAr
            ? (item.topic_category_ar || getLocalizedCategory(item.topic_category, "ar"))
            : (item.topic_category || getLocalizedCategory(item.topic_category, "en"));
          return `
            <button type="button" class="arch-bar-row" onclick="jumpToCategoryFromStats('${escapeJsStr(item.topic_category)}')">
              <div class="arch-bar-top">
                <span>${escapeHtml(catLabel)}</span>
                <span class="arch-bar-count">${item.count} (${pct}%)</span>
              </div>
              <div class="arch-bar-track">
                <div class="arch-bar-fill" style="width: ${pct}%;"></div>
              </div>
            </button>
          `;
        })
        .join("");
    } else {
      catTable.innerHTML = `<p class="sheet-sub" style="margin:0;">${escapeHtml(emptyLedgerMsg)}</p>`;
    }
  }

  const spkTable = document.getElementById("stats-speakers-table");
  if (spkTable) {
    if (speakersData.length) {
      const maxSpkVal = Math.max(...speakersData.map((s) => Number(s.qa_count || s.videos_count || 1)), 1);
      spkTable.innerHTML = speakersData
        .map((spk) => {
          const val = Number(spk.qa_count || spk.videos_count || 0);
          const pct = Math.max(Math.round((val / maxSpkVal) * 100), 4);
          const disp = isAr ? spk.name_ar || spk.name_en : spk.name_en || spk.name_ar;
          const unitLbl = spk.qa_count > 0 ? `${spk.qa_count} Q&A · ${spk.videos_count} vid` : `${spk.videos_count} vid`;
          return `
            <button type="button" class="arch-bar-row" onclick="setSpeakerFilter('${escapeJsStr(spk.speaker_key)}')">
              <div class="arch-bar-top">
                <span>${escapeHtml(disp)}</span>
                <span class="arch-bar-count">${escapeHtml(unitLbl)}</span>
              </div>
              <div class="arch-bar-track">
                <div class="arch-bar-fill" style="width: ${pct}%;"></div>
              </div>
            </button>
          `;
        })
        .join("");
    } else {
      spkTable.innerHTML = `<p class="sheet-sub" style="margin:0;">${escapeHtml(emptyLedgerMsg)}</p>`;
    }
  }

  const interTable = document.getElementById("stats-interlocutors-table");
  if (interTable) {
    if (interlocutorsData.length) {
      interTable.innerHTML = interlocutorsData
        .map((item) => {
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
        })
        .join("");
    } else {
      interTable.innerHTML = `<p class="sheet-sub" style="margin:0;">${escapeHtml(emptyLedgerMsg)}</p>`;
    }
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
  document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));

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

function setSpeakerFilter(speakerKey) {
  currentSpeaker = speakerKey;
  renderFacetRails();
  toggleMobileRail(false);
  const activeScenPane = document.getElementById("scenarios-tab")?.classList.contains("active");
  if (activeScenPane) {
    loadScenarioCatalog(true);
  } else {
    switchTab("search-tab");
    performSearch(true);
    loadScenarioCatalog(true);
  }
}

function setVideoFilter(videoId, videoTitle = "") {
  currentVideoId = videoId;
  currentVideoLabel = videoTitle || videoId;
  switchTab("search-tab");
  performSearch(true);
}

function resetAllFilters() {
  currentCategory = "all";
  currentInterlocutor = "all";
  currentSpeaker = "all";
  currentVideoId = "all";
  currentVideoLabel = "";
  currentSort = "default";

  const searchInput = document.getElementById("search-input");
  if (searchInput) searchInput.value = "";
  const sortSelect = document.getElementById("sort-select");
  if (sortSelect) sortSelect.value = "default";
  const clearBtn = document.getElementById("clear-search-btn");
  if (clearBtn) clearBtn.hidden = true;

  renderFacetRails();
  performSearch(true);
  loadScenarioCatalog(true);
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
  const params = new URLSearchParams({
    q: query,
    category: currentCategory,
    interlocutor: currentInterlocutor,
    speaker: currentSpeaker,
    video_id: currentVideoId,
    sort: currentSort,
    limit: String(PAGE_SIZE),
    offset: String(currentOffset)
  });

  fetch(`/api/search?${params.toString()}`)
    .then((r) => r.json())
    .then((data) => {
      const batch = data.results || [];
      if (
        resetOffset &&
        query &&
        batch.length === 0 &&
        (currentCategory !== "all" || currentInterlocutor !== "all" || currentSpeaker !== "all" || currentVideoId !== "all")
      ) {
        currentCategory = "all";
        currentInterlocutor = "all";
        currentSpeaker = "all";
        currentVideoId = "all";
        renderFacetRails();
        performSearch(true);
        return;
      }
      currentTotal = typeof data.total === "number" ? data.total : batch.length;
      loadedCards = resetOffset ? batch : loadedCards.concat(batch);
      renderActiveFilterChips();
      updateResultsCounterText();
      renderCards(loadedCards);
    })
    .catch((err) => console.error("Search error:", err));
}

function loadMoreResults() {
  currentOffset += PAGE_SIZE;
  performSearch(false);
}

function updateResultsCounterText() {
  const t = translations[currentLang];
  const counterEl = document.getElementById("results-count-text");
  if (counterEl) {
    if (currentTotal === 0 || loadedCards.length === 0) {
      counterEl.innerText = t.showingZero;
    } else {
      counterEl.innerText = t.showingRange
        .replace("{start}", "1")
        .replace("{end}", loadedCards.length.toLocaleString())
        .replace("{total}", currentTotal.toLocaleString());
    }
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
  if (currentSpeaker !== "all") {
    chips.push(`
      <button type="button" class="filter-tag-chip" onclick="setSpeakerFilter('all')">
        <span>${escapeHtml(getLocalizedSpeaker(currentSpeaker))}</span>
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
  if (currentVideoId !== "all") {
    chips.push(`
      <button type="button" class="filter-tag-chip" onclick="setVideoFilter('all')">
        <span>${escapeHtml(currentVideoLabel.slice(0, 32))}</span>
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
    const isFreshV2 = Number(corpusStats.total_qa || 0) === 0;
    if (isFreshV2) {
      container.innerHTML = `
        <div class="empty-dossier-state">
          <h3>${escapeHtml(t.emptyV2Title)}</h3>
          <p style="max-width:68ch; margin-inline:auto;">${escapeHtml(t.emptyV2Sub)}</p>
          <button type="button" class="btn-primary-emerald" onclick="switchTab('scenarios-tab')">
            ${SVG_ICONS.scenario}
            <span>${escapeHtml(t.browseCatalogBtn)}</span>
          </button>
        </div>
      `;
    } else {
      container.innerHTML = `
        <div class="empty-dossier-state">
          <h3>${escapeHtml(t.noResultsTitle)}</h3>
          <p>${escapeHtml(t.noResultsSub)}</p>
          <button type="button" class="btn-primary-emerald" onclick="resetAllFilters()">${escapeHtml(t.resetLabel)}</button>
        </div>
      `;
    }
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

    const categoryName = isAr
      ? (card.topic_category_ar || getLocalizedCategory(card.topic_category, "ar"))
      : (card.topic_category || getLocalizedCategory(card.topic_category, "en"));
    const interlocutorName = getLocalizedInterlocutor(card.interlocutor_type);
    const subcategoryText = card.topic_subcategory && !isNoneText(card.topic_subcategory) ? card.topic_subcategory : "";

    const daeeFilterKey = card.daee_profile_id || card.daee_name_en || "";
    const daeeDisplay = isAr
      ? card.daee_name_ar || card.daee_name_en || ""
      : card.daee_name_en || card.daee_name_ar || "";
    const episodeIdx = Number(card.episode_index || 1);

    const questionBannerHtml = showParallel
      ? `
        <div class="question-parallel-grid">
          <div class="q-col-en"><h3>${escapeHtml(qEn)}</h3></div>
          <div class="q-col-ar"><h3>${escapeHtml(qAr)}</h3></div>
        </div>
      `
      : `<div class="question-single"><h3>${escapeHtml(primaryQuestion)}</h3></div>`;

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
    const activeFallacy = isAr ? fallacyAr || fallacyEn : fallacyEn || fallacyAr;
    let fallacyHtml = "";
    if (activeFallacy && !isNoneText(activeFallacy)) {
      const fallacyBody =
        showParallel && fallacyEn && fallacyAr && fallacyEn !== fallacyAr
          ? `<div>${escapeHtml(fallacyEn)}</div><div class="ar-font" dir="rtl" style="margin-top:0.35rem; padding-top:0.35rem; border-top:1px dashed rgba(166,43,29,0.2);">${escapeHtml(fallacyAr)}</div>`
          : escapeHtml(activeFallacy);
      fallacyHtml = `
        <div class="fallacy-panel">
          <div class="fallacy-panel-bar">${SVG_ICONS.fallacy}<span>${escapeHtml(t.fallacyLabel)}</span></div>
          <div class="fallacy-panel-body">${fallacyBody}</div>
        </div>
      `;
    }

    // Socratic Turn
    const socraticEn = card.socratic_question;
    const socraticAr = card.socratic_question_ar;
    const activeSocratic = isAr ? socraticAr || socraticEn : socraticEn || socraticAr;
    let socraticHtml = "";
    if (activeSocratic && !isNoneText(activeSocratic)) {
      const socraticBody =
        showParallel && socraticEn && socraticAr && socraticEn !== socraticAr
          ? `
            <div>“${escapeHtml(socraticEn)}”</div>
            <div class="ar-font" dir="rtl" style="margin-top:0.4rem; padding-top:0.4rem; border-top:1px dashed rgba(138,75,8,0.22); font-style:normal;">«${escapeHtml(socraticAr)}»</div>
          `
          : `“${escapeHtml(activeSocratic)}”`;

      socraticHtml = `
        <div class="socratic-turn-panel">
          <div class="socratic-turn-bar">
            <span class="socratic-badge-title">${SVG_ICONS.socratic}<span>${escapeHtml(t.socraticLabel)}</span></span>
            <div class="socratic-micro-actions">
              <button type="button" class="socratic-action-btn" onclick="copySocraticText('${escapeJsStr(activeSocratic)}')">
                ${SVG_ICONS.copy}<span>${escapeHtml(t.copyBtn)}</span>
              </button>
              <button type="button" class="socratic-action-btn" onclick="askAssistant('${escapeJsStr(primaryQuestion)}')">
                ${SVG_ICONS.bot}<span>${escapeHtml(t.testBotBtn)}</span>
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

    // Dual Verbatim Quotes (Interlocutor Objection + Da'ee Evidence)
    const interQuote = isAr
      ? card.interlocutor_verbatim_quote_ar || card.interlocutor_verbatim_quote
      : card.interlocutor_verbatim_quote || card.interlocutor_verbatim_quote_ar;
    const daeeQuote = isAr
      ? card.verbatim_quote_ar || card.verbatim_quote
      : card.verbatim_quote || card.verbatim_quote_ar;

    const verbatimParts = [];
    if (interQuote && !isNoneText(interQuote)) {
      verbatimParts.push(`
        <div class="verbatim-interlocutor-strip">
          <div class="verbatim-strip-label">
            ${SVG_ICONS.verbatim}
            <span>${escapeHtml(t.interVerbatimLabel)} [${escapeHtml(card.timestamp_formatted || "00:00")}]</span>
          </div>
          <div>“${escapeHtml(interQuote)}”</div>
        </div>
      `);
    }
    if (daeeQuote && !isNoneText(daeeQuote)) {
      const ansTs = card.answer_timestamp_formatted || card.timestamp_formatted || "00:00";
      verbatimParts.push(`
        <div class="verbatim-strip">
          <div class="verbatim-strip-label">
            ${SVG_ICONS.verbatim}
            <span>${escapeHtml(t.verbatimLabel)} [${escapeHtml(ansTs)}]</span>
          </div>
          <div>“${escapeHtml(daeeQuote)}”</div>
        </div>
      `);
    }
    const quoteHtml = verbatimParts.length ? `<div class="verbatim-stack">${verbatimParts.join("")}</div>` : "";

    // Deductive Syllogism Premises
    const cleanPremiseStr = (str) =>
      String(str || "")
        .trim()
        .replace(
          /^(Premise\s*\d+\s*[:.-]?\s*|Conclusion\s*[:.-]?\s*|المقدمة\s*(الأولى|الثانية|الثالثة|الرابعة|الخامسة|\d+)\s*[:.-]?\s*|النتيجة\s*[:.-]?\s*|الاستنتاج\s*[:.-]?\s*)/i,
          ""
        );

    const rawPremisesEn = (card.logical_premises || []).map(cleanPremiseStr).filter((p) => p.length > 0);
    const rawPremisesAr = (card.logical_premises_ar || []).map(cleanPremiseStr).filter((p) => p.length > 0);
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
            ${activePremises
              .map((p, idx) => {
                const isLast = idx === activePremises.length - 1 && activePremises.length > 1;
                const badgeCode = isLast ? (isAr ? "النتيجة" : "Conclusion") : `${idx + 1}`;
                const badgeClass = isLast ? "step-badge step-badge-conclusion" : "step-badge";
                const pEnItem = rawPremisesEn[idx] || p;
                const pArItem = rawPremisesAr[idx] || "";
                const stepContent =
                  showParallel && pArItem && pEnItem !== pArItem
                    ? `<div>${escapeHtml(pEnItem)}</div><div class="ar-font" dir="rtl" style="margin-top:0.3rem; padding-top:0.3rem; border-top:1px dashed var(--border-hairline); color:var(--ink-secondary);">${escapeHtml(pArItem)}</div>`
                    : escapeHtml(p);
                return `
                  <li class="syllogism-step">
                    <span class="${badgeClass}">${badgeCode}</span>
                    <div>${stepContent}</div>
                  </li>
                `;
              })
              .join("")}
          </ol>
        </div>
      `;
    }

    // Concrete Analogy
    const analogyEn = card.analogy_or_metaphor;
    const analogyAr = card.analogy_or_metaphor_ar;
    const activeAnalogy = isAr ? analogyAr || analogyEn : analogyEn || analogyAr;
    let analogyHtml = "";
    if (activeAnalogy && !isNoneText(activeAnalogy)) {
      const analogyBody =
        showParallel && analogyEn && analogyAr && analogyEn !== analogyAr
          ? `<div>${escapeHtml(analogyEn)}</div><div class="ar-font" dir="rtl" style="margin-top:0.35rem; padding-top:0.35rem; border-top:1px dashed rgba(15,82,62,0.2);">${escapeHtml(analogyAr)}</div>`
          : escapeHtml(activeAnalogy);
      analogyHtml = `
        <div class="analogy-panel">
          <div class="analogy-bar">${SVG_ICONS.analogy}<span>${escapeHtml(t.analogyLabel)}</span></div>
          <div class="analogy-body">${analogyBody}</div>
        </div>
      `;
    }

    // Chronological Dual-Speaker Exchange Timeline (V2)
    const exchanges = Array.isArray(card.dialogue_exchange) ? card.dialogue_exchange : [];
    let exchangeHtml = "";
    if (exchanges.length) {
      exchangeHtml = `
        <div class="exchange-timeline-panel">
          <div class="exchange-timeline-bar">
            <span>${SVG_ICONS.scenario} ${escapeHtml(t.exchangeLabel)}</span>
            <span>${exchanges.length} TURNS</span>
          </div>
          <div class="exchange-steps-list">
            ${exchanges
              .slice(0, 6)
              .map((ex) => {
                const isDaeeRole = String(ex.role || "").includes("DAEE");
                const roleCls = isDaeeRole ? "exchange-role-daee" : "exchange-role-inter";
                let spkName = "";
                if (isAr) {
                  spkName = ex.speaker_label_ar && !["MAIN_DAEE", "INTERLOCUTOR", "CO_DAEE"].includes(ex.speaker_label_ar)
                    ? ex.speaker_label_ar
                    : (isDaeeRole ? (card.daee_name_ar || "الداعية المحاور") : "المحاور");
                } else {
                  spkName = ex.speaker_label_en && !["MAIN_DAEE", "INTERLOCUTOR", "CO_DAEE"].includes(ex.speaker_label_en)
                    ? ex.speaker_label_en
                    : (isDaeeRole ? (card.daee_name_en || "Main Da'ee") : "Interlocutor");
                }
                const exTxt = isAr ? ex.text_ar || ex.text_en || "" : ex.text_en || ex.text_ar || "";
                const exTs = ex.timestamp_formatted || card.timestamp_formatted || "00:00";
                const exSec = Number(ex.timestamp_seconds || card.timestamp_seconds || 0);
                return `
                  <button type="button" class="exchange-step-item" onclick="openScenarioStudio('${escapeJsStr(card.video_id || "")}', '${escapeJsStr(ex.turn_id || "")}', ${exSec})">
                    <div class="exchange-step-meta">
                      <span class="${roleCls}">${escapeHtml(spkName)} · ${escapeHtml(ex.move_type || "")}</span>
                      <span>${escapeHtml(exTs)}</span>
                    </div>
                    <div class="exchange-step-text">${escapeHtml(exTxt)}</div>
                  </button>
                `;
              })
              .join("")}
          </div>
        </div>
      `;
    }

    const tags = Array.isArray(card.tags) ? card.tags.slice(0, 5) : [];
    const tagsHtml = tags.length
      ? `
        <div class="concept-tags-row">
          ${tags
            .map(
              (tag) =>
                `<button type="button" class="concept-tag" onclick="filterByTag('${escapeJsStr(tag)}')">#${escapeHtml(tag)}</button>`
            )
            .join("")}
        </div>
      `
      : "";

    const watchTime = card.timestamp_formatted || "00:00";
    const videoUrl = card.timestamp_url || card.youtube_url || "#";
    const firstTurnId = Array.isArray(card.source_turn_ids) && card.source_turn_ids.length ? card.source_turn_ids[0] : "";
    const startSec = Number(card.timestamp_seconds || 0);

    const article = document.createElement("article");
    // Defer off-screen rendering ONLY on below-the-fold cards (index >= 3)
    article.className = index >= 3 ? "dossier-card dossier-card-deferred" : "dossier-card";
    article.innerHTML = `
      <div class="dossier-banner">
        <div class="dossier-meta-bar">
          <div class="meta-badges-left">
            <span class="badge-serial">${escapeHtml(serialCode)}</span>
            ${
              daeeDisplay
                ? `<button type="button" class="badge-daee" onclick="setSpeakerFilter('${escapeJsStr(daeeFilterKey)}')">${SVG_ICONS.speaker}<span>${escapeHtml(daeeDisplay)}</span></button>`
                : ""
            }
            <button type="button" class="badge-pillar" onclick="setCategoryFilter('${escapeJsStr(card.topic_category)}')">
              ${escapeHtml(categoryName)}
            </button>
            ${subcategoryText ? `<span class="badge-subcat">· ${escapeHtml(subcategoryText)}</span>` : ""}
            <button type="button" class="badge-interlocutor" onclick="setInterlocutorFilter('${escapeJsStr(card.interlocutor_type)}')">
              ${escapeHtml(interlocutorName)}
            </button>
            ${episodeIdx > 1 ? `<span class="badge-episode">EP ${episodeIdx}</span>` : ""}
          </div>
          <button type="button" class="card-lang-peek-btn ${showParallel ? "active" : ""}" onclick="toggleCardParallel('${escapeJsStr(cardId)}')">
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
          ${exchangeHtml}
          <div class="proof-footer-area">
            ${tagsHtml}
            <div class="video-proof-bar">
              <div class="source-video-meta">
                <span class="timestamp-chip">${escapeHtml(watchTime)}</span>
                <span class="source-video-title" title="${escapeHtml(card.video_title_ar || "")}">${escapeHtml(card.video_title_ar || "SCDawah Arabic Archive")}</span>
              </div>
              <div class="proof-actions">
                ${
                  card.video_id
                    ? `<button type="button" class="btn-outline-action" onclick="openScenarioStudio('${escapeJsStr(card.video_id)}', '${escapeJsStr(firstTurnId)}', ${startSec})">
                        ${SVG_ICONS.scenario}
                        <span>${escapeHtml(t.scenarioJumpBtn)}</span>
                      </button>`
                    : ""
                }
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

/* ==========================================================================
   SCENARIO TRANSCRIPTS CATALOG & FULL-TEXT TURN SEARCH
   ========================================================================== */

function handleScenarioSearchInput() {
  clearTimeout(scenarioSearchDebounce);
  scenarioSearchDebounce = setTimeout(() => {
    loadScenarioCatalog(true);
    runDirectTurnSearch();
  }, 200);
}

function runDirectTurnSearch() {
  const q = (document.getElementById("scenario-search-input")?.value || "").trim();
  const role = document.getElementById("scenario-role-select")?.value || "all";
  const box = document.getElementById("transcript-turn-hits-box");
  const list = document.getElementById("transcript-turn-hits-list");
  const cntEl = document.getElementById("turn-hits-count");
  if (!box || !list || !cntEl) return;

  if (q.length < 2) {
    box.hidden = true;
    list.innerHTML = "";
    return;
  }

  fetch(`/api/transcript-search?q=${encodeURIComponent(q)}&role=${encodeURIComponent(role)}&limit=25`)
    .then((r) => r.json())
    .then((data) => {
      const turns = data.turns || [];
      if (!turns.length) {
        box.hidden = true;
        return;
      }
      box.hidden = false;
      cntEl.innerText = String(data.total || turns.length);
      list.innerHTML = turns
        .map((turn) => {
          const sec = Math.floor(Number(turn.start_ms || 0) / 1000);
          const spk = currentLang === "ar"
            ? turn.speaker_label_ar || turn.speaker_label_en || turn.speaker_role
            : turn.speaker_label_en || turn.speaker_role;
          return `
            <button type="button" class="turn-hit-row" onclick="openScenarioStudio('${escapeJsStr(turn.video_id)}', '${escapeJsStr(turn.turn_id)}', ${sec})">
              <div class="turn-hit-top">
                <span class="${String(turn.speaker_role).includes("DAEE") ? "exchange-role-daee" : "exchange-role-inter"}">
                  [${escapeHtml(turn.start_fmt || "00:00")}] ${escapeHtml(spk)}
                </span>
                <span class="ar-font" style="color:var(--ink-secondary);">${escapeHtml((turn.video_title_ar || "").slice(0, 65))}</span>
              </div>
              <div style="font-size:0.94rem; color:var(--ink-primary);">${escapeHtml(turn.text_original || turn.text_ar || "")}</div>
            </button>
          `;
        })
        .join("");
    })
    .catch((err) => console.error("Transcript search error:", err));
}

function loadScenarioCatalog(resetOffset = true) {
  if (resetOffset) {
    scenariosOffset = 0;
    loadedScenarios = [];
  }

  const q = (document.getElementById("scenario-search-input")?.value || "").trim();
  const onlyDiarized = Boolean(document.getElementById("only-diarized-checkbox")?.checked);
  const params = new URLSearchParams({
    q,
    speaker: currentSpeaker,
    only_diarized: String(onlyDiarized),
    limit: String(SCENARIOS_PAGE_SIZE),
    offset: String(scenariosOffset)
  });

  fetch(`/api/videos?${params.toString()}`)
    .then((r) => r.json())
    .then((data) => {
      const batch = data.videos || [];
      scenariosTotal = typeof data.total === "number" ? data.total : batch.length;
      loadedScenarios = resetOffset ? batch : loadedScenarios.concat(batch);
      updateScenariosCounterText();
      renderScenarioCatalogCards(loadedScenarios);
    })
    .catch((err) => console.error("Video catalog load error:", err));
}

function loadMoreScenarios() {
  scenariosOffset += SCENARIOS_PAGE_SIZE;
  loadScenarioCatalog(false);
}

function updateScenariosCounterText() {
  const isAr = currentLang === "ar";
  const el = document.getElementById("scenarios-count-text");
  if (el) {
    el.innerText = isAr
      ? `${scenariosTotal.toLocaleString()} فيديو في القائمة`
      : `${scenariosTotal.toLocaleString()} cataloged videos`;
  }

  const pagBar = document.getElementById("scenarios-pagination-bar");
  const pagStatus = document.getElementById("scenarios-pagination-status");
  const moreBtn = document.getElementById("scenarios-load-more-btn");
  if (pagBar && pagStatus && moreBtn) {
    if (scenariosTotal > 0) {
      pagBar.hidden = false;
      pagStatus.innerText = isAr
        ? `تم عرض ${loadedScenarios.length.toLocaleString()} من أصل ${scenariosTotal.toLocaleString()} فيديو`
        : `Showing ${loadedScenarios.length.toLocaleString()} of ${scenariosTotal.toLocaleString()} videos`;
      moreBtn.hidden = loadedScenarios.length >= scenariosTotal;
    } else {
      pagBar.hidden = true;
    }
  }
}

function formatDurationSec(sec) {
  const s = Math.max(0, Math.round(Number(sec || 0)));
  if (!s) return "--:--";
  const hrs = Math.floor(s / 3600);
  const mins = Math.floor((s % 3600) / 60);
  const rem = s % 60;
  if (hrs > 0) {
    return `${hrs}:${String(mins).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
  }
  return `${String(mins).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
}

function renderScenarioCatalogCards(videos) {
  const container = document.getElementById("scenarios-cards-container");
  if (!container) return;
  container.innerHTML = "";

  const t = translations[currentLang];
  const isAr = currentLang === "ar";

  if (!videos.length) {
    container.innerHTML = `
      <div class="empty-dossier-state">
        <h3>${escapeHtml(isAr ? "لا توجد مناظرات مطابقة للفلتر الحالي" : "No scenario videos match the current filter")}</h3>
      </div>
    `;
    return;
  }

  videos.forEach((v, idx) => {
    const isReady = v.stage1_status === "completed";
    const statusBadge = isReady
      ? `<span class="status-pill-ready">${isAr ? "سيناريو مكتمل صوتياً • M5/M6" : "DIARIZED SCENARIO • M5/M6"}</span>`
      : `<span class="status-pill-queued">${isAr ? "في قائمة المعالجة • M0" : "CATALOG QUEUED • M0"}</span>`;

    const cardEl = document.createElement("article");
    cardEl.className = idx >= 6 ? "scenario-video-card dossier-card-deferred" : "scenario-video-card";

    cardEl.innerHTML = `
      <div class="scenario-card-main">
        <div class="scenario-card-badges">
          ${statusBadge}
          <span class="badge-episode">${escapeHtml(v.video_id)}</span>
          ${v.is_short ? `<span class="badge-episode">#SHORT</span>` : ""}
          ${v.qa_count > 0 ? `<span class="badge-pillar">${v.qa_count} Q&A</span>` : ""}
        </div>
        <h3 class="scenario-card-title" dir="rtl">${escapeHtml(v.title_ar || v.video_id)}</h3>
        <div class="scenario-metrics-row">
          <span>DURATION: ${escapeHtml(formatDurationSec(v.duration_sec))}</span>
          <span>SPEAKERS: ${Number(v.speakers_count || 0)}</span>
          <span>TURNS: ${Number(v.turns_count || 0)}</span>
          <span>OVERLAPS UNMIXED: ${Number(v.overlap_windows_count || 0)}</span>
          <span>PEDESTRIANS ISOLATED: ${Number(v.pedestrian_turns_count || 0)}</span>
        </div>
      </div>
      <div class="scenario-card-actions">
        ${
          v.qa_count > 0
            ? `<button type="button" class="btn-outline-action" onclick="setVideoFilter('${escapeJsStr(v.video_id)}', '${escapeJsStr(v.title_ar || v.video_id)}')">
                <span>${escapeHtml(t.filterVideoQaBtn)} (${v.qa_count})</span>
              </button>`
            : ""
        }
        <button type="button" class="btn-primary-emerald" onclick="openScenarioStudio('${escapeJsStr(v.video_id)}', '', 0)">
          ${SVG_ICONS.scenario}
          <span>${escapeHtml(t.openStudioBtn)}</span>
        </button>
      </div>
    `;
    container.appendChild(cardEl);
  });
}

/* ==========================================================================
   INTERACTIVE SCENARIO TRANSCRIPT STUDIO MODAL & SYNCED YOUTUBE PLAYER
   ========================================================================== */

function openScenarioStudio(videoId, targetTurnId = "", startSec = 0) {
  if (!videoId) return;
  const dialog = document.getElementById("scenario-studio-dialog");
  if (!dialog) return;

  activeTargetTurnId = targetTurnId || "";
  activeStudioEpisode = "all";
  const filterInp = document.getElementById("studio-turn-filter");
  if (filterInp) filterInp.value = "";

  if (!dialog.open) {
    dialog.showModal();
  }

  fetch(`/api/videos/${encodeURIComponent(videoId)}/scenario?include_pedestrians=true`)
    .then((r) => {
      if (!r.ok) throw new Error("Scenario not found");
      return r.json();
    })
    .then((payload) => {
      activeScenarioPayload = payload;
      renderStudioContents(payload);
      mountOrSeekStudioPlayer(videoId, startSec);

      if (activeTargetTurnId) {
        setTimeout(() => scrollToStudioTurn(activeTargetTurnId), 120);
      }
    })
    .catch((err) => console.error("Studio load error:", err));
}

function closeScenarioStudio() {
  const dialog = document.getElementById("scenario-studio-dialog");
  if (dialog && dialog.open) {
    dialog.close();
  }
}

function renderStudioContents(payload) {
  const t = translations[currentLang];
  const isAr = currentLang === "ar";
  const video = payload.video || {};
  const speakers = payload.speakers || [];
  const turns = payload.turns || [];
  const qas = payload.qa_units || [];

  setText("scenario-studio-title", video.title_ar || video.video_id || "Scenario Studio");

  const metaEl = document.getElementById("scenario-studio-meta");
  if (metaEl) {
    metaEl.innerHTML = `
      <span class="badge-serial">${escapeHtml(video.video_id || "")}</span>
      <span>DURATION: ${escapeHtml(formatDurationSec(video.duration_sec))}</span>
      <span>TURNS: ${turns.length}</span>
      <span>OVERLAPS: ${Number(video.overlap_windows_count || 0)}</span>
      <span>PEDESTRIANS: ${Number(video.pedestrian_turns_count || 0)}</span>
    `;
  }

  // Update Pedestrian Checkbox label with live count
  const pedCount = turns.filter((x) => Number(x.is_pedestrian_interruption) === 1).length;
  setText("hide-pedestrians-label", `${t.hidePedestriansBase} (${pedCount})`);

  // Speaker Roster
  const spkList = document.getElementById("studio-speakers-list");
  if (spkList) {
    if (speakers.length) {
      spkList.innerHTML = speakers
        .map((s) => {
          const roleCls = String(s.role).includes("DAEE")
            ? "role-daee"
            : s.role === "PEDESTRIAN"
            ? ""
            : "role-inter";
          const label = isAr ? s.display_name_ar || s.display_name_en : s.display_name_en || s.display_name_ar;
          return `
            <div class="studio-speaker-item ${roleCls}">
              <strong>${escapeHtml(label)}</strong>
              <span class="studio-speaker-time">${Math.round(Number(s.speaking_sec || 0))}s · ${s.turn_count || 0}t</span>
            </div>
          `;
        })
        .join("");
    } else {
      spkList.innerHTML = `<div class="studio-speaker-time">${escapeHtml(isAr ? "في انتظار التمييز الصوتي (M3)" : "Pending NeMo MSDD Diarization (M3)")}</div>`;
    }
  }

  // Extracted Q&As inside this video
  const qaPanel = document.getElementById("studio-qa-panel");
  const qaJumps = document.getElementById("studio-qa-jumps");
  if (qaPanel && qaJumps) {
    if (qas.length) {
      qaPanel.hidden = false;
      qaJumps.innerHTML = qas
        .map((qa) => {
          const qTitle = isAr ? qa.question_ar || qa.question_en : qa.question_en || qa.question_ar;
          const tid = Array.isArray(qa.source_turn_ids) && qa.source_turn_ids.length ? qa.source_turn_ids[0] : "";
          return `
            <button type="button" class="exchange-step-item" onclick="seekStudioToTurn('${escapeJsStr(tid)}', ${Number(qa.timestamp_seconds || 0)})">
              <div class="exchange-step-meta">
                <span class="exchange-role-daee">${escapeHtml(qa.timestamp_formatted || "00:00")}</span>
                <span>EP ${qa.episode_index || 1}</span>
              </div>
              <div class="exchange-step-text">${escapeHtml(qTitle)}</div>
            </button>
          `;
        })
        .join("");
    } else {
      qaPanel.hidden = true;
    }
  }

  // Episode Tabs
  const epTabs = document.getElementById("studio-episode-tabs");
  if (epTabs) {
    const episodes = Array.from(new Set(turns.map((x) => Number(x.episode_index || 1)))).sort((a, b) => a - b);
    if (episodes.length > 1) {
      epTabs.innerHTML = `
        <button type="button" class="episode-tab-btn ${activeStudioEpisode === "all" ? "active" : ""}" onclick="setStudioEpisode('all')">
          ${escapeHtml(t.allEpisodesLabel)}
        </button>
        ${episodes
          .map(
            (ep) => `
          <button type="button" class="episode-tab-btn ${activeStudioEpisode === ep ? "active" : ""}" onclick="setStudioEpisode(${ep})">
            ${escapeHtml(t.episodePrefix)} ${ep}
          </button>
        `
          )
          .join("")}
      `;
    } else {
      epTabs.innerHTML = "";
    }
  }

  renderStudioTurns();
}

function setStudioEpisode(ep) {
  activeStudioEpisode = ep;
  if (activeScenarioPayload) {
    renderStudioContents(activeScenarioPayload);
  }
}

function renderSingleTurnCardHtml(turn, isDeferred, isAr) {
  const role = String(turn.speaker_role || "INTERLOCUTOR");
  const roleClass = role.includes("DAEE")
    ? "turn-daee"
    : role === "PEDESTRIAN"
    ? "turn-pedestrian"
    : "turn-inter";
  const targetClass = turn.turn_id === activeTargetTurnId ? "is-targeted" : "";
  const deferClass = isDeferred && !targetClass ? "scenario-turn-deferred" : "";
  const spkLabel = isAr
    ? turn.speaker_label_ar || turn.speaker_label_en || role
    : turn.speaker_label_en || role;
  const sec = Math.floor(Number(turn.start_ms || 0) / 1000);
  const txt = turn.text_original || turn.text_ar || "";

  return `
    <div
      class="scenario-turn ${roleClass} ${targetClass} ${deferClass}"
      id="studio-turn-${escapeHtml(turn.turn_id)}"
      data-turn-id="${escapeHtml(turn.turn_id)}"
      data-start-ms="${Number(turn.start_ms || 0)}"
      data-end-ms="${Number(turn.end_ms || 0)}"
      onclick="seekStudioToTurn('${escapeJsStr(turn.turn_id)}', ${sec})"
    >
      <div class="turn-header-line">
        <span class="turn-speaker-tag">${escapeHtml(spkLabel)}</span>
        <span class="turn-time-badge">${escapeHtml(turn.start_fmt || "00:00")} → ${escapeHtml(turn.end_fmt || "00:00")}</span>
      </div>
      <div class="turn-body-text">${escapeHtml(txt)}</div>
    </div>
  `;
}

function renderStudioTurns() {
  const viewport = document.getElementById("studio-turns-viewport");
  if (!viewport || !activeScenarioPayload) return;

  const t = translations[currentLang];
  const isAr = currentLang === "ar";
  const allTurns = activeScenarioPayload.turns || [];

  if (!allTurns.length) {
    viewport.innerHTML = `
      <div class="empty-dossier-state" style="margin:auto;">
        <p style="max-width:56ch; margin:0 auto;">${escapeHtml(t.queuedScenarioNotice)}</p>
      </div>
    `;
    return;
  }

  const hidePeds = Boolean(document.getElementById("hide-pedestrians-checkbox")?.checked);
  const inlineQuery = (document.getElementById("studio-turn-filter")?.value || "").trim().toLowerCase();

  const filtered = allTurns.filter((turn) => {
    if (hidePeds && Number(turn.is_pedestrian_interruption) === 1 && turn.turn_id !== activeTargetTurnId) {
      return false;
    }
    if (activeStudioEpisode !== "all" && Number(turn.episode_index || 1) !== Number(activeStudioEpisode)) {
      return false;
    }
    if (inlineQuery) {
      const hay = `${turn.text_original || ""} ${turn.text_ar || ""} ${turn.speaker_label_en || ""} ${turn.speaker_label_ar || ""}`.toLowerCase();
      if (!hay.includes(inlineQuery)) return false;
    }
    return true;
  });

  // Group simultaneous overlap turns sharing overlap_group_id into side-by-side split-track callouts
  const blocks = [];
  let i = 0;
  let renderIdx = 0;
  while (i < filtered.length) {
    const curr = filtered[i];
    const ovGroup = curr.is_overlap && curr.overlap_group_id ? curr.overlap_group_id : "";

    if (ovGroup) {
      const groupTurns = [curr];
      let j = i + 1;
      while (j < filtered.length && filtered[j].overlap_group_id === ovGroup) {
        groupTurns.push(filtered[j]);
        j++;
      }

      if (groupTurns.length >= 2) {
        blocks.push(`
          <div class="overlap-split-callout ${renderIdx >= 15 ? "scenario-turn-deferred" : ""}">
            <div class="overlap-callout-bar">
              <span>${SVG_ICONS.overlap} ${escapeHtml(t.overlapBannerLabel)}</span>
              <span>${escapeHtml(groupTurns[0].start_fmt || "00:00")}</span>
            </div>
            <div class="overlap-tracks-grid">
              ${groupTurns.map((gt) => renderSingleTurnCardHtml(gt, false, isAr)).join("")}
            </div>
          </div>
        `);
        i = j;
        renderIdx += groupTurns.length;
        continue;
      }
    }

    blocks.push(renderSingleTurnCardHtml(curr, renderIdx >= 15, isAr));
    i++;
    renderIdx++;
  }

  viewport.innerHTML = blocks.join("");
}

function scrollToStudioTurn(turnId) {
  if (!turnId) return;
  const el = document.getElementById(`studio-turn-${turnId}`);
  if (!el) return;
  // Remove deferred content-visibility before measuring scroll offset
  el.classList.remove("scenario-turn-deferred");
  el.closest(".scenario-turn-deferred")?.classList.remove("scenario-turn-deferred");
  el.scrollIntoView({ behavior: "smooth", block: "center" });
}

function seekStudioToTurn(turnId, seconds) {
  activeTargetTurnId = turnId || "";
  document.querySelectorAll("#studio-turns-viewport .scenario-turn.is-targeted").forEach((n) => n.classList.remove("is-targeted"));
  if (turnId) {
    const targetEl = document.getElementById(`studio-turn-${turnId}`);
    if (targetEl) {
      targetEl.classList.add("is-targeted");
      scrollToStudioTurn(turnId);
    }
  }

  const videoId = activeScenarioPayload?.video?.video_id;
  if (!videoId) return;

  if (studioPlayer && typeof studioPlayer.seekTo === "function") {
    try {
      studioPlayer.seekTo(seconds, true);
      studioPlayer.playVideo();
      return;
    } catch (_) {}
  }

  // Direct iframe src fallback if YouTube JS API hasn't initialized yet
  const iframe = document.querySelector("#studio-player-wrap iframe");
  if (iframe) {
    iframe.src = `https://www.youtube.com/embed/${encodeURIComponent(videoId)}?start=${Math.max(0, Math.floor(seconds))}&autoplay=1&enablejsapi=1&rel=0`;
  }
}

function mountOrSeekStudioPlayer(videoId, startSec = 0) {
  stopStudioSyncLoop();
  const cleanStart = Math.max(0, Math.floor(Number(startSec || 0)));

  if (window.YT && window.YT.Player) {
    studioYtApiReady = true;
    createOrUpdateYtPlayer(videoId, cleanStart);
    return;
  }

  // Render an immediate responsive embed iframe while loading YT IFrame API in background
  const wrap = document.getElementById("studio-player-wrap");
  if (wrap) {
    wrap.innerHTML = `
      <iframe
        id="studio-yt-iframe"
        src="https://www.youtube.com/embed/${encodeURIComponent(videoId)}?start=${cleanStart}&enablejsapi=1&rel=0"
        title="YouTube Debate Player"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen
      ></iframe>
    `;
  }

  if (!studioYtApiLoading) {
    studioYtApiLoading = true;
    window.onYouTubeIframeAPIReady = () => {
      studioYtApiReady = true;
      const openVid = activeScenarioPayload?.video?.video_id;
      const dialog = document.getElementById("scenario-studio-dialog");
      if (openVid && dialog && dialog.open) {
        createOrUpdateYtPlayer(openVid, cleanStart);
      }
    };
    const tag = document.createElement("script");
    tag.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(tag);
  }
}

function createOrUpdateYtPlayer(videoId, startSec) {
  const wrap = document.getElementById("studio-player-wrap");
  if (!wrap) return;
  wrap.innerHTML = `<div id="studio-yt-player"></div>`;

  studioPlayer = new window.YT.Player("studio-yt-player", {
    videoId,
    playerVars: {
      start: startSec,
      rel: 0,
      modestbranding: 1
    },
    events: {
      onReady: () => {
        startStudioSyncLoop();
      },
      onStateChange: (ev) => {
        if (ev.data === 1) {
          startStudioSyncLoop();
        }
      }
    }
  });
}

function startStudioSyncLoop() {
  stopStudioSyncLoop();
  studioSyncTimer = setInterval(() => {
    if (!studioPlayer || typeof studioPlayer.getCurrentTime !== "function") return;
    let curSec = 0;
    try {
      curSec = studioPlayer.getCurrentTime();
    } catch (_) {
      return;
    }
    const curMs = Math.round(curSec * 1000);
    const turnNodes = document.querySelectorAll("#studio-turns-viewport .scenario-turn");
    let matchedId = "";

    for (const node of turnNodes) {
      const s = Number(node.getAttribute("data-start-ms") || 0);
      const e = Number(node.getAttribute("data-end-ms") || 0);
      const active = curMs >= s && curMs <= e + 250;
      node.classList.toggle("is-speaking", active);
      if (active && !matchedId) {
        matchedId = node.getAttribute("data-turn-id") || "";
      }
    }

    if (matchedId && matchedId !== lastHighlightedTurnId) {
      lastHighlightedTurnId = matchedId;
      const autoFollow = Boolean(document.getElementById("auto-follow-checkbox")?.checked);
      if (autoFollow) {
        const activeEl = document.getElementById(`studio-turn-${matchedId}`);
        if (activeEl) {
          activeEl.classList.remove("scenario-turn-deferred");
          activeEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      }
    }
  }, 250);
}

function stopStudioSyncLoop() {
  if (studioSyncTimer) {
    clearInterval(studioSyncTimer);
    studioSyncTimer = null;
  }
}

/* ==========================================================================
   SMART DIALECTICAL ASSISTANT
   ========================================================================== */

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
    .then((r) => r.json())
    .then((data) => {
      const isAr = data.language === "ar" || currentLang === "ar";
      const t = translations[isAr ? "ar" : "en"];
      const pkg = data.response_package;

      let botResponseHtml = "";
      if (pkg) {
        const noticeHtml =
          data.status === "partial_match" && data.message
            ? `<div class="socratic-turn-panel" style="margin-bottom:0.85rem; padding:0.65rem 0.85rem; font-size:0.84rem; font-weight:600; color:var(--socratic-ochre);">${escapeHtml(data.message)}</div>`
            : "";

        const cleanPremises = (pkg.logical_premises || [])
          .map((p) => String(p || "").trim())
          .filter((p) => p.length > 0);
        const premisesHtml = cleanPremises.length
          ? `
            <div style="margin-top:0.85rem;">
              <div class="proof-section-bar">
                <span>${SVG_ICONS.proof} ${escapeHtml(t.premisesLabel)}</span>
                <span>CONFIDENCE: ${escapeHtml(String(data.confidence_score || 95))}%</span>
              </div>
              <ol class="syllogism-list" style="margin-top:0.45rem;">
                ${cleanPremises
                  .map((p, idx) => {
                    const isLast = idx === cleanPremises.length - 1 && cleanPremises.length > 1;
                    const badgeCode = isLast ? (isAr ? "النتيجة" : "Conclusion") : `${idx + 1}`;
                    const badgeClass = isLast ? "step-badge step-badge-conclusion" : "step-badge";
                    return `
                      <li class="syllogism-step">
                        <span class="${badgeClass}">${badgeCode}</span>
                        <span>${escapeHtml(p)}</span>
                      </li>
                    `;
                  })
                  .join("")}
              </ol>
            </div>
          `
          : "";

        const fallacyHtml =
          pkg.fallacy_exposed && !isNoneText(pkg.fallacy_exposed)
            ? `
              <div class="fallacy-panel" style="margin-top:0.75rem;">
                <div class="fallacy-panel-bar">${SVG_ICONS.fallacy} <span>${escapeHtml(t.fallacyLabel)}</span></div>
                <div class="fallacy-panel-body">${escapeHtml(pkg.fallacy_exposed)}</div>
              </div>
            `
            : "";

        const socraticHtml =
          pkg.socratic_question && !isNoneText(pkg.socratic_question)
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

        const analogyHtml =
          pkg.analogy && !isNoneText(pkg.analogy)
            ? `
              <div class="analogy-panel" style="margin-top:0.75rem;">
                <div class="analogy-bar">${SVG_ICONS.analogy} <span>${escapeHtml(t.analogyLabel)}</span></div>
                <div class="analogy-body">${escapeHtml(pkg.analogy)}</div>
              </div>
            `
            : "";

        const firstTid = Array.isArray(pkg.source_turn_ids) && pkg.source_turn_ids.length ? pkg.source_turn_ids[0] : "";

        const relatedHtml =
          data.related_questions && data.related_questions.length
            ? `
              <div style="margin-top:0.9rem; padding-top:0.75rem; border-top:1px dashed var(--border-hairline);">
                <div style="font-family:var(--font-mono); font-size:0.75rem; font-weight:700; text-transform:uppercase; color:var(--ink-muted); margin-bottom:0.4rem;">
                  ${isAr ? "ملفات ذات صلة في الأرشيف:" : "Related Dossiers in Corpus:"}
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:0.4rem;">
                  ${data.related_questions
                    .map(
                      (rq) => `
                    <button type="button" class="btn-outline-action" onclick="askAssistant('${escapeJsStr(rq.question)}')">
                      ${escapeHtml(rq.question)}
                    </button>
                  `
                    )
                    .join("")}
                </div>
              </div>
            `
            : "";

        botResponseHtml = `
          ${noticeHtml}
          <div class="bot-msg-header">
            <div style="display:flex; gap:0.4rem; flex-wrap:wrap;">
              ${pkg.daee_name ? `<span class="badge-daee">${SVG_ICONS.speaker}<span>${escapeHtml(pkg.daee_name)}</span></span>` : ""}
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
            <div class="proof-actions">
              ${
                pkg.video_id
                  ? `<button type="button" class="btn-outline-action" onclick="openScenarioStudio('${escapeJsStr(pkg.video_id)}', '${escapeJsStr(firstTid)}', 0)">
                      ${SVG_ICONS.scenario}
                      <span>${escapeHtml(t.scenarioJumpBtn)}</span>
                    </button>`
                  : ""
              }
              <a href="${escapeHtml(pkg.timestamp_url || "#")}" target="_blank" rel="noopener" class="btn-primary-emerald">
                ${SVG_ICONS.play}
                <span>${escapeHtml(t.watchBtn)} ${escapeHtml(pkg.timestamp || "00:00")}</span>
              </a>
            </div>
          </div>
          ${relatedHtml}
        `;
      } else {
        const scenHits = Array.isArray(data.scenario_matches) ? data.scenario_matches : [];
        const scenFallbackHtml = scenHits.length
          ? `
            <div style="margin-top:0.85rem; display:flex; flex-direction:column; gap:0.45rem;">
              <div style="font-family:var(--font-mono); font-size:0.74rem; font-weight:700; color:var(--emerald-primary);">
                ${isAr ? "تطابقات مباشرة من نصوص المناظرات المميزة صوتياً:" : "Direct Stage-1 Diarized Scenario Matches:"}
              </div>
              ${scenHits
                .map((sh) => {
                  const sec = Math.floor(Number(sh.start_ms || 0) / 1000);
                  return `
                    <button type="button" class="turn-hit-row" onclick="openScenarioStudio('${escapeJsStr(sh.video_id)}', '${escapeJsStr(sh.turn_id)}', ${sec})">
                      <div class="turn-hit-top">
                        <span>[${escapeHtml(sh.start_fmt || "00:00")}] ${escapeHtml(sh.speaker_label_en || sh.speaker_role)}</span>
                        <span class="ar-font">${escapeHtml((sh.video_title_ar || "").slice(0, 50))}</span>
                      </div>
                      <div style="font-size:0.94rem;">${escapeHtml(sh.text_original || "")}</div>
                    </button>
                  `;
                })
                .join("")}
            </div>
          `
          : "";

        const suggestionsHtml =
          data.related_questions && data.related_questions.length
            ? `
              <div style="margin-top:0.85rem; display:flex; flex-direction:column; gap:0.4rem;">
                ${data.related_questions
                  .map(
                    (q) => `
                  <button type="button" class="starter-chip-btn" onclick="askAssistant('${escapeJsStr(q.question)}')">
                    ${escapeHtml(q.question)}
                  </button>
                `
                  )
                  .join("")}
              </div>
            `
            : "";
        botResponseHtml = `
          <p style="font-size:0.9rem; color:var(--ink-primary); white-space:pre-line;">${escapeHtml(data.message || "")}</p>
          ${scenFallbackHtml}
          ${suggestionsHtml}
        `;
      }

      const botMsgEl = document.createElement("div");
      botMsgEl.className = "message assistant-msg";
      botMsgEl.innerHTML = `<div class="msg-content">${botResponseHtml}</div>`;
      messagesBox.appendChild(botMsgEl);
      messagesBox.scrollTop = messagesBox.scrollHeight;
    })
    .catch((err) => console.error("Chat error:", err));
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
