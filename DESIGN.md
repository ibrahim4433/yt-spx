---
name: Dawah Dialectics
description: Bilingual Islamic Apologetics & Street Debate Claim-and-Proof Dossier System
colors:
  emerald-primary: "#0F523E"
  emerald-hover: "#0A3B2C"
  emerald-wash: "#E8F4EF"
  emerald-border: "#B0D6C9"
  rust-primary: "#A62B1D"
  rust-wash: "#FDF3F1"
  rust-border: "#E6B8B0"
  socratic-ochre: "#8A4B08"
  socratic-wash: "#FEF9ED"
  socratic-border: "#DEC38C"
  bg-primary: "#EFF2F0"
  bg-rail: "#E4EAE6"
  bg-elevated: "#F7F9F8"
  bg-surface: "#FFFFFF"
  bg-subtle: "#E8ECE9"
  ink-primary: "#141716"
  ink-secondary: "#373D3A"
  ink-muted: "#525B56"
  border-hairline: "#CBD4CE"
  border-strong: "#141716"
typography:
  display:
    fontFamily: "Public Sans, Noto Naskh Arabic, sans-serif"
    fontSize: "clamp(1.35rem, 1.1rem + 0.9vw, 1.75rem)"
    fontWeight: 800
    lineHeight: 1.25
    letterSpacing: "-0.025em"
  headline:
    fontFamily: "Public Sans, Amiri, Noto Naskh Arabic, sans-serif"
    fontSize: "clamp(1.15rem, 1rem + 0.45vw, 1.38rem)"
    fontWeight: 800
    lineHeight: 1.32
    letterSpacing: "-0.02em"
  title:
    fontFamily: "Public Sans, Noto Naskh Arabic, sans-serif"
    fontSize: "1.08rem"
    fontWeight: 800
    lineHeight: 1.25
  body:
    fontFamily: "Public Sans, Noto Naskh Arabic, sans-serif"
    fontSize: "0.94rem"
    fontWeight: 400
    lineHeight: 1.65
  label:
    fontFamily: "Chivo Mono, monospace"
    fontSize: "0.75rem"
    fontWeight: 700
    lineHeight: 1.4
    letterSpacing: "0.04em"
rounded:
  xs: "4px"
  sm: "6px"
  md: "8px"
  lg: "10px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.emerald-primary}"
    textColor: "{colors.bg-surface}"
    rounded: "{rounded.sm}"
    padding: "8px 15px"
  button-primary-hover:
    backgroundColor: "{colors.emerald-hover}"
  button-outline:
    backgroundColor: "{colors.bg-surface}"
    textColor: "{colors.ink-primary}"
    rounded: "{rounded.sm}"
    padding: "8px 12px"
  chip-pillar:
    backgroundColor: "{colors.emerald-wash}"
    textColor: "{colors.emerald-primary}"
    rounded: "{rounded.xs}"
    padding: "3px 10px"
  card-dossier:
    backgroundColor: "{colors.bg-surface}"
    textColor: "{colors.ink-primary}"
    rounded: "{rounded.lg}"
    padding: "22px 24px"
---

# Design System: Dawah Dialectics

## Overview

**Creative North Star: "The Architectural Claim-and-Proof Dossier"**

Dawah Dialectics replaces generic dark-slate AI chatbot aesthetics with an editorial, high-contrast Claim-and-Proof Dossier system built for seekers, skeptics, and researchers examining real street-debate arguments in English and Arabic. Every philosophical or theological doubt is decomposed visually into two paired structural zones: the **Diagnostic & Socratic Turn** column (Core Thesis, Exposed Fallacy, Socratic Counter-Question, and Verbatim Street Quote) and the **Deductive Proof & Evidence** column (Numbered Syllogism Steps `P1 → P2 → ∴`, Concrete Real-World Analogy, Concept Tags, and Timestamped YouTube Video Proof).

The system treats English (LTR), Arabic (RTL), and Side-by-Side Parallel Study (`EN ⇄ عربي`) as first-class reading modes controlled from a unified 3-way segmented bar in the command header and per-card quick toggles.

**Key Characteristics:**
- **Paired Claim-vs-Proof Dossier Grid:** Every Q&A unit separates the skeptic's objection and its exposed fallacy from the numbered deductive syllogism and concrete analogy.
- **Semantic Dialectical Color Coding:** Mineral sage-stone neutrals (`#EFF2F0`, `#E4EAE6`) anchor the canvas while three functional colors encode logical roles: Deep Emerald-Oxide (`#0F523E`) for valid deductive proofs, Terracotta-Rust (`#A62B1D`) for exposed fallacies, and Warm Bronze-Ochre (`#8A4B08`) for Socratic counter-questions.
- **First-Class Bilingual & Parallel Typography:** Pairs `Public Sans` with `Noto Naskh Arabic` and `Amiri` for classical Arabic clarity, supported by `Chivo Mono` (`tabular-nums`) for serial IDs (`QA-001`..`QA-607`), live corpus counts, and video timecodes.

## Colors

The palette uses a mineral sage-limestone neutral scale paired with three strictly semantic dialectical accents.

### Primary
- **Deep Emerald-Oxide** (`#0F523E`): Primary proof accent, deductive syllogism step badges (`P1`, `P2`, `∴`), active pillar indicators, and primary YouTube timestamp proof actions.
- **Deep Emerald Hover** (`#0A3B2C`): Interactive hover state for primary actions.
- **Emerald Wash** (`#E8F4EF`) & **Emerald Border** (`#B0D6C9`): Subtle radial proof pane tint and concrete analogy callout surface.

### Secondary
- **Terracotta-Rust** (`#A62B1D`): Exclusively signals the **Fallacy Exposed** (`المغالطة المكشوفة`) header bar and icon.
- **Terracotta Wash** (`#FDF3F1`) & **Terracotta Border** (`#E6B8B0`): Surface and 1px structural border for exposed fallacy panels.

### Tertiary
- **Socratic Bronze-Ochre** (`#8A4B08`): Exclusively signals **The Socratic Turn** (`السؤال السقراطي المعاكس`) counter-question panel, quotation glyph, and micro-actions (`Copy` / `Test in Bot`).
- **Socratic Wash** (`#FEF9ED`) & **Socratic Border** (`#DEC38C`): Warm parchment surface and 1px border for Socratic counter-question callouts.

### Neutral
- **Mineral Limestone Ground** (`#EFF2F0`): Main application canvas background.
- **Architectural Facet Rail** (`#E4EAE6`): Persistent left sidebar background housing the Beta extraction ledger and pillar/interlocutor filters.
- **Elevated Banner Sheet** (`#F7F9F8`): Sticky command bar and Dossier top question banner.
- **Pure Dossier Surface** (`#FFFFFF`): Main Claim-and-Proof card surface.
- **Charcoal Ink** (`#141716`): Primary headings, core thesis prose, and architectural card borders.
- **Slate Secondary Ink** (`#373D3A`) & **Muted Ink** (`#525B56`): Supporting metadata, Arabic parallel secondary prose, and labels (all maintaining ≥4.8:1 contrast).
- **Structural Hairline** (`#CBD4CE`): 1px internal grid dividers and panel borders.

### Named Rules
**The Dialectical Role Color Rule.** Emerald-Oxide (`#0F523E`) is reserved for valid deductive proofs and primary verification actions, Terracotta-Rust (`#A62B1D`) is reserved for exposed logical fallacies, and Bronze-Ochre (`#8A4B08`) is reserved for Socratic counter-questions. Never mix or swap these semantic roles.

## Typography

**Display & UI Font:** `Public Sans` (with `-apple-system, BlinkMacSystemFont, sans-serif` fallback)
**Arabic Display & Prose Fonts:** `Amiri` (display questions & Socratic turns) and `Noto Naskh Arabic` (UI & body prose)
**Label / Monospace Font:** `Chivo Mono` (`font-variant-numeric: tabular-nums`)

**Character:** Authoritative, editorial, and effortlessly scannable across both Latin LTR and Arabic RTL scripts, with strict tabular alignment on every serial code (`QA-001`), corpus count (`607 / 2,348`), and debate timestamp (`04:18`).

### Hierarchy
- **Display** (`800`, `clamp(1.35rem, 1.1rem + 0.9vw, 1.75rem)`, `1.25`): View titles (`Claim & Proof Dialectical Dossier`).
- **Headline** (`800` Latin / `700` Amiri Arabic, `clamp(1.15rem, 1rem + 0.45vw, 1.38rem)`, `1.32`–`1.42`): Core theological doubt / question in the Dossier banner.
- **Title** (`800`, `1.08rem`, `1.25`): Brand title and section sheet headings.
- **Body** (`400`–`500`, `0.94rem`, `1.65`, max `68ch`–`74ch`): Core thesis summaries, deductive premise steps, and analogies.
- **Label** (`700`, `0.75rem`, `0.04em` tracking, uppercase): Panel header bars (`FALLACY EXPOSED`, `SOCRATIC COUNTER-QUESTION`, `DEDUCTIVE SYLLOGISM`), serial badges, and facet headers.

### Named Rules
**The Minimum 12px Functional Floor Rule.** No functional label, badge, count, or metadata item may drop below `0.75rem` (`12px`), and no body prose may drop below `0.8125rem`–`0.875rem` (`13px`–`14px`).

## Layout

The application uses a two-column shell (`292px` sticky Left Architectural Facet Rail + `1fr` Main Workspace) topped by a sticky `64px` command header. The Dossier Stream uses CSS Container Queries (`container-type: inline-size`) so each `.dossier-card` splits into a two-column Claim-vs-Proof grid (`1fr 1.15fr`) at wide container widths and collapses cleanly to a single column below `860px` container width or `680px` viewport width.

## Elevation & Depth

Surfaces rely primarily on architectural `1px solid` borders (`#141716` outer card frame, `#CBD4CE` internal hairlines) and tonal surface layering (`#E4EAE6` rail → `#EFF2F0` canvas → `#F7F9F8` banner → `#FFFFFF` proof surface), supported by a soft ambient offset shadow.

### Shadow Vocabulary
- **Soft Resting Dossier** (`box-shadow: 0 10px 28px -6px rgba(20, 23, 22, 0.06), 0 2px 6px -1px rgba(20, 23, 22, 0.04)`): Default elevation for `.dossier-card`.
- **Elevated Hover Dossier** (`box-shadow: 0 16px 36px -8px rgba(20, 23, 22, 0.10)`): Hover state for `.dossier-card`.

### Named Rules
**The 1px Full-Frame Rule.** Callout panels (Fallacy, Socratic Turn, Analogy, Syllogism Steps) use full `1px solid` borders with a tinted header bar—never thick `3px` or `4px` colored `border-left` stripes or hard neobrutalist offset shadows.

## Shapes

Corners follow a restrained architectural scale: `4px` (`--radius-xs`) for serial IDs, step badges, and concept tags; `6px` (`--radius-sm`) for buttons, inputs, and syllogism step rows; `8px` (`--radius-md`) for internal diagnostic/proof panels; and `10px` (`--radius-lg`) for outer Dossier cards.

## Components

### Buttons
- **Shape:** Subtle architectural rounding (`6px` radius).
- **Primary (`btn-primary-emerald`):** Solid Deep Emerald-Oxide (`#0F523E`) fill, `#FFFFFF` text, `0.5rem 0.95rem` padding, inline 1.5px-stroke SVG icon.
- **Hover / Focus:** Transitions to `#0A3B2C` on hover; `2px solid #0F523E` focus ring with `2px` offset.
- **Outline (`btn-outline-action`):** `#FFFFFF` background, `1px solid #CBD4CE` border, shifting to `#141716` border on hover.

### Chips
- **Style:** Pillar badges use `#E8F4EF` fill with `1px solid #B0D6C9` and `#0F523E` text; interlocutor badges use `#E8ECE9` fill with `1px solid #CBD4CE`.
- **State:** Clicking any badge filters the Dossier archive immediately and renders a removable active filter pill in the toolbar.

### Cards / Containers
- **Corner Style:** `10px` outer radius (`--radius-lg`).
- **Background:** `#F7F9F8` top question banner over `#FFFFFF` split Claim/Proof body with a radial `#E8F4EF` wash in the top-right corner of the proof column.
- **Border:** `1px solid #141716` outer frame with `1px solid #CBD4CE` internal column and banner dividers.
- **Internal Padding:** `1.25rem 1.5rem` on desktop; `1rem` on mobile.

### Inputs / Fields
- **Style:** `1px solid #141716` border on `#FFFFFF` surface, `6px` radius, with inline search SVG and keyboard shortcut badge (`/`).
- **Focus:** `border-color: #0F523E` with `0 0 0 3px rgba(15, 82, 62, 0.14)` focus ring.

### Navigation
- **Top Command Bar:** Houses the 3 view tabs (`Claim & Proof Archive`, `Smart Dialectical Assistant`, `Corpus & Beta Ledger`), global instant search, and the **3-Way Segmented Language & Parallel Mode Switcher** (`EN` | `العربية` | `EN ⇄ عربي`).
- **Left Facet Rail:** Sticky `100dvh` rail with Beta extraction progress bar (`607 / 2,348`) and scrollable Pillar & Interlocutor facet buttons with tabular counts; converts to a slide-over drawer on viewports `≤1080px`.

### The Socratic Turn Pivot Panel
A dedicated dialectical component (`.socratic-turn-panel`) styled in warm parchment (`#FEF9ED`) with a `1px solid #DEC38C` border and a header strip containing the Socratic icon, bilingual title (`SOCRATIC COUNTER-QUESTION • السؤال السقراطي`), and two 1-click micro-actions (`Copy` and `Test in Bot`), framing the counter-question alongside a quotation glyph.

## Do's and Don'ts

### Do:
- **Do** pair every theological objection with its structured deductive syllogism (`P1`, `P2`, `∴`), exposed fallacy, Socratic counter-question, and exact YouTube timestamp (`MM:SS`).
- **Do** support all three reading modes (`EN`, `العربية`, and `EN ⇄ عربي Parallel`) seamlessly across the archive.
- **Do** use consistent 1.5px-stroke inline SVG icons and `tabular-nums` on all numeric counts and timestamps.

### Don't:
- **Don't** use Unicode emojis (`🕌`, `🔍`, `🧠`, `💡`, `❓`) anywhere in the interface.
- **Don't** place small uppercase kickers or eyebrows directly above `<h1>`–`<h4>` headings.
- **Don't** use thick colored `border-left` or `border-right` stripes (`>1px`) on callout boxes or cards.
