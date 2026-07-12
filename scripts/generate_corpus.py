"""
scripts/generate_corpus.py — Sprint 1: Corpus Authoring

Generates all synthetic Meridian Global Corp source documents.
Safe to rerun — skips files that already exist.

Usage:
    python scripts/generate_corpus.py            # generate everything
    python scripts/generate_corpus.py --dry-run  # print what would be generated

Requires:
    .env with GEMINI_API_KEY_A
    pip install reportlab Pillow gTTS pdfminer.six google-genai
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# ── Load env ────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_KEY = os.environ.get("GEMINI_API_KEY_A")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
CORPUS_DIR = Path(__file__).parent.parent / "corpus"

# ── Document manifest ────────────────────────────────────────────────────────

DOCUMENTS = [
    # ── HR policies — EN + HI + ZH ──────────────────────────────────────────
    {
        "bu": "hr", "slug": "expense_policy", "langs": ["en", "hi", "zh"], "type": "pdf",
        "title": "Employee Expense Reimbursement Policy",
        "topic": (
            "meal allowances (₹500/day domestic, ₹2,000/day international), "
            "travel reimbursement (economy class for flights under 6 hours, business class above), "
            "hotel cap ($150/night domestic, $250/night international), "
            "advance request procedure (submit 5 business days prior), "
            "receipts required above ₹200, manager approval within 5 business days, "
            "non-reimbursable items (alcohol, personal entertainment, fines)"
        ),
    },
    {
        "bu": "hr", "slug": "leave_policy", "langs": ["en", "hi", "zh"], "type": "pdf",
        "title": "Leave and Attendance Policy",
        "topic": (
            "earned leave 18 days/year (accrues monthly), sick leave 10 days/year, "
            "casual leave 6 days/year, maternity leave 26 weeks fully paid, "
            "paternity leave 5 days, carry-forward cap 30 days, "
            "encashment allowed up to 15 days/year, "
            "leave application process via HR portal with 7-day advance notice for planned leave"
        ),
    },
    {
        "bu": "hr", "slug": "code_of_conduct", "langs": ["en", "hi", "zh"], "type": "pdf",
        "title": "Employee Code of Conduct",
        "topic": (
            "workplace ethics and respect for all colleagues, "
            "zero-tolerance anti-harassment policy with reporting to HR within 48 hours, "
            "conflict of interest disclosure form required annually, "
            "confidentiality obligations for 2 years post-employment, "
            "social media guidelines (no disclosure of unreleased products), "
            "disciplinary steps: verbal warning → written warning → termination, "
            "whistleblower protection with anonymous reporting hotline 1800-555-0199"
        ),
    },

    # ── IT Security policies — EN + HI + ZH ─────────────────────────────────
    {
        "bu": "it_security", "slug": "access_policy", "langs": ["en", "hi", "zh"], "type": "pdf",
        "title": "System Access Control Policy",
        "topic": (
            "user account provisioning requires manager approval and IT ticket within 24 hours, "
            "passwords minimum 12 characters with uppercase, lowercase, number and symbol, "
            "MFA mandatory for all corporate systems since January 2025, "
            "privileged access review every 90 days, "
            "VPN required for all remote access outside corporate network, "
            "offboarding access revocation within 4 hours of last working day, "
            "shared accounts prohibited, contractor accounts expire automatically after 90 days"
        ),
    },
    {
        "bu": "it_security", "slug": "incident_response", "langs": ["en", "hi", "zh"], "type": "pdf",
        "title": "Cybersecurity Incident Response Policy",
        "topic": (
            "P1 Critical (data breach, ransomware): 15-minute response, immediate CISO escalation; "
            "P2 High (system compromise): 1-hour response; "
            "P3 Medium (malware detection): 4-hour response; "
            "P4 Low (policy violation): 5-business-day response. "
            "Containment steps: isolate affected system, preserve logs, notify legal. "
            "Regulatory notification to CERT-In within 6 hours of P1 discovery. "
            "Post-incident review mandatory within 5 business days for P1/P2."
        ),
    },

    # ── Product specs — EN + ZH ──────────────────────────────────────────────
    {
        "bu": "product", "slug": "specsheet_v1", "langs": ["en", "zh"], "type": "pdf",
        "title": "Meridian Analytics Platform v1 — Product Specification",
        "topic": (
            "cross-lingual enterprise data query platform supporting English, Hindi, Chinese. "
            "API throughput 10,000 req/min, 99.9% uptime SLA, 256-bit AES encryption. "
            "Supported integrations: Salesforce, SAP, Oracle, AWS S3, Azure Blob. "
            "Pricing: Starter $299/mo (5 users), Professional $999/mo (25 users), Enterprise custom. "
            "ISO 27001 and SOC 2 Type II certified. Deployment: cloud-only for v1."
        ),
    },
    {
        "bu": "product", "slug": "specsheet_v2", "langs": ["en", "zh"], "type": "pdf",
        "title": "Meridian Analytics Platform v2 — Product Specification",
        "topic": (
            "v2 adds real-time streaming analytics (sub-100ms latency), multi-region deployment "
            "(US, EU, APAC), AI-powered query suggestions, custom dashboards (drag-and-drop), "
            "new REST and GraphQL APIs. Migration from v1: automated migration tool available. "
            "Breaking change: /v1/legacy endpoint deprecated, sunset October 2026. "
            "Pricing: v2 Starter $399/mo, Professional $1,299/mo, Enterprise custom with dedicated support."
        ),
    },
    {
        "bu": "product", "slug": "api_reference", "langs": ["en", "zh"], "type": "pdf",
        "title": "Meridian Platform REST API Reference",
        "topic": (
            "Authentication: OAuth 2.0 bearer tokens (24-hour expiry) and API keys. "
            "Core endpoints: POST /v2/query (natural language query), POST /v2/ingest (data upload), "
            "GET /v2/reports/{id} (fetch report), DELETE /v2/data/{dataset_id}. "
            "Rate limits: 1,000 req/min per API key, 10,000 req/min enterprise tier. "
            "Error codes: 400 Bad Request, 401 Unauthorized, 429 Rate Limited, 500 Server Error. "
            "Pagination: cursor-based with page_size max 100. Webhook support for async operations."
        ),
    },
    {
        "bu": "product", "slug": "roadmap_2026", "langs": ["en", "zh"], "type": "pdf",
        "title": "Meridian Product Roadmap 2026",
        "topic": (
            "Q1 2026: GraphQL API GA, SSO (SAML 2.0) integration, mobile app beta. "
            "Q2 2026: Advanced analytics (predictive models), custom embedding support, audit logging. "
            "Q3 2026: Meridian 3.0 launch — AI-native architecture, 10x performance, on-premise option. "
            "Q4 2026: Marketplace (100+ connectors), multi-tenant SaaS hardening, HIPAA compliance. "
            "Investment priorities: AI/ML capabilities (40% engineering), platform reliability (30%), "
            "enterprise features (20%), developer experience (10%)."
        ),
    },
    {
        "bu": "product", "slug": "release_notes_q3", "langs": ["en", "zh"], "type": "pdf",
        "title": "Meridian Platform Q3 2026 Release Notes — v2.4",
        "topic": (
            "New in v2.4: AI query suggestions (powered by fine-tuned LLM), bulk CSV export (up to 10M rows), "
            "RBAC enhancements (row-level security, column masking). "
            "47 bugs resolved including critical fix for query timeout on datasets >500GB. "
            "Performance: 40% faster query response via query plan caching. "
            "Breaking: /v1/legacy endpoint returns 410 Gone as of September 1 2026 — migrate to /v2. "
            "Known issue: GraphQL subscriptions may drop on network interruption (fix in v2.4.1). "
            "Upgrade: `pip install meridian-sdk==2.4.0`."
        ),
    },

    # ── Exec Comms slide decks — EN only ─────────────────────────────────────
    {
        "bu": "exec_comms", "slug": "q3_allhands", "langs": ["en"],
        "type": "slides", "n_slides": 5,
        "title": "Q3 2026 All-Hands Meeting",
        "topic": (
            "Q3 revenue $36M ARR (+23% YoY), 3 major enterprise wins (GlobalBank, TechCorp Asia, ManuPro), "
            "Meridian v2.4 shipped on time, 50 new hires (total headcount 1,240), "
            "H2 priorities: Meridian 3.0 launch, APAC expansion (Singapore office opens November), "
            "employee recognition: Engineering team of the quarter award"
        ),
    },
    {
        "bu": "exec_comms", "slug": "annual_review_2025", "langs": ["en"],
        "type": "slides", "n_slides": 5,
        "title": "Annual Review 2025 — Board Presentation",
        "topic": (
            "FY2025 $48M ARR (68% gross margin, up from 62%), 2,400 enterprise customers, "
            "NPS score 67 (industry average 42), launched in 12 new countries (total: 28), "
            "raised Series C $120M at $1.2B valuation, "
            "2026 guidance: $65M ARR target, 85% gross margin, IPO readiness assessment in Q4"
        ),
    },
    {
        "bu": "exec_comms", "slug": "product_launch_q4", "langs": ["en"],
        "type": "slides", "n_slides": 5,
        "title": "Q4 Product Launch — Meridian 3.0",
        "topic": (
            "Meridian 3.0 GA: October 15 2026, AI-native architecture (10x query performance), "
            "natural language in English/Hindi/Chinese, automatic schema detection, "
            "go-to-market: 500 design partners onboarded, $0 migration cost for existing customers, "
            "new pricing model (consumption-based), partner ecosystem (AWS, Azure, Google Cloud)"
        ),
    },
    {
        "bu": "exec_comms", "slug": "market_strategy_2026", "langs": ["en"],
        "type": "slides", "n_slides": 5,
        "title": "2026 Market Strategy — APAC Expansion",
        "topic": (
            "$2.3B TAM in APAC enterprise analytics, primary targets: financial services, healthcare, manufacturing, "
            "country entry: India (existing), Singapore (Q4 2026), Japan (Q1 2027), "
            "competitive positioning vs. Tableau and PowerBI (price/AI advantages), "
            "sales plan: 40 new AEs in APAC, channel partner program launch Q1 2027, "
            "3-year APAC revenue target: $30M by FY2028"
        ),
    },

    # ── Product slide decks — EN only ────────────────────────────────────────
    {
        "bu": "product", "slug": "demo_deck", "langs": ["en"],
        "type": "slides", "n_slides": 5,
        "title": "Meridian Platform — Enterprise Demo",
        "topic": (
            "company overview (founded 2018, 1,240 employees, $1.2B valuation), "
            "platform demo: natural language query 'show me Q3 revenue by region in Hindi', "
            "architecture: cloud-native, SOC 2 Type II, 256-bit AES, "
            "customer success: TechCorp Asia 60% cost reduction, GlobalBank 3x faster reporting, "
            "next steps: 30-day free trial, dedicated onboarding engineer assigned"
        ),
    },
    {
        "bu": "product", "slug": "tech_architecture", "langs": ["en"],
        "type": "slides", "n_slides": 5,
        "title": "Meridian Technical Architecture Deep-Dive",
        "topic": (
            "4-layer architecture: ingestion (Kafka, 1M events/sec), processing (Spark Streaming), "
            "storage (columnar + vector DB), API layer (REST + GraphQL + gRPC). "
            "Scalability: auto-scaling to 100k concurrent users, multi-region active-active. "
            "Security: zero-trust network, end-to-end encryption, SOC 2 / ISO 27001 / GDPR. "
            "Integration: 80+ native connectors, webhook support, SDK in Python/Java/Node"
        ),
    },
    {
        "bu": "product", "slug": "feature_overview_v2", "langs": ["en"],
        "type": "slides", "n_slides": 5,
        "title": "Meridian v2 Feature Overview",
        "topic": (
            "headline features: real-time streaming (<100ms latency), AI query suggestions, custom dashboards. "
            "v1 vs v2 comparison: API speed 3x faster, storage 40% cheaper, 25 new integrations added. "
            "Migration: automated tool migrates v1 configs in <30 minutes, zero downtime. "
            "Customer impact: avg 2.4x reduction in time-to-insight, 35% reduction in dashboard creation time. "
            "Roadmap beyond v2: GraphQL GA Q1 2027, on-premise Q3 2027"
        ),
    },
    {
        "bu": "product", "slug": "onboarding_slides", "langs": ["en"],
        "type": "slides", "n_slides": 5,
        "title": "Customer Onboarding — Getting Started",
        "topic": (
            "30-60-90 day plan: Day 30 first dashboard live, Day 60 5+ users active, Day 90 ROI measured. "
            "Setup checklist: API keys (10 min), SSO config (30 min), first data connector (1 hour). "
            "First dashboard walkthrough: drag-and-drop builder, 50+ chart types, share in 1 click. "
            "Support resources: docs.meridian.io, Slack community 12,000+ members, dedicated CSM for Enterprise. "
            "Success metrics: query volume, dashboard views, user adoption rate (target >80% in 90 days)"
        ),
    },

    # ── Audio clips — EN only ────────────────────────────────────────────────
    {
        "bu": "exec_comms", "slug": "q3_allhands", "langs": ["en"], "type": "audio",
        "title": "Q3 2026 All-Hands Recording",
        "topic": (
            "CEO opening: Q3 revenue $36 million ARR, 23 percent year-over-year growth. "
            "Three major enterprise wins: GlobalBank, TechCorp Asia, and ManuPro International. "
            "Engineering shipped Meridian version 2.4 on schedule with 47 bugs fixed and 40 percent "
            "query performance improvement. We welcomed 50 new colleagues this quarter, "
            "bringing our total headcount to 1,240 across 15 countries. "
            "For H2, our priorities are Meridian 3.0 general availability on October 15, "
            "opening our Singapore office in November, and achieving 85 percent gross margin by year-end. "
            "I want to thank each of you for your dedication and look forward to finishing the year strong."
        ),
    },
    {
        "bu": "exec_comms", "slug": "product_launch_q4", "langs": ["en"], "type": "audio",
        "title": "Q4 Product Launch Announcement",
        "topic": (
            "Attention all Meridian customers and partners. We are thrilled to announce "
            "the general availability of Meridian 3.0, launching October 15, 2026. "
            "Meridian 3.0 is our most significant release ever, featuring an AI-native query engine "
            "that delivers 10 times the performance of version 2, natural language querying "
            "in English, Hindi, and Chinese, automatic schema detection that eliminates manual "
            "data mapping, and a new consumption-based pricing model starting at 50 dollars per month. "
            "All existing version 2 customers will receive a complimentary migration with zero downtime. "
            "Visit meridian.io slash 3 to join 500 design partners already live on the platform."
        ),
    },
    {
        "bu": "exec_comms", "slug": "ceo_message_2026", "langs": ["en"], "type": "audio",
        "title": "CEO New Year Message 2026",
        "topic": (
            "To all 40,000 Meridian Global Corp employees across 28 countries: "
            "2025 was a defining year. We reached unicorn status with our 1.2 billion dollar valuation, "
            "grew to 2,400 enterprise customers, and earned a Net Promoter Score of 67, "
            "well above the industry average of 42. "
            "In 2026, our vision is clear: become the leading enterprise intelligence platform in Asia Pacific. "
            "We are opening our Singapore headquarters in November, "
            "launching Meridian 3.0 with AI-native capabilities, "
            "and announcing an employee equity refresh program for all full-time staff with over one year of tenure. "
            "Thank you for your innovation, your customer obsession, and your resilience. "
            "The best is yet to come."
        ),
    },
]


# ── Font setup ────────────────────────────────────────────────────────────────

def _setup_fonts() -> dict:
    """Register fonts for reportlab. Returns {'zh': fontname, 'hi': fontname}."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont

    fonts = {"en": "Helvetica", "zh": "Helvetica", "hi": "Helvetica"}

    # Chinese: built-in CID font (STSong-Light, Simplified Chinese)
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        fonts["zh"] = "STSong-Light"
    except Exception as e:
        print(f"  [WARN] Could not register Chinese CID font: {e}")

    # Hindi: Mangal TTF from Windows system fonts
    mangal_candidates = [
        r"C:\Windows\Fonts\mangal.ttf",
        r"C:\Windows\Fonts\Mangal.ttf",
    ]
    for path in mangal_candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("Mangal", path))
                fonts["hi"] = "Mangal"
                break
            except Exception as e:
                print(f"  [WARN] Could not register Mangal font from {path}: {e}")

    if fonts["hi"] == "Helvetica":
        print("  [WARN] Hindi font (Mangal) not found — Hindi PDFs will use Helvetica "
              "(text still stored as UTF-8 Unicode, readable by pdfminer)")

    return fonts


# ── Content generation via Gemini ────────────────────────────────────────────

def _generate_pdf_text(ds_client, lang: str, title: str, topic: str) -> str:
    """Call DeepSeek to write a corporate policy/spec document.

    NOTE: Gemini KEY_A is reserved for the runtime pipeline (PII Redactor, Router).
    All corpus generation goes through DeepSeek to avoid the 20 req/day free-tier cap.
    """
    lang_name = {"en": "English", "hi": "Hindi (Devanagari script)", "zh": "Simplified Chinese (Mandarin)"}[lang]
    prompt = (
        f"Write a realistic corporate document in {lang_name} for Meridian Global Corp, "
        f"a fictional 40,000-employee multinational technology company.\n\n"
        f"Document title: {title}\n"
        f"Content must cover: {topic}\n\n"
        f"Requirements:\n"
        f"- Write in {lang_name} throughout — no English headings in non-English documents\n"
        f"- Approximately 500 to 700 words\n"
        f"- Professional corporate language with specific numbers, dates, and names\n"
        f"- Use '===' before each section heading (e.g. === Section Title ===)\n"
        f"- No placeholders like [INSERT NAME] — write complete, realistic content\n"
        f"- Do not include any preamble or explanation — start directly with the document body"
    )
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return resp.text.strip()


def _generate_slide_texts(ds_client, title: str, topic: str, n_slides: int) -> list[dict]:
    """Call DeepSeek to create structured slide content. Returns list of {title, bullets}."""
    prompt = (
        f"Create content for a {n_slides}-slide corporate presentation in English "
        f"for Meridian Global Corp.\n\n"
        f"Presentation title: {title}\n"
        f"Topic details: {topic}\n\n"
        f"Format each slide EXACTLY as:\n"
        f"SLIDE N: [slide title]\n"
        f"\u2022 [bullet \u2014 max 15 words, include specific numbers/dates]\n"
        f"\u2022 [bullet]\n"
        f"\u2022 [bullet]\n\n"
        f"Rules:\n"
        f"- Exactly {n_slides} slides\n"
        f"- 3 to 5 bullets per slide\n"
        f"- No long paragraphs \u2014 bullets only\n"
        f"- Include specific metrics wherever possible"
    )
    resp = ds_client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    text = resp.choices[0].message.content or ""
    return _parse_slides(text.strip(), n_slides)


def _generate_audio_text(ds_client, title: str, topic: str) -> str:
    """Call DeepSeek to write a speech script for TTS."""
    prompt = (
        f"Write a 60-90 second spoken speech in natural English for a corporate audio recording.\n\n"
        f"Title: {title}\n"
        f"Content to cover: {topic}\n\n"
        f"Requirements:\n"
        f"- Write as spoken word \u2014 natural, warm, professional\n"
        f"- No stage directions, no [pause], no formatting\n"
        f"- Approximately 150 to 200 words\n"
        f"- Do not start with 'Hello' or 'Hi' \u2014 begin directly\n"
        f"- Output only the speech text, nothing else"
    )
    resp = ds_client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_slides(text: str, n_slides: int) -> list[dict]:
    """Parse Gemini slide output into list of {title, bullets}."""
    slides = []
    blocks = re.split(r"SLIDE\s+\d+:\s*", text, flags=re.IGNORECASE)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        slide_title = lines[0].rstrip(":")
        bullets = [
            l.lstrip("•·-– ").strip()
            for l in lines[1:]
            if l.lstrip("•·-– ").strip()
        ]
        if bullets:
            slides.append({"title": slide_title, "bullets": bullets})
    # Pad if Gemini returned fewer slides than requested
    while len(slides) < n_slides:
        slides.append({"title": f"Slide {len(slides)+1}", "bullets": ["Content pending"]})
    return slides[:n_slides]


# ── PDF writer ────────────────────────────────────────────────────────────────

def _write_pdf(filepath: Path, title: str, content: str, lang: str, fonts: dict) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors

    font = fonts.get(lang, "Helvetica")
    font_size = 10 if lang in ("zh",) else 11

    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    title_style = ParagraphStyle(
        "MeridianTitle", fontName=font, fontSize=16, leading=20,
        textColor=colors.HexColor("#1a2744"), spaceAfter=12, spaceBefore=0,
    )
    heading_style = ParagraphStyle(
        "MeridianHeading", fontName=font, fontSize=12, leading=16,
        textColor=colors.HexColor("#2d5a9e"), spaceAfter=6, spaceBefore=12,
    )
    body_style = ParagraphStyle(
        "MeridianBody", fontName=font, fontSize=font_size, leading=16,
        textColor=colors.HexColor("#333333"), spaceAfter=8,
    )
    footer_style = ParagraphStyle(
        "MeridianFooter", fontName="Helvetica", fontSize=8,
        textColor=colors.HexColor("#999999"), spaceAfter=0,
    )

    story = [
        Paragraph(title, title_style),
        Paragraph("Meridian Global Corp — Confidential Internal Document", footer_style),
        Spacer(1, 0.4 * cm),
    ]

    for line in content.splitlines():
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.2 * cm))
            continue
        if line.startswith("===") and line.endswith("==="):
            heading_text = line.strip("=").strip()
            story.append(Paragraph(heading_text, heading_style))
        else:
            # Escape XML special chars for reportlab
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe, body_style))

    doc.build(story)


# ── Slide PNG writer ──────────────────────────────────────────────────────────

def _write_slide(
    filepath: Path,
    deck_title: str,
    slide: dict,
    slide_num: int,
    total_slides: int,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1280, 720
    BG = (26, 39, 68)        # navy blue
    ACCENT = (45, 90, 158)   # mid blue
    WHITE = (255, 255, 255)
    LIGHT = (180, 200, 230)
    GRAY = (140, 160, 190)

    img = Image.new("RGB", (W, H), color=BG)
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([(0, 0), (W, 80)], fill=ACCENT)

    # Try to load a system font; fall back to default
    def _font(size: int):
        for name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]:
            for base in [r"C:\Windows\Fonts", r"/usr/share/fonts/truetype/dejavu"]:
                p = os.path.join(base, name)
                if os.path.exists(p):
                    try:
                        return ImageFont.truetype(p, size)
                    except Exception:
                        continue
        return ImageFont.load_default()

    font_sm = _font(18)
    font_md = _font(24)
    font_lg = _font(34)

    # Company name in header
    draw.text((30, 22), "MERIDIAN GLOBAL CORP", font=font_sm, fill=WHITE)
    # Slide counter top-right
    counter = f"{slide_num} / {total_slides}"
    draw.text((W - 90, 26), counter, font=font_sm, fill=LIGHT)

    # Deck subtitle
    draw.text((30, 100), deck_title.upper(), font=font_sm, fill=GRAY)

    # Slide title
    draw.text((30, 135), slide["title"], font=font_lg, fill=WHITE)

    # Accent line under title
    draw.rectangle([(30, 185), (W - 30, 188)], fill=ACCENT)

    # Bullets
    y = 210
    for bullet in slide["bullets"]:
        draw.text((50, y), "▶", font=font_md, fill=ACCENT)
        draw.text((85, y), bullet, font=font_md, fill=WHITE)
        y += 52
        if y > H - 80:
            break

    # Footer
    draw.rectangle([(0, H - 40), (W, H)], fill=(15, 25, 50))
    draw.text((30, H - 28), "CONFIDENTIAL — MERIDIAN GLOBAL CORP", font=font_sm, fill=GRAY)

    img.save(str(filepath), "PNG")


# ── Audio writer ──────────────────────────────────────────────────────────────

def _write_audio(filepath: Path, speech_text: str) -> None:
    from gtts import gTTS
    tts = gTTS(text=speech_text[:3000], lang="en", slow=False)
    tts.save(str(filepath))


# ── File path builder ─────────────────────────────────────────────────────────

def _pdf_path(bu: str, slug: str, lang: str) -> Path:
    return CORPUS_DIR / bu / f"{bu}_{slug}.{lang}.pdf"


def _slide_path(bu: str, slug: str, lang: str, n: int) -> Path:
    return CORPUS_DIR / bu / f"{bu}_{slug}.{lang}_slide{n:03d}.png"


def _audio_path(bu: str, slug: str, lang: str) -> Path:
    return CORPUS_DIR / bu / f"{bu}_{slug}.{lang}.mp3"


# ── Main orchestrator ─────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    if not GEMINI_KEY:
        print("ERROR: GEMINI_API_KEY_A not set in .env")
        sys.exit(1)
    if not DEEPSEEK_KEY:
        print("ERROR: DEEPSEEK_API_KEY not set in .env")
        sys.exit(1)

    from google import genai
    from openai import OpenAI

    gemini_client = genai.Client(api_key=GEMINI_KEY)
    deepseek_client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

    if not dry_run:
        fonts = _setup_fonts()
    else:
        fonts = {}

    # Ensure corpus subdirectories exist
    for bu in ("hr", "it_security", "product", "exec_comms"):
        (CORPUS_DIR / bu).mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    generated = 0

    for doc in DOCUMENTS:
        bu = doc["bu"]
        slug = doc["slug"]
        doc_type = doc["type"]
        title = doc["title"]
        topic = doc["topic"]

        for lang in doc["langs"]:
            if doc_type == "pdf":
                targets = [(_pdf_path(bu, slug, lang), "pdf")]
            elif doc_type == "slides":
                n = doc.get("n_slides", 5)
                targets = [(_slide_path(bu, slug, lang, i + 1), "slide") for i in range(n)]
            elif doc_type == "audio":
                targets = [(_audio_path(bu, slug, lang), "audio")]
            else:
                continue

            total += len(targets)

            # Check if all targets already exist
            if all(t[0].exists() for t in targets):
                skipped += len(targets)
                print(f"  SKIP  {targets[0][0].name}" + (f" (+{len(targets)-1} more)" if len(targets) > 1 else ""))
                continue

            print(f"  GEN   [{doc_type.upper():6}] [{lang.upper()}] {title[:55]}")
            if dry_run:
                generated += len(targets)
                continue

            try:
                if doc_type == "pdf":
                    # All content generation uses DeepSeek — Gemini KEY_A is
                    # reserved for the runtime pipeline, not batch generation.
                    content = _generate_pdf_text(deepseek_client, lang, title, topic)
                    _write_pdf(targets[0][0], title, content, lang, fonts)
                    generated += 1

                elif doc_type == "slides":
                    # Slides use DeepSeek (avoids Gemini 20 req/day free-tier cap)
                    slides = _generate_slide_texts(deepseek_client, title, topic, doc.get("n_slides", 5))
                    for i, (path, _) in enumerate(targets):
                        _write_slide(path, title, slides[i], i + 1, len(targets))
                    generated += len(targets)

                elif doc_type == "audio":
                    # Audio scripts use DeepSeek for same reason
                    speech = _generate_audio_text(deepseek_client, title, topic)
                    _write_audio(targets[0][0], speech)
                    generated += 1

                # Polite rate limiting
                time.sleep(1.5)

            except Exception as e:
                print(f"  ERROR {targets[0][0].name}: {e}")

    print(f"\n{'DRY RUN — ' if dry_run else ''}Done: {generated} generated, {skipped} skipped, {total} total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Meridian corpus documents")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be generated without doing it")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
