"""Generate the RAG Pipeline Failure Mode Analysis PDF."""

from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

OUTPUT = "docs/rag-pipeline-failure-mode-analysis.pdf"

FAILURE_MODES = [
    {
        "id": "FM-01",
        "title": "Scanned PDFs (Image-Only Documents)",
        "trigger": (
            "A PDF where all content is rendered as a raster image (e.g., produced by a "
            "scanner, screen capture, or image-to-PDF converter) with no embedded text layer."
        ),
        "current": (
            "PyPDFLoader loads the file without raising an exception and returns pages with "
            "empty page_content. split_documents produces zero chunks. The document is "
            "silently skipped — nothing is indexed, no error or warning is surfaced to the "
            "operator, and user queries about the document's content receive 'I don't know.'"
        ),
        "recommended": (
            "After loading, check whether any pages returned empty or whitespace-only "
            "page_content. Log a named warning per file at ingestion time (e.g., "
            "'WARNING: scanned-policy.pdf — 0 text pages extracted, skipping'). "
            "For documents that must be indexed, add OCR support via "
            "UnstructuredPDFLoader or pytesseract. Include an ingestion summary "
            "that counts zero-content files so operators know what was skipped."
        ),
    },
    {
        "id": "FM-02",
        "title": "Very Long Documents (100+ Pages)",
        "trigger": (
            "A PDF exceeding roughly 50 pages. Tested at 120 pages, which produces "
            "approximately 600 chunks at the default chunk_size of 1000 characters."
        ),
        "current": (
            "Chunking completes without error and produces the expected number of chunks. "
            "Retrieval correctly returns exactly k=4 results and all chunks carry start_index "
            "metadata for provenance. However, retrieval quality degrades silently: only 4 of "
            "600 chunks are surfaced per query, so relevant context in distant sections of the "
            "document is statistically unlikely to be retrieved."
        ),
        "recommended": (
            "For long documents, adopt parent-child (hierarchical) chunking: index smaller "
            "child chunks for retrieval but return their larger parent for context. "
            "Alternatively, increase k dynamically based on estimated document length. "
            "Consider a summarization pass at ingestion time for documents exceeding a "
            "configurable page threshold. Track chunk count per source in ingestion logs "
            "so operators can identify unexpectedly large documents."
        ),
    },
    {
        "id": "FM-03",
        "title": "Inconsistent Formatting",
        "trigger": (
            "Documents exported from HTML, Microsoft Word, Google Docs, or other rich-text "
            "sources that embed raw HTML tags, Markdown syntax, pipe-table delimiters, "
            "HTML comments, or mixed Unicode/Latin-1 characters."
        ),
        "current": (
            "Chunking proceeds without error. All formatting artifacts survive unchanged "
            "into chunks: HTML tags such as <h1> and <strong>, Markdown ## headers, "
            "pipe-table delimiters (|), HTML comments, and non-ASCII characters (€, é, í) "
            "all appear verbatim in the context string sent to the model. The model "
            "receives raw markup that wastes context tokens and may confuse its output."
        ),
        "recommended": (
            "Add a post-load sanitization step before chunking. Use Python's html.parser "
            "or the markdownify library to strip or convert HTML tags to plain text. "
            "Normalize whitespace and remove HTML comments. For tables, convert to "
            "prose summaries or filter them — they rarely chunk cleanly at character "
            "boundaries and produce incoherent fragments. Apply sanitization as a "
            "pipeline step in ingestion/loader.py so all documents are normalized "
            "before reaching the splitter."
        ),
    },
    {
        "id": "FM-04",
        "title": "Near-Duplicate Documents",
        "trigger": (
            "Multiple versions of the same policy coexisting in the store — for example, "
            "a PTO policy updated annually where older versions were never removed. "
            "Documents are 90–99% textually identical with key facts changed."
        ),
        "current": (
            "Both versions are retrieved for the same query (both appear in k=4 results), "
            "but there is no signal in the context about which version is current. The "
            "model may cite either value or blend them into a confident but wrong answer. "
            "With many near-duplicate chunks, all k=4 slots may be occupied by variants "
            "of the same document, crowding out other relevant context."
        ),
        "recommended": (
            "Add effective_date and version metadata to documents at ingestion time, "
            "derived from filename conventions or document content. Pass these fields "
            "through to chunks so retrieval results carry version context. Switch "
            "retrieval to Maximum Marginal Relevance (MMR) via "
            "as_retriever(search_type='mmr') to enforce result diversity. At ingestion, "
            "when a new version of a document is loaded, mark or remove prior versions "
            "using a document ID scheme. Include source and effective_date in the "
            "system prompt so the model can prefer the most recent source."
        ),
    },
    {
        "id": "FM-05",
        "title": "Missing or Incorrect Metadata",
        "trigger": (
            "Documents ingested with no 'department' key in metadata, a None value, "
            "a key name typo (e.g., 'deparment'), or a wrong department value. "
            "This can happen from bulk imports, manual additions, or loader bugs."
        ),
        "current": (
            "ChromaDB's $and filter silently excludes malformed documents at query time — "
            "no exception is raised. Documents missing the department key or carrying a "
            "typo are never surfaced to any user, regardless of content or permission "
            "level. This is safe from a data-leakage perspective but completely invisible "
            "operationally: there is no log entry, no counter, and no way to discover "
            "how many documents were silently lost."
        ),
        "recommended": (
            "Add metadata validation in ingestion/loader.py before calling add_documents. "
            "Define a required schema (department, permission_level, source) and reject "
            "or quarantine any document that fails validation, logging the filename and "
            "the missing or invalid field. Maintain an ingestion error counter and "
            "surface it in startup logs. A simple check — assert 'department' in "
            "doc.metadata — before indexing would catch all current failure cases."
        ),
    },
    {
        "id": "FM-06",
        "title": "Confidently Wrong Answers",
        "trigger": (
            "Two or more documents with directly contradictory facts about the same "
            "topic are present in the store simultaneously. The classic case: a policy "
            "updated in a new year where the old version was not removed."
        ),
        "current": (
            "Both contradictory documents are retrieved and concatenated into the same "
            "context string. The model receives both conflicting values with no signal "
            "about which is authoritative or more recent. The system prompt instructs "
            "the model to say 'I don't know' if context is insufficient — but the model "
            "has information (just conflicting information) and typically produces a "
            "confident, specific answer citing one of the two values."
        ),
        "recommended": (
            "The primary fix is source authority metadata: add effective_date and version "
            "to all documents and rank or filter retrieved chunks by recency before "
            "building the context string. A secondary fix is explicit citation: include "
            "the source filename and effective_date in the context sent to the model and "
            "instruct it to cite its source, giving users the ability to evaluate recency. "
            "For high-stakes queries, add a contradiction-detection step: if retrieved "
            "chunks contain conflicting numeric values for the same entity, surface the "
            "conflict explicitly in the response rather than resolving it arbitrarily."
        ),
    },
]


def build_pdf(output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Normal"],
        fontSize=22,
        leading=28,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#555555"),
        spaceAfter=4,
        fontName="Helvetica",
    )
    section_id_style = ParagraphStyle(
        "SectionId",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#888888"),
        fontName="Helvetica-Bold",
        spaceAfter=2,
    )
    section_title_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Normal"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1a1a2e"),
        fontName="Helvetica-Bold",
        spaceAfter=10,
    )
    label_style = ParagraphStyle(
        "Label",
        parent=styles["Normal"],
        fontSize=8,
        fontName="Helvetica-Bold",
        textColor=colors.white,
        leading=10,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#333333"),
        fontName="Helvetica",
        spaceAfter=0,
    )
    intro_style = ParagraphStyle(
        "Intro",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#444444"),
        fontName="Helvetica",
        spaceAfter=0,
    )

    LABEL_COLORS = {
        "TRIGGER":           colors.HexColor("#5c4a8a"),
        "CURRENT BEHAVIOR":  colors.HexColor("#b85c38"),
        "RECOMMENDED":       colors.HexColor("#2e7d57"),
    }

    story = []

    # ── Cover ──────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("RAG Pipeline", title_style))
    story.append(Paragraph("Failure Mode Analysis", title_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Date: 2026-04-08 · Version: 1.0", subtitle_style))
    story.append(Paragraph(
        "System: HR document RAG application · Models tested: PyPDFLoader + "
        "RecursiveCharacterTextSplitter + ChromaDB + Claude Sonnet",
        subtitle_style,
    ))
    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph(
        "This document characterizes six failure modes identified through adversarial document "
        "testing of the RAG ingestion and retrieval pipeline. For each mode it describes "
        "(1) the condition that triggers it, (2) the system&apos;s current behavior, and "
        "(3) the recommended remediation. All behaviors were verified by automated pytest "
        "tests in tests/adversarial/ &#8212; no live model calls required.",
        intro_style,
    ))
    story.append(Spacer(1, 0.35 * inch))

    # Summary table
    summary_data = [["ID", "Failure Mode", "Severity"]]
    severity = ["High", "Medium", "Low", "Medium", "High", "High"]
    sev_colors = {
        "High": colors.HexColor("#fdecea"),
        "Medium": colors.HexColor("#fff8e1"),
        "Low": colors.HexColor("#e8f5e9"),
    }
    for i, fm in enumerate(FAILURE_MODES):
        summary_data.append([fm["id"], fm["title"], severity[i]])

    summary_table = Table(
        summary_data,
        colWidths=[0.65 * inch, 4.4 * inch, 0.75 * inch],
    )
    summary_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])
    for i, sev in enumerate(severity, start=1):
        summary_style.add(
            "BACKGROUND", (2, i), (2, i), sev_colors[sev]
        )
        summary_style.add(
            "TEXTCOLOR", (2, i), (2, i), colors.HexColor("#333333")
        )
        summary_style.add("FONTNAME", (2, i), (2, i), "Helvetica-Bold")
        summary_style.add("FONTSIZE", (2, i), (2, i), 8)
        summary_style.add("ALIGN", (2, i), (2, i), "CENTER")

    summary_table.setStyle(summary_style)
    story.append(summary_table)
    story.append(PageBreak())

    # ── One page per failure mode ───────────────────────────────────────────────
    for fm in FAILURE_MODES:
        story.append(Paragraph(fm["id"], section_id_style))
        story.append(Paragraph(fm["title"], section_title_style))
        story.append(HRFlowable(
            width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=14
        ))

        for label, text, bg_color in [
            ("TRIGGER",          xml_escape(fm["trigger"]),      LABEL_COLORS["TRIGGER"]),
            ("CURRENT BEHAVIOR", xml_escape(fm["current"]),      LABEL_COLORS["CURRENT BEHAVIOR"]),
            ("RECOMMENDED",      xml_escape(fm["recommended"]),  LABEL_COLORS["RECOMMENDED"]),
        ]:
            row = Table(
                [[Paragraph(label, label_style), Paragraph(text, body_style)]],
                colWidths=[1.1 * inch, 5.55 * inch],
            )
            row.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), bg_color),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (0, 0), 8),
                ("LEFTPADDING", (1, 0), (1, 0), 12),
                ("RIGHTPADDING", (1, 0), (1, 0), 8),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
            ]))
            story.append(row)
            story.append(Spacer(1, 0.08 * inch))

        story.append(PageBreak())

    doc.build(story)
    print(f"Written: {output_path}")


if __name__ == "__main__":
    build_pdf(OUTPUT)
