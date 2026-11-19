from datetime import datetime, date, timedelta
import functools
import hashlib
import io
import json
import re
import sqlite3
from pathlib import Path
from collections import Counter
from xml.sax.saxutils import escape

import pandas as pd
import plotly.express as px
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FeedbackIQ",
    page_icon="📊",
    layout="wide",
)

st.title("📊 FeedbackIQ")
st.caption("PRODUCT FEEDBACK ANALYSIS | Python · Pandas · VADER · Streamlit")


# ============================================================
# CONSTANTS AND SENTIMENT SETUP
# ============================================================

analyzer = SentimentIntensityAnalyzer()

# Add a few software/product-specific words to VADER's vocabulary.
analyzer.lexicon.update({
    "crash": -2.5,
    "crashes": -2.5,
    "crashed": -2.5,
    "crashing": -2.5,
    "bug": -2.0,
    "bugs": -2.0,
    "glitch": -2.0,
    "glitches": -2.0,
    "laggy": -1.8,
    "freezes": -2.0,
    "freezing": -2.0,
    "overpriced": -2.0,
    # Common complaint words that VADER scores as neutral by default.
    "slow": -1.5,
    "lag": -1.8,
    "freeze": -2.0,
    "drain": -1.5,
    "drains": -1.5,
    # VADER treats "wish" as positive, which makes feature requests
    # such as "I wish it had X" look like praise.
    "wish": 0.0,
})

FEEDBACK_COLUMN_NAMES = [
    "feedback", "review", "reviews", "comment", "comments"
]

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]

SENTIMENT_COLORS = {
    "Positive": "#2EAD68",
    "Neutral": "#8A929B",
    "Negative": "#E05252",
}

SAFE_PHRASES = [
    "doesn't crash",
    "does not crash",
    "no crashes",
    "no issue",
    "no issues",
    "no problem",
    "no problems",
    "works fine",
    "working properly",
    "not too expensive",
    "not bad",
]

FEATURE_REQUEST_PATTERNS = [
    r"\bwould like\b",
    r"\bwould love (?:to see|a|an|it if)\b",
    r"\bi wish\b",
    r"\bplease (?:add|include|make|allow|support)\b",
    r"\b(?:can|could) you (?:please )?(?:add|include|make|allow|support)\b",
    r"\bshould (?:have|add|include|allow|support|offer)\b",
    r"\bneeds? (?:a|an|more|better|to have|support for)\b",
    r"\bit would be (?:great|nice|good|helpful|useful) if\b",
    r"\bfeature request\b",
    r"\bwant(?:ed)? (?:a|an|to see|more|the ability)\b",
]

# These are used to identify common product complaints in negative reviews.
COMPLAINT_KEYWORDS = [
    "crash", "crashes", "crashed", "crashing",
    "bug", "bugs", "glitch", "glitches",
    "slow", "lag", "laggy", "freeze", "freezes",
    "error", "errors", "failed", "failure",
    "issue", "issues", "problem", "problems",
    "expensive", "overpriced", "price",
    "payment", "refund", "login", "support",
    "loading", "battery", "ads",
]

WORD_VARIANTS = {
    "crashes": "crash", "crashed": "crash", "crashing": "crash",
    "bugs": "bug", "glitches": "glitch",
    "freezes": "freeze", "freezing": "freeze", "frozen": "freeze",
    "errors": "error", "issues": "issue", "problems": "problem",
    "failed": "fail", "fails": "fail", "failure": "fail",
    "laggy": "lag", "overpriced": "price",
}

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "you", "your",
    "was", "were", "are", "is", "but", "not", "have", "has",
    "had", "from", "they", "them", "their", "there", "then",
    "than", "when", "what", "which", "who", "how", "why",
    "very", "really", "just", "also", "too", "app", "product",
    "use", "using", "used", "can", "could", "would", "should",
    "please", "more", "most", "some", "all", "any", "one",
    "it", "its", "my", "our", "out", "get", "got", "been",
    "being", "about", "after", "before", "again", "once",
}

RECOMMENDATIONS = {
    "crash": "Investigate crash reports and test the latest release across supported devices.",
    "bug": "Reproduce reported bugs, prioritise them, and track fixes.",
    "glitch": "Identify the affected screens and test the user journey where the glitch occurs.",
    "slow": "Measure page and screen load times, then optimise the slowest areas.",
    "lag": "Profile performance and investigate lag on the affected devices.",
    "freeze": "Check for freezes and hangs in frequently used parts of the product.",
    "error": "Review error logs and make error messages clearer for users.",
    "fail": "Review the failing user flow and improve error handling.",
    "payment": "Test the payment flow and verify that failed transactions are handled correctly.",
    "refund": "Review the refund process and make its status clear to customers.",
    "login": "Test sign-in and password recovery across supported devices.",
    "expensive": "Review pricing feedback and communicate the value of each plan clearly.",
    "price": "Review pricing feedback and compare it with the value customers receive.",
    "support": "Review support response times and the quality of support answers.",
    "ads": "Review the frequency and placement of advertisements.",
    "battery": "Measure battery usage and investigate background activity.",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_feedback_column(dataframe):
    """Find a supported feedback column without case sensitivity."""
    column_lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in FEEDBACK_COLUMN_NAMES:
        if candidate in column_lookup:
            return column_lookup[candidate]

    return None


def clean_feedback(dataframe):
    """Remove missing, blank and duplicate feedback rows."""
    dataframe = dataframe.copy()
    uploaded_count = len(dataframe)

    dataframe = dataframe.dropna(subset=["feedback"]).copy()
    missing_count = uploaded_count - len(dataframe)

    dataframe["feedback"] = (
        dataframe["feedback"].astype(str).str.strip()
    )

    before = len(dataframe)
    dataframe = dataframe[dataframe["feedback"] != ""].copy()
    blank_count = before - len(dataframe)

    before = len(dataframe)
    dataframe = dataframe.loc[
        ~dataframe["feedback"].str.casefold().duplicated()
    ].copy()
    duplicate_count = before - len(dataframe)

    dataframe = dataframe.reset_index(drop=True)

    cleaning_report = {
        "uploaded": uploaded_count,
        "missing": missing_count,
        "blank": blank_count,
        "duplicates": duplicate_count,
        "clean": len(dataframe),
    }

    return dataframe, cleaning_report


def is_feature_request(text):
    """Identify common phrases that indicate a feature request."""
    text = str(text).lower()
    return any(
        re.search(pattern, text)
        for pattern in FEATURE_REQUEST_PATTERNS
    )


def classify_sentiment(score, text):
    """
    Classify sentiment using VADER, with a small set of
    product-specific rules to catch common software complaints.
    """
    text = str(text).lower().strip()

    # Clear positive phrases should not be misclassified because
    # they contain a word such as 'crash' or 'issue'.
    if any(phrase in text for phrase in SAFE_PHRASES):
        # VADER can score phrases like "no issues" as negative. Remove the safe
        # phrases and judge only what is left of the review.
        remainder = text
        for phrase in SAFE_PHRASES:
            remainder = remainder.replace(phrase, " ")
        remainder_score = analyzer.polarity_scores(remainder)["compound"]

        if remainder_score <= -0.05:
            return "Negative"  # something else in the review is still negative
        return "Positive"

    # VADER handles most negation and sentence context.
    # Apply a complaint override only when the review is also negative.
    has_complaint = any(
        re.search(r"\b" + re.escape(word) + r"\b", text)
        for word in COMPLAINT_KEYWORDS
    )

    if has_complaint and score < 0:
        return "Negative"

    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def analyse_feedback(dataframe):
    """Add sentiment score, sentiment label, feature request and estimated score."""
    dataframe = dataframe.copy()

    dataframe["sentiment_score"] = dataframe["feedback"].apply(
        lambda text: analyzer.polarity_scores(str(text))["compound"]
    )

    dataframe["sentiment"] = [
        classify_sentiment(score, text)
        for score, text in zip(
            dataframe["sentiment_score"],
            dataframe["feedback"],
        )
    ]

    dataframe["feature_request"] = dataframe["feedback"].apply(
        is_feature_request
    )

    # A polite request is generally neutral rather than praise.
    polite_request = (
        dataframe["feature_request"]
        & (dataframe["sentiment"] == "Positive")
        & (dataframe["sentiment_score"] < 0.5)
    )
    dataframe.loc[polite_request, "sentiment"] = "Neutral"

    # This is an estimated sentiment score, not a real customer rating.
    dataframe["estimated_sentiment_rating"] = (
        (dataframe["sentiment_score"] + 1) * 5
    ).round(1)

    return dataframe


def extract_words(text):
    """Return useful words from feedback for keyword analysis."""
    words = re.findall(r"[a-z']+", str(text).lower())
    cleaned_words = []

    for word in words:
        word = WORD_VARIANTS.get(word, word)
        if len(word) > 2 and word not in STOPWORDS:
            cleaned_words.append(word)

    return cleaned_words


def get_top_words(dataframe, sentiment, limit=10):
    """Count how many reviews mention each word."""
    selected_reviews = dataframe.loc[
        dataframe["sentiment"] == sentiment, "feedback"
    ]

    counts = Counter()
    for review in selected_reviews:
        # Count each word once per review.
        counts.update(set(extract_words(review)))

    return counts.most_common(limit)


def get_complaint_summary(dataframe, limit=None):
    """Find common complaint keywords in negative feedback."""
    negative_reviews = dataframe.loc[
        dataframe["sentiment"] == "Negative", "feedback"
    ]

    total_negative = len(negative_reviews)

    if total_negative == 0:
        return []

    # Count each canonical keyword at most once per review, so a review that
    # says "crash" and "crashed" is not counted twice (share can never pass 100%).
    merged = Counter()
    for review in negative_reviews:
        review_text = str(review).lower()
        found_in_review = set()
        for keyword in COMPLAINT_KEYWORDS:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, review_text):
                found_in_review.add(WORD_VARIANTS.get(keyword, keyword))
        merged.update(found_in_review)

    # Highest count first; ties are sorted alphabetically so the order is stable.
    results = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        results = results[:limit]

    complaint_details = []
    for keyword, mentions in results:
        share = mentions / total_negative * 100

        if share >= 40:
            priority = "High"
        elif share >= 20:
            priority = "Medium"
        else:
            priority = "Low"

        complaint_details.append({
            "issue": keyword,
            "mentions": mentions,
            "share": share,
            "priority": priority,
            "action": RECOMMENDATIONS.get(
                keyword,
                f"Review customer comments mentioning '{keyword}'.",
            ),
        })

    return complaint_details


def build_insights(dataframe):
    """Build a reusable summary for the Insights and Reports pages."""
    total = len(dataframe)

    if total == 0:
        return None

    counts = (
        dataframe["sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
    )

    positive_pct = counts["Positive"] / total * 100
    neutral_pct = counts["Neutral"] / total * 100
    negative_pct = counts["Negative"] / total * 100

    average_estimated_rating = dataframe[
        "estimated_sentiment_rating"
    ].mean()

    net_sentiment = positive_pct - negative_pct

    if net_sentiment >= 40:
        overall_mood = "Very positive"
    elif net_sentiment >= 10:
        overall_mood = "Generally positive"
    elif net_sentiment > -10:
        overall_mood = "Mixed"
    elif net_sentiment > -40:
        overall_mood = "Generally negative"
    else:
        overall_mood = "Very negative"

    headline = (
        f"Overall customer sentiment is {overall_mood.lower()}. "
        f"{positive_pct:.1f}% of feedback is positive, "
        f"{neutral_pct:.1f}% is neutral and "
        f"{negative_pct:.1f}% is negative."
    )

    return {
        "total": total,
        "counts": counts.to_dict(),
        "positive_pct": positive_pct,
        "neutral_pct": neutral_pct,
        "negative_pct": negative_pct,
        "average_estimated_rating": average_estimated_rating,
        "mood": overall_mood,
        "headline": headline,
        "complaints": get_complaint_summary(dataframe),
        "strengths": get_top_words(dataframe, "Positive", 8),
        "feature_requests": dataframe.loc[
            dataframe["feature_request"], "feedback"
        ].tolist(),
    }


def build_report(insights):
    """Create a downloadable Markdown report."""
    generated_date = datetime.now().strftime("%d %B %Y")
    counts = insights["counts"]

    lines = [
        "# FeedbackIQ - Customer Feedback Report",
        "",
        f"Generated on: {generated_date}",
        "",
        "## Overall Summary",
        "",
        insights["headline"],
        "",
        "| Measure | Result |",
        "|---|---:|",
        f"| Total feedback | {insights['total']} |",
        f"| Positive | {counts['Positive']} ({insights['positive_pct']:.1f}%) |",
        f"| Neutral | {counts['Neutral']} ({insights['neutral_pct']:.1f}%) |",
        f"| Negative | {counts['Negative']} ({insights['negative_pct']:.1f}%) |",
        f"| Average estimated sentiment score | {insights['average_estimated_rating']:.1f}/10 |",
        "",
        "> Note: The estimated sentiment score is calculated from text sentiment. "
        "It is not an actual customer-provided rating.",
        "",
        "## Common Complaints",
        "",
    ]

    if insights["complaints"]:
        lines.extend([
            "| Issue | Reviews | Share of negative reviews | Priority |",
            "|---|---:|---:|---|",
        ])

        for item in insights["complaints"]:
            lines.append(
                f"| {item['issue']} | {item['mentions']} | "
                f"{item['share']:.1f}% | {item['priority']} |"
            )

        lines.extend(["", "### Suggested Actions", ""])
        for item in insights["complaints"]:
            lines.append(f"- **{item['issue'].title()}:** {item['action']}")
    else:
        lines.append("No common complaint keywords were detected.")

    lines.extend(["", "## What Customers Like", ""])
    if insights["strengths"]:
        for word, mentions in insights["strengths"]:
            lines.append(f"- {word} ({mentions} reviews)")
    else:
        lines.append("No positive feedback keywords were detected.")

    lines.extend(["", "## Feature Requests", ""])
    if insights["feature_requests"]:
        for request in insights["feature_requests"]:
            lines.append(f"- {request}")
    else:
        lines.append("No feature requests were detected.")

    lines.extend([
        "",
        "## Method",
        "",
        "FeedbackIQ uses VADER sentiment analysis with a small set of "
        "product-specific rules. Keyword counts and feature requests are "
        "identified using Python text-processing rules. Results may not "
        "understand sarcasm or every sentence context, so important findings "
        "should be checked against the original feedback.",
    ])

    return "\n".join(lines)


def show_feedback_table(dataframe):
    """Display the main analysed feedback columns."""
    display_columns = [
        "feedback",
        "sentiment",
        "sentiment_score",
        "estimated_sentiment_rating",
        "feature_request",
    ]

    st.dataframe(
        dataframe[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "sentiment_score": st.column_config.NumberColumn(
                "Sentiment score", format="%.3f"
            ),
            "estimated_sentiment_rating": st.column_config.NumberColumn(
                "Estimated sentiment (0-10)", format="%.1f"
            ),
            "feature_request": st.column_config.CheckboxColumn(
                "Feature request"
            ),
        },
    )


# ============================================================
# SQLITE HISTORY AND PDF REPORT HELPERS
# ============================================================

DB_PATH = str(Path(__file__).resolve().parent / "feedbackiq_history.db")


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def initialise_database():
    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS upload_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uploaded_at TEXT NOT NULL,
                file_name TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                data_json TEXT NOT NULL
            )
        """)
        connection.commit()


def save_upload_to_database(dataframe, file_name):
    """Save one analysed upload as a dated historical batch."""
    columns = [
        "feedback", "sentiment", "sentiment_score",
        "estimated_sentiment_rating", "feature_request"
    ]
    stored_dataframe = dataframe[columns].copy()
    data_json = stored_dataframe.to_json(orient="records", force_ascii=False)
    uploaded_at = datetime.now().isoformat(timespec="seconds")

    with get_db_connection() as connection:
        cursor = connection.execute(
            """INSERT INTO upload_history
               (uploaded_at, file_name, row_count, data_json)
               VALUES (?, ?, ?, ?)""",
            (uploaded_at, file_name, len(stored_dataframe), data_json),
        )
        connection.commit()
        return cursor.lastrowid, uploaded_at


def get_upload_history():
    with get_db_connection() as connection:
        rows = connection.execute(
            """SELECT id, uploaded_at, file_name, row_count
               FROM upload_history ORDER BY uploaded_at DESC"""
        ).fetchall()
    return rows


def get_history_data(start_date=None, end_date=None):
    """Load saved feedback batches whose upload dates match the filter."""
    query = """SELECT id, uploaded_at, file_name, row_count, data_json
               FROM upload_history"""
    params = []
    conditions = []

    if start_date is not None:
        conditions.append("date(uploaded_at) >= date(?)")
        params.append(start_date.isoformat())
    if end_date is not None:
        conditions.append("date(uploaded_at) <= date(?)")
        params.append(end_date.isoformat())
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY uploaded_at DESC"

    with get_db_connection() as connection:
        rows = connection.execute(query, params).fetchall()

    frames = []
    for batch_id, uploaded_at, file_name, row_count, data_json in rows:
        try:
            records = json.loads(data_json)
        except (ValueError, TypeError):
            continue  # damaged saved data: skip it instead of crashing the app
        if records:
            frame = pd.DataFrame(records)
            frame["_batch_id"] = batch_id
            frame["_uploaded_at"] = uploaded_at
            frame["_source_file"] = file_name
            frames.append(frame)

    if not frames:
        return pd.DataFrame(), rows

    return pd.concat(frames, ignore_index=True), rows


def make_pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitleCustom", parent=styles["Title"],
        fontSize=19, leading=23, alignment=TA_CENTER,
        textColor=colors.HexColor("#243B53"), spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="SectionCustom", parent=styles["Heading2"],
        fontSize=12, leading=15, textColor=colors.HexColor("#243B53"),
        spaceBefore=10, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="SmallCell", parent=styles["BodyText"],
        fontSize=7.5, leading=9
    ))
    styles.add(ParagraphStyle(
        name="SmallCellWhite", parent=styles["BodyText"],
        fontSize=7.5, leading=9, textColor=colors.white
    ))
    return styles


def pdf_table(data, widths=None):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E2EC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F4F7FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build_summary_pdf(insights, period_label, heading="Period"):
    """Create a PDF containing the customer feedback summary."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
        topMargin=16*mm, bottomMargin=16*mm
    )
    styles = make_pdf_styles()
    story = [
        Paragraph("FeedbackIQ - Customer Summary", styles["ReportTitleCustom"]),
        Paragraph(f"{escape(heading)}: {escape(period_label)}", styles["BodyText"]),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles["BodyText"]),
        Spacer(1, 10),
        Paragraph("Overall Summary", styles["SectionCustom"]),
        Paragraph(escape(insights["headline"]), styles["BodyText"]),
        Spacer(1, 8),
    ]

    counts = insights["counts"]
    summary_rows = [
        ["Measure", "Value"],
        ["Total feedback", str(insights["total"])],
        ["Positive", f"{counts['Positive']} ({insights['positive_pct']:.1f}%)"],
        ["Neutral", f"{counts['Neutral']} ({insights['neutral_pct']:.1f}%)"],
        ["Negative", f"{counts['Negative']} ({insights['negative_pct']:.1f}%)"],
        ["Average estimated sentiment", f"{insights['average_estimated_rating']:.1f}/10"],
    ]
    story += [pdf_table(summary_rows, [75*mm, 85*mm]), Spacer(1, 8)]

    story.append(Paragraph("Common Complaints and Suggested Actions", styles["SectionCustom"]))
    if insights["complaints"]:
        rows = [["Issue", "Reviews", "Share", "Priority", "Suggested action"]]
        for item in insights["complaints"]:
            rows.append([
                Paragraph(escape(item["issue"]), styles["SmallCell"]),
                str(item["mentions"]), f"{item['share']:.1f}%",
                item["priority"],
                Paragraph(escape(item["action"]), styles["SmallCell"]),
            ])
        story.append(pdf_table(rows, [25*mm, 15*mm, 18*mm, 18*mm, 84*mm]))
    else:
        story.append(Paragraph("No common complaint keywords were detected.", styles["BodyText"]))

    story.append(Paragraph("What Customers Like", styles["SectionCustom"]))
    if insights["strengths"]:
        strength_text = ", ".join(f"{word} ({count} reviews)" for word, count in insights["strengths"])
        story.append(Paragraph(escape(strength_text), styles["BodyText"]))
    else:
        story.append(Paragraph("No recurring positive keywords were detected.", styles["BodyText"]))

    story.append(Paragraph("Feature Requests", styles["SectionCustom"]))
    if insights["feature_requests"]:
        for request in insights["feature_requests"]:
            story.append(Paragraph("• " + escape(str(request)), styles["BodyText"]))
    else:
        story.append(Paragraph("No feature requests were detected.", styles["BodyText"]))

    story += [Spacer(1, 10), Paragraph(
        "Method note: FeedbackIQ uses VADER sentiment analysis and Python rules. "
        "The estimated sentiment score is not a customer-provided rating. "
        "Keyword matching can miss context or sarcasm; review original comments "
        "before making decisions.", styles["Italic"]
    )]
    doc.build(story)
    return buffer.getvalue()


def build_analytics_pdf(dataframe, period_label):
    """Create a PDF containing metrics, complaint keywords and feedback rows."""
    insights = build_insights(dataframe)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4), rightMargin=14*mm, leftMargin=14*mm,
        topMargin=13*mm, bottomMargin=13*mm
    )
    styles = make_pdf_styles()
    story = [
        Paragraph("FeedbackIQ - Analytics Report", styles["ReportTitleCustom"]),
        Paragraph(f"Period: {escape(period_label)}", styles["BodyText"]),
        Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles["BodyText"]),
        Spacer(1, 8),
        Paragraph("Sentiment Metrics", styles["SectionCustom"]),
    ]

    counts = insights["counts"]
    metric_rows = [
        ["Total", "Positive", "Neutral", "Negative", "Positive rate", "Negative rate"],
        [str(insights["total"]), str(counts["Positive"]), str(counts["Neutral"]),
         str(counts["Negative"]), f"{insights['positive_pct']:.1f}%",
         f"{insights['negative_pct']:.1f}%"],
    ]
    story += [pdf_table(metric_rows, [35*mm, 35*mm, 35*mm, 35*mm, 40*mm, 40*mm])]

    story.append(Paragraph("Common Complaint Keywords", styles["SectionCustom"]))
    if insights["complaints"]:
        rows = [["Keyword", "Reviews", "Share of negative reviews", "Priority"]]
        for item in insights["complaints"]:
            rows.append([item["issue"], str(item["mentions"]), f"{item['share']:.1f}%", item["priority"]])
        story.append(pdf_table(rows, [55*mm, 35*mm, 65*mm, 35*mm]))
    else:
        story.append(Paragraph("No common complaint keywords were detected.", styles["BodyText"]))

    story.append(Paragraph("Analysed Feedback", styles["SectionCustom"]))
    feedback_rows = [["Feedback", "Sentiment", "Sentiment score", "Feature request"]]
    for _, row in dataframe.head(250).iterrows():
        feedback_rows.append([
            Paragraph(escape(str(row["feedback"])[:900]), styles["SmallCell"]),
            str(row["sentiment"]),
            f"{float(row['sentiment_score']):.3f}",
            "Yes" if bool(row["feature_request"]) else "No",
        ])
    story.append(pdf_table(feedback_rows, [165*mm, 32*mm, 35*mm, 35*mm]))
    if len(dataframe) > 250:
        story.append(Paragraph(
            f"Showing first 250 of {len(dataframe)} feedback records in this PDF. "
            "Download the analysed CSV for the complete dataset.", styles["BodyText"]
        ))

    doc.build(story)
    return buffer.getvalue()


# ------------------------------------------------------------
# Individual report for one saved upload
# ------------------------------------------------------------

def get_single_upload(batch_id):
    """
    Load ONE saved upload by its database id.
    Returns (info, dataframe). The dataframe is None when the saved
    data is missing or damaged.
    """
    with get_db_connection() as connection:
        row = connection.execute(
            """SELECT id, uploaded_at, file_name, row_count, data_json
               FROM upload_history WHERE id = ?""",
            (batch_id,),
        ).fetchone()

    if row is None:
        return None, None

    info = {"id": row[0], "uploaded_at": row[1], "file_name": row[2], "row_count": row[3]}

    try:
        frame = pd.DataFrame(json.loads(row[4]))
    except (ValueError, TypeError):
        return info, None

    needed = {"feedback", "sentiment", "sentiment_score",
              "estimated_sentiment_rating", "feature_request"}
    if frame.empty or not needed.issubset(frame.columns):
        return info, None

    return info, frame


def make_report_filename(file_name, batch_id, duplicate_name=False):
    """Example: Customer_Feedback.csv -> Customer_Feedback_Report.pdf"""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(file_name).stem).strip("_") or "Upload"
    if duplicate_name:
        # Two uploads share a name, so add the upload number to tell them apart.
        return f"{stem}_Report_upload{batch_id}.pdf"
    return f"{stem}_Report.pdf"


def build_upload_report_pdf(batch_id):
    """Generate the PDF for one saved upload, using only that upload's data."""
    info, frame = get_single_upload(batch_id)

    if frame is None:
        # Never crash the download; explain the problem inside the PDF.
        buffer = io.BytesIO()
        styles = make_pdf_styles()
        SimpleDocTemplate(buffer, pagesize=A4).build([
            Paragraph("FeedbackIQ - Report unavailable", styles["ReportTitleCustom"]),
            Paragraph(
                "The saved data for this upload is missing or could not be read, "
                "so a report could not be created.", styles["BodyText"]),
        ])
        return buffer.getvalue()

    uploaded_text = datetime.fromisoformat(info["uploaded_at"]).strftime("%d %b %Y, %I:%M %p")
    label = f"{info['file_name']} | Uploaded {uploaded_text} | {len(frame)} records"
    return build_summary_pdf(build_insights(frame), label, heading="Upload")


def show_upload_history_with_downloads(batches, key_prefix):
    """List saved uploads, each with its own Download Report button."""
    stem_counts = Counter(
        Path(row[2]).stem.casefold() for row in get_upload_history()
    )

    header = st.columns([3, 2.2, 1.3, 1.8])
    for column, title in zip(header, ["File name", "Upload date", "Records", "Report"]):
        column.markdown(f"**{title}**")

    for row in batches:
        batch_id, uploaded_at, file_name, row_count = row[:4]
        cells = st.columns([3, 2.2, 1.3, 1.8])
        cells[0].write(file_name)
        cells[1].write(datetime.fromisoformat(uploaded_at).strftime("%d %b %Y, %I:%M %p"))
        cells[2].write(str(row_count))
        cells[3].download_button(
            "Download Report",
            # A function is passed (not the PDF), so the PDF is built only when
            # this button is clicked, from this upload's saved data.
            data=functools.partial(build_upload_report_pdf, batch_id),
            file_name=make_report_filename(
                file_name, batch_id,
                duplicate_name=stem_counts[Path(file_name).stem.casefold()] > 1,
            ),
            mime="application/pdf",
            on_click="ignore",
            key=f"{key_prefix}_report_{batch_id}",
        )


initialise_database()

# ============================================================
# SESSION STATE
# ============================================================

if "feedback_data" not in st.session_state:
    st.session_state["feedback_data"] = None

if "cleaning_report" not in st.session_state:
    st.session_state["cleaning_report"] = None

if "last_saved_upload_hash" not in st.session_state:
    st.session_state["last_saved_upload_hash"] = None


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 FeedbackIQ")
st.sidebar.caption("PRODUCT FEEDBACK INTELLIGENCE")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Upload Feedback",
        "Analytics",
        "Feedback Insights",
        "Reports",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Version 3.0 | Internship-level project")
st.sidebar.subheader("History date filter")
period_choice = st.sidebar.selectbox(
    "Show saved feedback from",
    ["Recent 10 days", "Last 1 month", "All time", "Select date range"],
    key="history_period_choice",
)

today = date.today()
if period_choice == "Recent 10 days":
    filter_start, filter_end = today - timedelta(days=9), today
    period_label = f"{filter_start:%d %b %Y} to {filter_end:%d %b %Y}"
elif period_choice == "Last 1 month":
    filter_start, filter_end = today - timedelta(days=30), today
    period_label = f"{filter_start:%d %b %Y} to {filter_end:%d %b %Y}"
elif period_choice == "Select date range":
    chosen_range = st.sidebar.date_input(
        "Choose start and end dates",
        value=(today - timedelta(days=30), today),
        key="history_custom_date_range",
    )
    if isinstance(chosen_range, (tuple, list)) and len(chosen_range) == 2:
        filter_start, filter_end = chosen_range
    elif isinstance(chosen_range, (tuple, list)) and len(chosen_range) == 1:
        filter_start, filter_end = chosen_range[0], chosen_range[0]
    else:
        filter_start, filter_end = today, today
    period_label = f"{filter_start:%d %b %Y} to {filter_end:%d %b %Y}"
else:
    filter_start, filter_end = None, None
    period_label = "All saved uploads"

selected_data, selected_batches = get_history_data(filter_start, filter_end)
period_batches = list(selected_batches)  # every upload in the date range

# Optional: narrow the reports down to one saved upload.
ALL_UPLOADS = "All uploads in this period"
upload_choices = {ALL_UPLOADS: None}
for batch in selected_batches:
    upload_time = datetime.fromisoformat(batch[1]).strftime("%d %b %Y, %I:%M %p")
    upload_choices[f"{batch[2]} - {upload_time} (#{batch[0]})"] = batch[0]

chosen_upload = st.sidebar.selectbox(
    "Show report for",
    list(upload_choices.keys()),
    key="history_upload_choice",
)

if upload_choices[chosen_upload] is not None:
    chosen_id = upload_choices[chosen_upload]
    selected_data = selected_data[selected_data["_batch_id"] == chosen_id]
    selected_batches = [b for b in selected_batches if b[0] == chosen_id]
    period_label = f"{period_label} | file: {selected_batches[0][2]}"

if st.sidebar.button("Clear current screen data", width="stretch"):
    st.session_state["feedback_data"] = None
    st.session_state["cleaning_report"] = None
    st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    st.header("Dashboard")
    st.write("A quick overview of the customer feedback you have analysed.")

    dataframe = selected_data

    st.caption(f"Selected period: {period_label}. The filter uses the date each file was uploaded.")

    if dataframe is None or dataframe.empty:
        st.info(
            "No feedback data is available yet. Open **Upload Feedback** "
            "from the sidebar to upload a CSV or Excel file."
        )
    else:
        insights = build_insights(dataframe)
        counts = insights["counts"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Feedback", insights["total"])
        col2.metric("Positive", counts["Positive"])
        col3.metric("Neutral", counts["Neutral"])
        col4.metric("Negative", counts["Negative"])

        st.divider()
        st.subheader("Key indicators")

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(
            "Positive feedback rate",
            f"{insights['positive_pct']:.1f}%",
        )
        kpi2.metric(
            "Negative feedback rate",
            f"{insights['negative_pct']:.1f}%",
        )
        kpi3.metric(
            "Average estimated sentiment",
            f"{insights['average_estimated_rating']:.1f}/10",
        )

        st.caption(
            "The estimated sentiment score is derived from text analysis; "
            "it is not a customer-provided star rating."
        )

        st.divider()
        st.subheader("Sentiment overview")

        chart_data = pd.DataFrame({
            "Sentiment": SENTIMENT_ORDER,
            "Number of reviews": [
                counts[name] for name in SENTIMENT_ORDER
            ],
        })

        figure = px.bar(
            chart_data,
            x="Sentiment",
            y="Number of reviews",
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            text="Number of reviews",
        )
        figure.update_layout(showlegend=False)
        st.plotly_chart(figure, width="stretch")

        st.divider()
        st.subheader("Feedback highlights")

        positive_reviews = dataframe[
            dataframe["sentiment"] == "Positive"
        ].head(3)

        negative_reviews = dataframe[
            dataframe["sentiment"] == "Negative"
        ].head(3)

        positive_col, negative_col = st.columns(2)

        with positive_col:
            st.markdown("**Positive feedback examples**")
            if positive_reviews.empty:
                st.caption("No positive feedback found.")
            else:
                for _, row in positive_reviews.iterrows():
                    st.success(row["feedback"])

        with negative_col:
            st.markdown("**Negative feedback examples**")
            if negative_reviews.empty:
                st.caption("No negative feedback found.")
            else:
                for _, row in negative_reviews.iterrows():
                    st.error(row["feedback"])


        st.divider()
        st.subheader("Saved upload history")
        if selected_batches:
            show_upload_history_with_downloads(period_batches, "dashboard")

            st.subheader("Download reports for selected period")
            summary_pdf = build_summary_pdf(insights, period_label)
            analytics_pdf = build_analytics_pdf(dataframe, period_label)
            pdf_col1, pdf_col2 = st.columns(2)
            with pdf_col1:
                st.download_button(
                    "Download Summary PDF",
                    data=summary_pdf,
                    file_name=f"feedbackiq_summary_{datetime.now():%Y%m%d}.pdf",
                    mime="application/pdf",
                    width="stretch",
                    key="dashboard_summary_pdf",
                )
            with pdf_col2:
                st.download_button(
                    "Download Analytics PDF",
                    data=analytics_pdf,
                    file_name=f"feedbackiq_analytics_{datetime.now():%Y%m%d}.pdf",
                    mime="application/pdf",
                    width="stretch",
                    key="dashboard_analytics_pdf",
                )
        else:
            st.caption("No saved uploads exist in this date range.")


# ============================================================
# UPLOAD FEEDBACK
# ============================================================

elif page == "Upload Feedback":
    st.header("Upload Customer Feedback")
    st.write("Upload a CSV or Excel file containing customer reviews.")

    uploaded_file = st.file_uploader(
        "Choose a feedback file",
        type=["csv", "xlsx"],
        help="The file must contain a column named feedback, review or comment.",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                try:
                    raw_dataframe = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    # CSV files saved by Excel on Windows are often not UTF-8.
                    uploaded_file.seek(0)
                    raw_dataframe = pd.read_csv(
                        uploaded_file,
                        encoding="cp1252",
                        encoding_errors="replace",
                    )
            else:
                raw_dataframe = pd.read_excel(uploaded_file)

            if raw_dataframe.empty:
                st.error("The uploaded file has no rows.")
            else:
                feedback_column = find_feedback_column(raw_dataframe)

                if feedback_column is None:
                    st.error(
                        "Could not find a feedback column. Name the column "
                        "feedback, review, reviews, comment or comments."
                    )
                    st.write("Columns found:", list(raw_dataframe.columns))
                else:
                    working_dataframe = raw_dataframe.rename(
                        columns={feedback_column: "feedback"}
                    )

                    cleaned_dataframe, report = clean_feedback(
                        working_dataframe
                    )

                    st.subheader("Dataset overview")
                    a, b = st.columns(2)
                    a.metric("Rows uploaded", report["uploaded"])
                    b.metric("Columns", len(raw_dataframe.columns))

                    st.subheader("Data cleaning results")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Missing rows", report["missing"])
                    c2.metric("Blank rows", report["blank"])
                    c3.metric("Duplicates removed", report["duplicates"])
                    c4.metric("Clean records", report["clean"])

                    if cleaned_dataframe.empty:
                        st.warning(
                            "No usable feedback remains after cleaning. "
                            "Please upload a file with non-empty feedback."
                        )
                    else:
                        st.subheader("Cleaned data preview")
                        st.dataframe(
                            cleaned_dataframe.head(10),
                            width="stretch",
                            hide_index=True,
                        )

                        analysed_dataframe = analyse_feedback(
                            cleaned_dataframe
                        )

                        st.session_state["feedback_data"] = analysed_dataframe
                        st.session_state["cleaning_report"] = report

                        upload_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
                        if st.session_state["last_saved_upload_hash"] != upload_hash:
                            batch_id, saved_time = save_upload_to_database(
                                analysed_dataframe, uploaded_file.name
                            )
                            st.session_state["last_saved_upload_hash"] = upload_hash
                            st.success(
                                f"Feedback analysed and saved to history (upload #{batch_id}). "
                                f"Saved on {datetime.fromisoformat(saved_time).strftime('%d %b %Y, %I:%M %p')}."
                            )
                        else:
                            st.success("This upload has already been saved in history during this session.")

                        st.subheader("Sentiment results")
                        result1, result2, result3 = st.columns(3)
                        result1.metric(
                            "Positive",
                            int((analysed_dataframe["sentiment"] == "Positive").sum()),
                        )
                        result2.metric(
                            "Neutral",
                            int((analysed_dataframe["sentiment"] == "Neutral").sum()),
                        )
                        result3.metric(
                            "Negative",
                            int((analysed_dataframe["sentiment"] == "Negative").sum()),
                        )

                        st.subheader("Classified feedback")
                        show_feedback_table(analysed_dataframe)

        except pd.errors.EmptyDataError:
            st.error("The CSV file is empty.")
        except (ValueError, KeyError) as error:
            st.error(f"Could not process the uploaded file: {error}")
        except Exception as error:
            st.error(
                "Something went wrong while reading this file. "
                f"Details: {error}"
            )
    else:
        st.info("Choose a CSV or Excel file to begin.")


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":
    st.header("Feedback Analytics")
    st.write("Explore sentiment distribution, complaint keywords and reviews.")

    dataframe = selected_data

    if dataframe is None or dataframe.empty:
        st.info("No saved feedback in this date range. Upload feedback or change the date filter.")
    else:
        counts = (
            dataframe["sentiment"]
            .value_counts()
            .reindex(SENTIMENT_ORDER, fill_value=0)
        )

        left, right = st.columns(2)

        with left:
            st.subheader("Sentiment distribution")
            pie_figure = px.pie(
                names=counts.index,
                values=counts.values,
                color=counts.index,
                color_discrete_map=SENTIMENT_COLORS,
                hole=0.45,
            )
            st.plotly_chart(pie_figure, width="stretch")

        with right:
            st.subheader("Estimated sentiment score distribution")
            rating_counts = (
                dataframe["estimated_sentiment_rating"]
                .round()
                .value_counts()
                .sort_index()
            )
            rating_figure = px.bar(
                x=rating_counts.index.astype(str),
                y=rating_counts.values,
                labels={
                    "x": "Estimated score (0-10)",
                    "y": "Number of reviews",
                },
            )
            st.plotly_chart(rating_figure, width="stretch")

        st.divider()
        st.subheader("Common complaint keywords")

        complaints = get_complaint_summary(dataframe, limit=10)

        if complaints:
            complaints_dataframe = pd.DataFrame(complaints)
            complaint_figure = px.bar(
                complaints_dataframe,
                x="mentions",
                y="issue",
                orientation="h",
                text="mentions",
                labels={
                    "mentions": "Negative reviews mentioning keyword",
                    "issue": "Keyword",
                },
            )
            complaint_figure.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )
            st.plotly_chart(
                complaint_figure,
                width="stretch",
            )
        else:
            st.info("No common complaint keywords were detected.")

        st.divider()
        st.subheader("Customer feedback details")

        filter_col, search_col = st.columns([1, 1])

        with filter_col:
            selected_sentiments = st.multiselect(
                "Filter by sentiment",
                options=SENTIMENT_ORDER,
                default=SENTIMENT_ORDER,
            )

        with search_col:
            search_text = st.text_input(
                "Search within feedback",
                placeholder="Enter a word or phrase...",
            )

        filtered_dataframe = dataframe[
            dataframe["sentiment"].isin(selected_sentiments)
        ]

        if search_text.strip():
            filtered_dataframe = filtered_dataframe[
                filtered_dataframe["feedback"].str.contains(
                    search_text.strip(),
                    case=False,
                    regex=False,
                    na=False,
                )
            ]

        st.caption(
            f"Showing {len(filtered_dataframe)} of {len(dataframe)} reviews."
        )

        if filtered_dataframe.empty:
            st.info("No reviews match these filters.")
        else:
            show_feedback_table(filtered_dataframe)


# ============================================================
# FEEDBACK INSIGHTS
# ============================================================

elif page == "Feedback Insights":
    st.header("Feedback Insights")
    st.write(
        "Automatically generated observations using VADER sentiment "
        "analysis and Python text-processing rules."
    )

    dataframe = selected_data

    if dataframe is None or dataframe.empty:
        st.info("No saved feedback in this date range. Upload feedback or change the date filter.")
    else:
        insights = build_insights(dataframe)

        st.subheader("Overall picture")
        st.info(insights["headline"])

        st.divider()
        st.subheader("Common complaints")

        if insights["complaints"]:
            complaints_dataframe = pd.DataFrame([
                {
                    "Issue": item["issue"],
                    "Reviews": item["mentions"],
                    "Share of negative reviews": round(item["share"], 1),
                    "Priority": item["priority"],
                    "Suggested action": item["action"],
                }
                for item in insights["complaints"]
            ])

            st.dataframe(
                complaints_dataframe,
                width="stretch",
                hide_index=True,
            )
        else:
            st.success("No common complaint keywords were detected.")

        st.divider()
        st.subheader("What customers like")

        if insights["strengths"]:
            strengths_dataframe = pd.DataFrame(
                insights["strengths"],
                columns=["Keyword", "Reviews mentioning keyword"],
            )
            st.dataframe(
                strengths_dataframe,
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No recurring positive keywords were detected.")

        st.divider()
        st.subheader("Feature requests")

        if insights["feature_requests"]:
            for request in insights["feature_requests"]:
                st.write(f"• {request}")
        else:
            st.caption("No feature requests were detected.")

        st.caption(
            "These insights are generated using predefined rules and "
            "keyword matching. They are not generated by a large language model."
        )


# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":
    st.header("Reports")
    st.write("Download PDF reports for the date range selected in the sidebar.")
    dataframe = selected_data

    if dataframe is None or dataframe.empty:
        st.info("No saved feedback in this date range. Upload feedback or change the date filter.")
    else:
        insights = build_insights(dataframe)
        st.caption(f"Report period: {period_label} | {len(dataframe)} feedback records")

        summary_pdf = build_summary_pdf(insights, period_label)
        analytics_pdf = build_analytics_pdf(dataframe, period_label)
        export_columns = [
            "feedback", "sentiment", "sentiment_score",
            "estimated_sentiment_rating", "feature_request"
        ]
        csv_data = dataframe[export_columns].to_csv(index=False).encode("utf-8-sig")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Download Summary PDF",
                data=summary_pdf,
                file_name=f"feedbackiq_summary_{datetime.now():%Y%m%d}.pdf",
                mime="application/pdf",
                width="stretch",
                key="reports_summary_pdf",
            )
        with col2:
            st.download_button(
                "Download Analytics PDF",
                data=analytics_pdf,
                file_name=f"feedbackiq_analytics_{datetime.now():%Y%m%d}.pdf",
                mime="application/pdf",
                width="stretch",
                key="reports_analytics_pdf",
            )
        with col3:
            st.download_button(
                "Download Analysed CSV",
                data=csv_data,
                file_name=f"feedbackiq_data_{datetime.now():%Y%m%d}.csv",
                mime="text/csv",
                width="stretch",
                key="reports_csv",
            )

        st.divider()
        st.subheader("Summary preview")
        st.info(insights["headline"])
        st.subheader("Upload history in this period")
        if period_batches:
            show_upload_history_with_downloads(period_batches, "reports")


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()
st.sidebar.caption("Built with Python, Streamlit, Pandas, Plotly, VADER, SQLite and ReportLab.")