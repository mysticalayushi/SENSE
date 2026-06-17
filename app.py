import csv
import io as _io
import random
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF

from src.sentiment import SENSE
from src.batch_analyzer import SENSE_BATCH
from word_highlight import show_word_highlight_section

# ---------------------------------------------------
# PDF Report Generator
# ---------------------------------------------------

def generate_single_report(text: str, result: dict) -> bytes:
    """
    Generates a plain PDF report for a single review analysis.
    Returns the PDF as bytes for st.download_button.
    Also saves a copy to outputs/reports/.
    """

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Header ──────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "SENSE", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "Sentiment Extraction Natural Language Scoring Engine", ln=True)
    pdf.cell(0, 6, f"Report generated: {datetime.now().strftime('%d %B %Y  %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.set_draw_color(215, 235, 233)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # ── Input Text ───────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Input Review", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.set_fill_color(245, 245, 245)
    pdf.multi_cell(0, 7, text.strip(), border=0, fill=True)
    pdf.ln(6)

    # ── Results Summary ───────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Analysis Results", ln=True)
    pdf.ln(2)

    sentiment     = result["Sentiment"]
    confidence    = result["Confidence (%)"]
    score         = round(result["Sentiment Score"], 4)
    normalized    = round((result["Sentiment Score"] + 1) / 2 * 100, 1)

    # Sentiment colour
    if sentiment == "Positive":
        pdf.set_text_color(0, 150, 60)
    elif sentiment == "Negative":
        pdf.set_text_color(180, 0, 0)
    else:
        pdf.set_text_color(180, 150, 0)

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"  {sentiment}", ln=True)
    pdf.set_text_color(60, 60, 60)

    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)

    rows = [
        ("Confidence",        f"{confidence:.2f}%"),
        ("Sentiment Score",   str(score)),
        ("Positive Strength", f"{normalized}%"),
    ]

    pdf.set_fill_color(240, 240, 240)
    for label, value in rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(60, 8, f"  {label}", border=0, fill=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, value, border=0, fill=False, ln=True)
    pdf.ln(6)

    # ── Probability Distribution ──────────────────────
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, "Probability Distribution", ln=True)
    pdf.ln(2)

    probs = result["Probabilities"]
    prob_rows = [
        ("Positive", probs["Positive"], (0, 150, 60)),
        ("Neutral",  probs["Neutral"],  (180, 150, 0)),
        ("Negative", probs["Negative"], (180, 0, 0)),
    ]

    for label, prob, colour in prob_rows:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(40, 7, f"  {label}", border=0)

        # Bar
        bar_x   = pdf.get_x()
        bar_y   = pdf.get_y() + 1
        bar_max = 110
        bar_w   = bar_max * (prob / 100)

        pdf.set_fill_color(230, 230, 230)
        pdf.rect(bar_x, bar_y, bar_max, 5, style="F")
        pdf.set_fill_color(*colour)
        if bar_w > 0:
            pdf.rect(bar_x, bar_y, bar_w, 5, style="F")

        pdf.set_x(bar_x + bar_max + 4)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 7, f"{prob:.2f}%", ln=True)
    pdf.ln(6)

    # ── Footer ────────────────────────────────────────
    pdf.set_draw_color(215, 235, 233)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(0, 6, "Built with RoBERTa, Streamlit and Hugging Face  -  SENSE v1.0", ln=True)

    return bytes(pdf.output())


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="SENSE",
    page_icon="assets/logo.png",
    layout="wide"
)

# ------------------
# Custom CSS Theme  
# ------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base & Background ─────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #060608 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main {
    background-color: #060608 !important;
}

[data-testid="stHeader"] {
    background-color: #060608 !important;
    border-bottom: none !important;
}

/* ── Sidebar ───────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #080809 !important;
    border-right: 1px solid #2B2B2B !important;
}

[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] h1 {
    color: #D7EBE9 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-bottom: none !important;
    padding-bottom: 0 !important;
}

[data-testid="stSidebar"] h3 {
    color: #B2C2C1 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    margin-top: 1.2rem !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li {
    color: #848D8D !important;
    font-size: 0.82rem !important;
    line-height: 1.75 !important;
}

/* ── Main headings ─────────────────────────────────── */
h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.4rem !important;
    color: #ffffff !important;
    letter-spacing: -0.02em !important;
    padding-bottom: 6px !important;
    border-bottom: none !important;
    display: block !important;
}

h2 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: #D7EBE9 !important;
    font-size: 1.1rem !important;
    margin-top: 1.4rem !important;
}

h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #B2C2C1 !important;
    font-size: 0.9rem !important;
}

p, li, label, .stMarkdown {
    color: #848D8D !important;
    font-size: 0.88rem !important;
}

/* ── Tabs ──────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #2B2B2B !important;
    gap: 4px !important;
    background: transparent !important;
}

[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: #55595B !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 22px !important;
    border-radius: 6px 6px 0 0 !important;
    transition: color 0.2s ease, background 0.2s ease !important;
}

[data-testid="stTabs"] button[role="tab"]:hover {
    color: #B2C2C1 !important;
    background: #2B2B2B !important;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #D7EBE9 !important;
    border-bottom: 2px solid #D7EBE9 !important;
    background: transparent !important;
}

/* ── Metric Cards ──────────────────────────────────── */
[data-testid="metric-container"] {
    background: #2B2B2B !important;
    border: 1px solid #55595B !important;
    border-left: 3px solid #D7EBE9 !important;
    border-radius: 10px !important;
    padding: 18px 22px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(215, 235, 233, 0.08) !important;
}

[data-testid="metric-container"] label {
    color: #55595B !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.55rem !important;
    font-weight: 500 !important;
    color: #D7EBE9 !important;
}

[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.76rem !important;
    color: #B2C2C1 !important;
}

/* ── Primary Buttons ───────────────────────────────── */
[data-testid="stButton"] button,
button[kind="primary"] {
    background: #D7EBE9 !important;
    color: #060608 !important;
    -webkit-text-fill-color: #060608 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 10px 26px !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s ease, transform 0.15s ease !important;
}

[data-testid="stButton"] button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* Download buttons — outlined ghost */
[data-testid="stDownloadButton"] button {
    background: transparent !important;
    color: #D7EBE9 !important;
    border: 1px solid #55595B !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 8px 20px !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
}

[data-testid="stDownloadButton"] button:hover {
    border-color: #D7EBE9 !important;
    background: rgba(215, 235, 233, 0.06) !important;
}

/* ── Text Area ─────────────────────────────────────── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background-color: #2B2B2B !important;
    border: 1px solid #55595B !important;
    border-radius: 8px !important;
    color: #D7EBE9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #D7EBE9 !important;
    box-shadow: 0 0 0 3px rgba(215, 235, 233, 0.1) !important;
}

/* ── Selectbox / Multiselect ───────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background-color: #2B2B2B !important;
    border: 1px solid #55595B !important;
    border-radius: 8px !important;
    color: #D7EBE9 !important;
}

/* ── Progress Bar ──────────────────────────────────── */
[data-testid="stProgressBar"] > div > div > div {
    background: linear-gradient(90deg, #B2C2C1, #D7EBE9) !important;
    border-radius: 4px !important;
}

[data-testid="stProgressBar"] > div > div {
    background-color: #2B2B2B !important;
    border-radius: 4px !important;
    border: 1px solid #55595B !important;
}

/* ── File Uploader ─────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: #2B2B2B !important;
    border: 1px dashed #55595B !important;
    border-radius: 10px !important;
    padding: 8px !important;
    transition: border-color 0.2s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #D7EBE9 !important;
}

/* ── Expander ──────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: #2B2B2B !important;
    border: 1px solid #55595B !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary {
    color: #848D8D !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── Dataframe ─────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #55595B !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Alert banners ─────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
}

/* ── Divider ───────────────────────────────────────── */
hr {
    border-color: #2B2B2B !important;
}

/* ── Caption / Footer ──────────────────────────────── */
[data-testid="stCaptionContainer"] p,
.stCaption {
    color: #55595B !important;
    font-size: 0.75rem !important;
}

/* ── Plotly chart transparency ─────────────────────── */
.js-plotly-plot .plotly,
.js-plotly-plot .plotly .svg-container {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.title("🧠 About SENSE")

st.sidebar.markdown("""
### Sentiment Extraction Natural Language Scoring Engine

SENSE is an NLP-powered sentiment analysis application built using:

- 🤖 RoBERTa Transformer Model
- 🐍 Python
- 📊 Streamlit
- 📈 Plotly Visualizations

### Features

✅ Real-time sentiment prediction

✅ Confidence scoring

✅ Sentiment strength meter

✅ Probability distribution charts

✅ Interactive dashboard

✅ Word-level sentiment highlighting (SHAP)

✅ Batch CSV analysis

### Supported Domains

- Product Reviews
- Food Reviews
- Movie Reviews
- Customer Feedback
- Social Media Comments
- General Text

### Model

RoBERTa-base Sentiment Model

### Developer

Ayushi Rai
""")

# ---------------------------------------------------
# Title
# ---------------------------------------------------

title_col1, title_col2 = st.columns([1, 5])

with title_col1:
    st.image("assets/logo.png", width=200)

with title_col2:
    st.title("SENSE")
    st.caption("Sentiment Extraction Natural Language Scoring Engine")
    st.markdown(
        "Analyze customer reviews, product feedback, social media comments and text sentiment "
        "using a RoBERTa-powered NLP engine."
    )

st.divider()

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------

tab_single, tab_batch = st.tabs(["🔍 Single Review", "📂 Batch Analysis"])


# ===================================================
# TAB 1 — Single Review
# ===================================================

with tab_single:

    example = st.selectbox(
        "Choose an example review",
        [
            "Select Example",
            "The food was delicious and delivery was very fast.",
            "Worst purchase ever. Completely disappointed.",
            "The product arrived on time and works as expected.",
            "The movie was fantastic and kept me engaged throughout.",
            "Customer support was rude and unhelpful.",
            "The laptop performance is decent for the price."
        ]
    )

    if example != "Select Example":
        text = st.text_area("Enter a review", value=example, height=150)
    else:
        text = st.text_area("Enter a review", height=150)

    if st.button("Analyze"):

        if not text.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            _single_messages = [
                "🧠 Asking RoBERTa what it thinks…",
                "📡 Scanning emotional frequencies…",
                "🔬 Dissecting your words under a microscope…",
                "🤖 Feeding text to the sentiment oracle…",
                "⚡ Charging up the neural pathways…",
                "🧪 Running sentiment through the lab…",
                "🌊 Riding the waves of your text…",
                "🎯 Locking on to sentiment signals…",
                "🔮 Consulting the RoBERTa crystal ball…",
            ]

            with st.spinner(random.choice(_single_messages)):
                result = SENSE(text)

            m_col1, m_col2, m_col3 = st.columns(3)

            with m_col1:
                st.metric("Sentiment", result["Sentiment"])

            with m_col2:
                st.metric("Confidence", f"{result['Confidence (%)']}%")

            with m_col3:
                st.metric("Score", round(result["Sentiment Score"], 3))

            score      = result["Sentiment Score"]
            normalized = (score + 1) / 2

            st.subheader("Sentiment Strength")
            st.write(f"**{normalized * 100:.1f}% Positive Strength**")
            st.progress(normalized)

            st.subheader("Prediction Confidence")
            st.write(f"**{result['Confidence (%)']:.2f}% Confidence**")
            st.progress(result["Confidence (%)"] / 100)

            prob_df = pd.DataFrame({
                "Sentiment":   ["Positive", "Neutral", "Negative"],
                "Probability": [
                    result["Probabilities"]["Positive"],
                    result["Probabilities"]["Neutral"],
                    result["Probabilities"]["Negative"]
                ]
            })

            st.subheader("Probability Distribution")

            fig = px.bar(
                prob_df,
                x="Sentiment",
                y="Probability",
                color="Sentiment",
                title="Sentiment Probabilities",
                text_auto=".2f",
                color_discrete_map={
                    "Positive": "#00C853",
                    "Neutral":  "#FFD600",
                    "Negative": "#D50000"
                }
            )
            fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#2B2B2B",
                font_color="#848D8D",
                title_font_color="#D7EBE9",
                yaxis_title="Probability (%)",
                xaxis_title="Sentiment",
                showlegend=False,
                title_x=0.5,
                yaxis=dict(gridcolor="#55595B"),
                xaxis=dict(gridcolor="#55595B"),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            show_word_highlight_section(text, result)

            with st.expander("View Detailed Results"):
                st.json(result)

            # ── PDF Report Download ──────────────────
            st.subheader("📄 Download Report")
            pdf_bytes = generate_single_report(text, result)
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"sense_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                help="Download a plain PDF report of this analysis.",
            )


# ===================================================
# TAB 2 — Batch Analysis
# ===================================================

with tab_batch:

    st.subheader("📂 Batch CSV Analysis")
    st.markdown(
        "Upload a CSV with a column of reviews. "
        "SENSE will score every row and give you aggregate stats + a downloadable results file."
    )

    _buf = _io.StringIO()
    _writer = csv.writer(_buf, quoting=csv.QUOTE_ALL)
    _writer.writerow(["review"])
    _writer.writerow(["The food was amazing!"])
    _writer.writerow(["Terrible experience, never coming back."])
    _writer.writerow(["It was okay, nothing special."])
    _writer.writerow(["Absolutely love this product, will buy again!"])
    _writer.writerow(["Shipping was slow but the item quality is great."])
    sample_csv = _buf.getvalue().encode("utf-8")

    st.download_button(
        label="⬇️ Download sample CSV",
        data=sample_csv,
        file_name="sample_reviews.csv",
        mime="text/csv",
        help="Download this to see the expected column format before uploading your own file.",
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="CSV must have at least one text column. You'll pick which column to analyze below.",
    )

    if uploaded_file is not None:

        try:
            df_input = pd.read_csv(uploaded_file, on_bad_lines="skip", quoting=0)
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
            df_input = None

        if df_input is not None:

            if df_input.empty:
                st.warning("The uploaded CSV is empty.")

            else:
                text_columns = df_input.select_dtypes(include="object").columns.tolist()

                if not text_columns:
                    st.error("No text columns found in the CSV.")

                else:
                    col_choice = st.selectbox(
                        "Select the column that contains reviews",
                        options=text_columns,
                        index=0,
                    )

                    st.write(f"**Preview** — first 5 rows of `{col_choice}`:")
                    st.dataframe(df_input[[col_choice]].head(), use_container_width=True)

                    row_count = len(df_input)
                    st.info(f"Ready to analyze **{row_count} reviews**.")

                    if st.button("🚀 Run Batch Analysis", type="primary"):

                        texts = df_input[col_choice].fillna("").astype(str).tolist()

                        _batch_messages = [
                            "🚀 Launching sentiment rockets on all rows…",
                            "📊 Crunching through your reviews one by one…",
                            "🤖 RoBERTa is reading fast, hold tight…",
                            "🧠 Mass neural processing in progress…",
                            "⚙️ The sentiment engine is running hot…",
                            "🔬 Putting every review under the microscope…",
                            "⚡ Electrifying all those rows with NLP magic…",
                            "🎯 Targeting sentiment in every single review…",
                        ]

                        with st.spinner(random.choice(_batch_messages)):
                            results_df = SENSE_BATCH(texts)

                        results_df = pd.concat(
                            [df_input.reset_index(drop=True), results_df[["Sentiment", "Confidence"]]],
                            axis=1,
                        )

                        # ── Store in session state so filter reruns don't wipe results
                        st.session_state["batch_results"] = results_df
                        st.success(f"✅ Done! Analyzed {row_count} reviews.")

                    # ── Results persist via session state ────────────────
                    if "batch_results" in st.session_state:

                        results_df = st.session_state["batch_results"]

                        st.divider()

                        st.subheader("📊 Aggregate Stats")

                        counts   = results_df["Sentiment"].value_counts()
                        total    = len(results_df)
                        pct_pos  = counts.get("Positive", 0) / total * 100
                        pct_neu  = counts.get("Neutral",  0) / total * 100
                        pct_neg  = counts.get("Negative", 0) / total * 100
                        avg_conf = results_df["Confidence"].mean()

                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("😊 Positive",       f"{pct_pos:.1f}%", f"{counts.get('Positive', 0)} reviews")
                        s2.metric("😐 Neutral",        f"{pct_neu:.1f}%", f"{counts.get('Neutral',  0)} reviews")
                        s3.metric("😞 Negative",       f"{pct_neg:.1f}%", f"{counts.get('Negative', 0)} reviews")
                        s4.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")

                        pie_df = pd.DataFrame({
                            "Sentiment": ["Positive", "Neutral", "Negative"],
                            "Count":     [
                                counts.get("Positive", 0),
                                counts.get("Neutral",  0),
                                counts.get("Negative", 0),
                            ],
                        })

                        fig_pie = px.pie(
                            pie_df,
                            names="Sentiment",
                            values="Count",
                            hole=0.55,
                            color="Sentiment",
                            color_discrete_map={
                                "Positive": "#00C853",
                                "Neutral":  "#FFD600",
                                "Negative": "#D50000",
                            },
                            title="Sentiment Distribution",
                        )
                        fig_pie.update_layout(
                            title_x=0.5,
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#848D8D",
                            title_font_color="#D7EBE9",
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                        st.divider()

                        st.subheader("🗂️ Results Table")

                        sentiment_filter = st.multiselect(
                            "Filter by sentiment",
                            options=["Positive", "Neutral", "Negative"],
                            default=["Positive", "Neutral", "Negative"],
                        )

                        filtered_df = results_df[results_df["Sentiment"].isin(sentiment_filter)]

                        st.dataframe(
                            filtered_df.style.applymap(
                                lambda v: (
                                    "color: #00C853" if v == "Positive"
                                    else "color: #D50000" if v == "Negative"
                                    else "color: #FFD600"
                                ),
                                subset=["Sentiment"],
                            ),
                            use_container_width=True,
                            height=400,
                        )

                        st.caption(f"Showing {len(filtered_df)} of {total} reviews")

                        csv_out = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇️ Download Full Results CSV",
                            data=csv_out,
                            file_name="sense_batch_results.csv",
                            mime="text/csv",
                        )

                        st.divider()
                        if st.button("🗑️ Clear Results"):
                            del st.session_state["batch_results"]
                            st.rerun()

st.divider()
st.caption("Built with RoBERTa, Streamlit and Hugging Face • SENSE v1.0")