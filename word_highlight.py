import streamlit as st
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shap
import pandas as pd

# ---------------------------------------------------
# Model loader (cached so it only loads once)
# ---------------------------------------------------

@st.cache_resource
def load_model_and_tokenizer():
    """Load the same RoBERTa model used in SENSE."""
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------
# SHAP attribution
# ---------------------------------------------------

def get_shap_values(text: str):
    """
    Compute SHAP token-level attributions for the predicted class.

    Returns
    -------
    clean_tokens : list[str]   — readable token strings
    token_scores : np.ndarray  — SHAP values aligned with tokens
    pred_label   : str         — 'Positive' | 'Neutral' | 'Negative'
    """
    tokenizer, model = load_model_and_tokenizer()
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    def predict_proba(texts):
        inputs = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.softmax(logits, dim=-1).numpy()   # (batch, 3)

    masker    = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(
        predict_proba, masker,
        output_names=["Negative", "Neutral", "Positive"]
    )
    shap_values = explainer([text], fixed_context=1, batch_size=8)

    probs     = predict_proba([text])[0]
    pred_idx  = int(np.argmax(probs))
    pred_label = label_map[pred_idx]

    # shap_values.values: (1, n_tokens, 3)
    token_scores = shap_values.values[0, :, pred_idx]
    raw_tokens   = shap_values.data[0]

    # Remove RoBERTa Ġ / Ċ byte-pair artifacts
    clean_tokens = [t.replace("Ġ", " ").replace("Ċ", "\n").strip() for t in raw_tokens]

    return clean_tokens, token_scores, pred_label


# ---------------------------------------------------
# HTML renderer
# ---------------------------------------------------

def _score_to_rgba(score: float, max_abs: float) -> str:
    """
    Positive SHAP → green  (word supports predicted sentiment)
    Negative SHAP → red    (word opposes predicted sentiment)
    """
    if max_abs == 0:
        return "rgba(0,0,0,0)"

    norm = score / max_abs          # -1 … +1

    if norm > 0:
        alpha = min(norm * 1.2, 0.85)
        return f"rgba(0, 200, 83, {alpha:.2f})"
    else:
        alpha = min(abs(norm) * 1.2, 0.85)
        return f"rgba(213, 0, 0, {alpha:.2f})"


def render_highlighted_html(tokens: list, scores: np.ndarray) -> str:
    """Return a self-contained HTML block with per-token backgrounds."""
    max_abs = float(np.max(np.abs(scores))) if len(scores) else 1.0

    spans = []
    for token, score in zip(tokens, scores):
        if not token:
            continue
        bg      = _score_to_rgba(score, max_abs)
        tooltip = f"SHAP: {score:+.4f}"
        spans.append(
            f'<span title="{tooltip}" style="'
            f'background-color:{bg};'
            f'border-radius:3px;'
            f'padding:2px 5px;'
            f'margin:2px 1px;'
            f'display:inline-block;'
            f'font-size:1rem;'
            f'line-height:2;'
            f'cursor:default;'
            f'">{token}</span>'
        )

    legend = """
    <div style="margin-top:14px;font-size:0.78rem;color:#999;display:flex;gap:18px;align-items:center;flex-wrap:wrap;">
        <span>
            <span style="background:rgba(0,200,83,0.75);border-radius:3px;padding:1px 10px;">&nbsp;</span>
            &nbsp;Supports predicted sentiment
        </span>
        <span>
            <span style="background:rgba(213,0,0,0.75);border-radius:3px;padding:1px 10px;">&nbsp;</span>
            &nbsp;Opposes predicted sentiment
        </span>
        <span style="color:#555;font-style:italic;">Hover a word for its exact SHAP value</span>
    </div>
    """

    return (
        '<div style="'
        'background:#111827;'
        'border:1px solid #2d2d2d;'
        'border-radius:10px;'
        'padding:18px 22px;'
        'line-height:2.2;'
        'word-wrap:break-word;'
        '">'
        + "".join(spans)
        + legend
        + "</div>"
    )


# ---------------------------------------------------
# Public Streamlit section
# ---------------------------------------------------

def show_word_highlight_section(text: str, result: dict):
    """
    Drop-in section for app.py.

    Usage
    -----
    Inside your  `if st.button("Analyze"):` block, after computing `result`:

        from word_highlight import show_word_highlight_section
        show_word_highlight_section(text, result)
    """

    st.subheader("🔍 Word-Level Sentiment Contribution")
    st.caption(
        "Words in **green** push the model toward the predicted sentiment. "
        "Words in **red** push against it. "
        "Colour intensity reflects how strongly each word contributes."
    )

    with st.spinner("Computing SHAP attributions… first run may take ~10 s"):
        try:
            tokens, scores, pred_label = get_shap_values(text)
        except Exception as exc:
            st.warning(f"SHAP computation failed: {exc}")
            return

    # Highlighted text block
    st.markdown(render_highlighted_html(tokens, scores), unsafe_allow_html=True)

    # Top-words breakdown table
    with st.expander("📊 Top contributing words (ranked by impact)"):
        df = pd.DataFrame({"Word": tokens, "SHAP Value": scores})
        df = df[df["Word"].str.strip() != ""]
        df["|SHAP|"] = df["SHAP Value"].abs()
        df = df.sort_values("|SHAP|", ascending=False).drop(columns="|SHAP|").head(12)
        df["Direction"] = df["SHAP Value"].apply(
            lambda v: "✅ Supports" if v > 0 else "❌ Opposes"
        )
        df["SHAP Value"] = df["SHAP Value"].map("{:+.5f}".format)
        st.dataframe(df, use_container_width=True, hide_index=True)