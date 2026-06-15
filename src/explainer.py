"""
explainer.py  —  src/explainer.py
Word-level sentiment contribution scores using SHAP + RoBERTa tokenizer.

Drop this file into your src/ folder.
"""

import numpy as np
import shap
import torch
from transformers import pipeline

# ── Reuse the same model pipeline your sentiment.py already loads ──────────
# Import your existing pipeline or rebuild it here.
# If your sentiment.py exposes `nlp_pipeline`, import it:
#   from src.sentiment import nlp_pipeline
# Otherwise we create a fresh one (cached via @st.cache_resource in app.py).

def build_pipeline():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1,
    )


def _pipe_wrapper(pipe):
    """
    SHAP's Text explainer expects a function: List[str] → np.ndarray (N, C).
    This wrapper converts the HF pipeline output to that shape.
    """
    label_order = ["Negative", "Neutral", "Positive"]   # fixed column order

    def predict(texts):
        results = pipe(list(texts))          # list of [{'label':..,'score':..}, ...]
        rows = []
        for scores_list in results:
            row_dict = {d["label"].capitalize(): d["score"] for d in scores_list}
            rows.append([row_dict.get(l, 0.0) for l in label_order])
        return np.array(rows)

    return predict


def get_word_scores(text: str, pipe=None) -> list[dict]:
    """
    Returns a list of dicts:
        [{"word": "fantastic", "score": 0.82, "sentiment": "positive"}, ...]

    score  > 0  → pushes toward the predicted sentiment  (green)
    score  < 0  → pushes against the predicted sentiment (red)
    score == 0  → neutral / stop-word                    (grey)

    Parameters
    ----------
    text : str   — raw input sentence
    pipe       — optional pre-built HF pipeline (avoids reloading weights)
    """
    if pipe is None:
        pipe = build_pipeline()

    predict_fn = _pipe_wrapper(pipe)

    # ── SHAP Partition explainer (works well on transformers) ───────────────
    masker = shap.maskers.Text(tokenizer=r"\W+")   # split on non-word chars
    explainer = shap.Explainer(predict_fn, masker, output_names=["Negative", "Neutral", "Positive"])

    shap_values = explainer([text])   # shape: (1, n_tokens, 3)

    # ── Pick the column matching the top predicted label ────────────────────
    probs = predict_fn([text])[0]           # [neg, neu, pos]
    top_label_idx = int(np.argmax(probs))   # 0=Neg, 1=Neu, 2=Pos
    label_names = ["Negative", "Neutral", "Positive"]
    top_label = label_names[top_label_idx]

    tokens = shap_values.data[0]            # array of word strings
    values = shap_values.values[0]          # shape: (n_tokens, 3)
    contributions = values[:, top_label_idx]   # 1-D: one score per token

    # Normalise to [-1, 1] for easy colour-mapping
    max_abs = np.max(np.abs(contributions)) + 1e-9
    norm = contributions / max_abs

    result = []
    for word, raw, n in zip(tokens, contributions, norm):
        sentiment = (
            "positive" if n > 0.05
            else "negative" if n < -0.05
            else "neutral"
        )
        result.append({
            "word": word,
            "score": float(raw),
            "norm": float(n),
            "sentiment": sentiment,
            "label": top_label,
        })

    return result