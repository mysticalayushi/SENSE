# SENSE 🧠 — **S**entiment **E**xtraction **N**atural Language **S**coring **E**ngine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-RoBERTa-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-85%25-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-22C55E?style=for-the-badge)

<br/>

**SENSE — a dual-model NLP pipeline that extracts, scores, and explains sentiment in any piece of text, packaged as an interactive Streamlit dashboard.**

</div>

---

## 📌 Project Overview

Understanding *what people feel* about a product, service, or experience is one of the most valuable signals a business can have — but reading thousands of reviews manually doesn't scale.

SENSE answers the question:

> **"Given any piece of text, what sentiment does it express — and why?"**

It combines a fast, interpretable rule-based model (**VADER**) with a context-aware transformer (**RoBERTa**) into a single reusable scoring engine, then wraps it in a full-featured dashboard for both single-text and batch analysis.

> 💡 **Note on the dataset:** SENSE's NLP engine was developed and benchmarked on a 5,000-review sample drawn from the Amazon Fine Food Reviews dataset (~568,000 reviews total) as a real-world testbed. While the validation set is food-focused, the underlying RoBERTa-based model generalizes to any text domain — product reviews, app feedback, social media comments, support tickets, and more.

---

## 📚 Table of Contents

<ul>
  <li><a href="#-project-overview">📌 Project Overview</a></li>
  <li><a href="#-live-demo">🚀 Live Demo</a></li>
  <li><a href="#-project-structure">📂 Project Structure</a></li>
  <li><a href="#️-tech-stack">🛠️ Tech Stack</a></li>
  <li><a href="#-validation-dataset">📊 Validation Dataset</a></li>
  <li><a href="#-model-training--performance">🤖 Model Training & Performance</a></li>
  <li><a href="#-model-validation-visuals">📈 Model Validation Visuals</a></li>
  <li><a href="#-sense-engine">🔮 The SENSE Engine</a></li>
  <li><a href="#-application-features">🚀 Application Features</a></li>
  <li><a href="#-Business-Impact">🎯 Use Cases</a></li>
  <li><a href="#️-run-locally">▶️ Run Locally</a></li>
  <li><a href="#-future-scope">🔭 Future Scope</a></li>
  <li><a href="#-author">👩‍💻 Author</a></li>
</ul>

---

## 🚀 Live Demo

<div align="center">

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![Notebook](https://img.shields.io/badge/Notebook-View%20Analysis-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](notebooks/sense.ipynb)


> ⚠️ **App may be sleeping** — Streamlit free tier may hibernate. Click **Live App** and wait ~30s for it to spin up.

</div>

---

## 📂 Project Structure

```
SENSE/
│
├── assets/
│   ├── visuals/                      # Model validation visuals (from notebook)
│   └── logo.png                      # App logo
│
├── models/
│   └── roberta_model.py              # RoBERTa model loading utilities
│
├── notebooks/
│   └── sense.ipynb                   # Full EDA + model training/comparison notebook
│
├── src/
│   ├── sentiment.py                  # Core SENSE() scoring function
│   ├── batch_analyzer.py             # SENSE_BATCH() for multi-review scoring
│   ├── explainer.py                  # SHAP-based word-level explainability
│   ├── preprocessing.py              # Text cleaning utilities
│   └── visualization.py              # Chart helper functions
│
├── word_highlight.py                 # Renders SHAP word-highlight UI
├── app.py                            # Streamlit application
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

<div align="center">

| 🏷️ Category | 🔧 Tools |
|---|---|
| 🐍 **Language** | Python 3.11 |
| 🤖 **Models** | RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`) |
| 🧠 **NLP / ML** | Hugging Face Transformers, SHAP, SciPy |
| 📊 **Data** | Pandas, NumPy |
| 📈 **Visualization** | Plotly |
| 🚀 **Deployment** | Streamlit Cloud |
| 📄 **Reporting** | fpdf2 (PDF export) |

</div>

---

## 📊 Validation Dataset

The model was benchmarked using the **Amazon Fine Food Reviews dataset from Kaggle**.

- **Total Records:** ~568,000 reviews (full dataset)
- **Sample Used:** 5,000 reviews — large enough to observe meaningful patterns, small enough to iterate fast
- **Key Columns:** `Id`, `Score` (1–5 star rating, used as ground truth), `Text` (raw review), `Summary` (review title)

**Resulting class distribution (5,000-review sample):**

| Sentiment | Count | % |
|---|---|---|
| 😊 Positive | 3,846 | 77.6% |
| 😠 Negative | 759 | 15.3% |
| 😐 Neutral | 395 | 8.0% |

---

## 🤖 Model Training & Performance

Two models were trained and benchmarked against the star-rating ground truth:

<div align="center">

| Metric | VADER (Rule-Based) | RoBERTa (Transformer) |
|---|---|---|
| ⚡ Overall Accuracy | 0.80 | **0.85** |
| 😊 Positive F1 | 0.89 | **0.93** |
| 😠 Negative F1 | 0.48 | **0.74** |
| 😐 Neutral F1 | 0.08 | **0.22** |
| 📊 Macro Avg F1 | 0.49 | **0.63** |

</div>

**Why RoBERTa wins:** VADER is a fast, lexicon-based scorer with no training required — great as a baseline, but it struggles with negation, sarcasm, and ambiguous/neutral language. RoBERTa, fine-tuned on ~58M tweets for 3-class sentiment, understands full-sentence context — giving it a decisive edge on the hardest classes (**Negative** and **Neutral**), which is exactly where business value lies in catching dissatisfied customers.

> ⚠️ Reviews exceeding RoBERTa's 512-token limit (44 reviews, 0.88% of the sample) were caught and skipped via a `try/except RuntimeError` guard during inference.

**Pipeline:** Tokenize → Forward Pass → Softmax → Score Dictionary, with raw logits normalised via `scipy.special.softmax` into interpretable probabilities summing to 1.0.


---

## 📈 Model Validation Visuals

<table>
  <tr>
    <td align="center" width="50%">
      <img src="assets/visuals/sentiment_class_distribution.png" alt="Sentiment Class Distribution" width="100%"/>
      <br/>
      <em>Fig 1 — Ground-truth sentiment class distribution (3,846 Positive / 759 Negative / 395 Neutral)</em>
    </td>
    <td align="center" width="50%">
      <img src="assets/visuals/pairplot_vader_vs_roberta.png" alt="VADER vs RoBERTa Pairplot" width="100%"/>
      <br/>
      <em>Fig 2 — Pairwise comparison of VADER vs RoBERTa sentiment scores, coloured by star rating</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="assets/visuals/vader_confusion_matrix.png" alt="VADER Confusion Matrix" width="100%"/>
      <br/>
      <em>Fig 3 — VADER confusion matrix: strong on Positive, weak on Neutral</em>
    </td>
    <td align="center" width="50%">
      <img src="assets/visuals/roberta_confusion_matrix.png" alt="RoBERTa Confusion Matrix" width="100%"/>
      <br/>
      <em>Fig 4 — RoBERTa confusion matrix: more diagonal-dominant, especially on Negative</em>
    </td>
  </tr>
</table>

---

## 🚀 Application Features

The deployed Streamlit dashboard includes:

- 🔍 **Single Review Analysis** — paste any text and get instant sentiment, confidence %, and a sentiment score
- 📊 **Sentiment Strength & Confidence Meters** — visual progress bars for at-a-glance interpretation
- 📈 **Probability Distribution Chart** — Plotly bar chart breaking down Positive / Neutral / Negative probabilities
- 🔬 **Word-Level SHAP Highlighting** — colour-coded tokens showing exactly which words drove the model's prediction, with hoverable contribution scores
- 🎯 **Overall Sentiment Score** — a single normalised score ranging from -1 (fully negative) to +1 (fully positive), giving an at-a-glance polarity reading beyond just the label
- 📂 **Batch CSV Analysis** — upload any CSV, pick the text column, and score every row at once
- 📊 **Aggregate Batch Stats** — % Positive/Neutral/Negative, average confidence, and a sentiment distribution donut chart
- 🗂️ **Filterable Results Table** with colour-coded sentiment labels and CSV export
- 📄 **PDF Report Export** — download a clean, branded PDF summary of any single-review analysis
- 🎨 **Custom Dark Theme** — polished, cohesive dark UI built with custom CSS

---

## 💼 BUSINESS IMPACT

SENSE transforms unstructured customer feedback into actionable business intelligence — enabling organizations to understand customer sentiment at scale and make data-driven decisions.

### 🛒 Customer Experience Monitoring
Analyze thousands of reviews, survey responses, and support interactions to identify customer satisfaction trends and emerging issues before they escalate.

### 🔧 Product Feedback Analysis
Detect recurring complaints and highly praised features — helping product teams prioritize improvements and make informed roadmap decisions.

### 📣 Brand Reputation Tracking
Monitor sentiment across social media, online reviews, and public feedback channels to assess brand perception and respond proactively.

### 🎫 Customer Support Intelligence
Identify high-frustration tickets and negative feedback early, enabling support teams to triage and prioritize critical customer issues faster.

### 🔬 Market Research & Voice of Customer (VoC)
Extract sentiment patterns from large volumes of qualitative feedback to uncover customer preferences, expectations, and pain points at scale.

### 📊 Executive Reporting
Convert raw textual feedback into measurable sentiment metrics, visual dashboards, and exportable reports that support strategic decision-making.

---

## ▶️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/mysticalayushi/SENSE.git
cd SENSE

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

Or explore the full analysis notebook:
```bash
jupyter notebook notebooks/sense.ipynb
```

---

## 🔭 Future Scope

- [ ] 🌐 REST API endpoint for programmatic access to the SENSE scoring engine
- [ ] 📝 Review summarization — auto-generate a concise summary of key points from batch reviews
- [ ] 📈 Trend analysis — track sentiment shifts over time across large review datasets
- [ ] 🤖 LLM integration — use a large language model to generate executive summaries highlighting  positive themes, negative themes, and actionable business insights

---

## 📋 Project Information

<div align="center">

| 📌 Field | 📝 Detail |
|---|---|
| 👩‍💻 **Created by** | Ayushi Rai |
| 🧠 **Models Used** | RoBERTa (Hugging Face Transformers) |
| 🎯 **Task** | 3-Class Sentiment Classification (Positive / Neutral / Negative) |
| 📊 **Validation Dataset** | Amazon Fine Food Reviews — 5,000-review sample (568K total) |
| 📈 **Best Accuracy** | 85% (RoBERTa) |
| 📅 **Date** | June 2026 |

</div>

---

## 👩‍💻 Author

<div align="center">

**Ayushi Rai**
[![GitHub](https://img.shields.io/badge/GitHub-mysticalayushi-181717?style=flat&logo=github)](https://github.com/mysticalayushi)

</div>

---

<div align="center">
<sub>If you found this project helpful, consider giving it a ⭐ on GitHub!</sub>
</div>

---

<div align="center">
  <a href="#sense-">⬆️ Back to Top</a>
</div>