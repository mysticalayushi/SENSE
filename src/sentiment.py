from models.roberta_model import polarity_scores_roberta

label_map = {
    "roberta_neg": "Negative",
    "roberta_neu": "Neutral",
    "roberta_pos": "Positive"
}

def SENSE(text):

    scores = polarity_scores_roberta(text)

    predicted_label = max(
        scores,
        key=scores.get
    )

    sentiment = label_map[predicted_label]

    confidence = float(
        round(
            scores[predicted_label] * 100,
            2
        )
    )

    sentiment_score = float(
        round(
            scores["roberta_pos"]
            - scores["roberta_neg"],
            3
        )
    )

    return {

        "Text": text,

        "Sentiment": sentiment,

        "Confidence (%)": confidence,

        "Sentiment Score": sentiment_score,

        "Probabilities": {

            "Positive": float(
                round(
                    scores["roberta_pos"] * 100,
                    2
                )
            ),

            "Neutral": float(
                round(
                    scores["roberta_neu"] * 100,
                    2
                )
            ),

            "Negative": float(
                round(
                    scores["roberta_neg"] * 100,
                    2
                )
            )

        }
    }