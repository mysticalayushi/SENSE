import pandas as pd

from src.sentiment import SENSE

def SENSE_BATCH(texts):

    results = []

    for text in texts:

        prediction = SENSE(text)

        results.append({

            "Review": text,

            "Sentiment":
            prediction["Sentiment"],

            "Confidence":
            prediction["Confidence (%)"]

        })

    return pd.DataFrame(results)