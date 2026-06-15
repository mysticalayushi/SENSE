import pandas as pd

def probability_dataframe(result):

    return pd.DataFrame(
        {
            "Sentiment":
            result["Probabilities"].keys(),

            "Probability":
            result["Probabilities"].values()
        }
    )