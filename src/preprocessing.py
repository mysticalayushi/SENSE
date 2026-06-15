import re

def clean_text(text):

    text = str(text)

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    return text.lower().strip()