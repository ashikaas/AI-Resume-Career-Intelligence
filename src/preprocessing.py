import re


def clean_text(text):
    if not text:
        return ""

    text = text.lower()

    text = text.replace("&", " and ")

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    text = re.sub(r"[^a-z0-9\s+#.-]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()