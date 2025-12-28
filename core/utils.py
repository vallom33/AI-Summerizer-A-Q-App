from langdetect import detect

def detect_lang(text: str) -> str:
    """
    Returns: 'en' or 'fr' or 'ar'
    """
    text = (text or "").strip()
    if not text:
        return "en"
    try:
        lang = detect(text)
    except:
        return "en"

    if lang.startswith("fr"):
        return "fr"
    if lang.startswith("ar"):
        return "ar"
    return "en"

def clean_text(t: str) -> str:
    return (t or "").strip()
