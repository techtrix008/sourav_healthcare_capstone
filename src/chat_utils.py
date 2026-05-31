from __future__ import annotations

import re
from datetime import date, timedelta


def extract_date(text: str) -> str | None:
    """Parse simple date references used in appointment chat follow-ups."""
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if match:
        return match.group(1)
    slash_match = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", text)
    if slash_match:
        day, month, year = slash_match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    text_l = text.lower()
    if "tomorrow" in text_l:
        return (date.today() + timedelta(days=1)).isoformat()
    if "today" in text_l:
        return date.today().isoformat()
    return None


def extract_problem(text: str) -> str | None:
    """Remove obvious date-only language so booking prompts keep the symptom/problem."""
    cleaned = re.sub(r"\b20\d{2}-\d{2}-\d{2}\b", "", text)
    cleaned = re.sub(r"\b\d{1,2}/\d{1,2}/20\d{2}\b", "", cleaned)
    cleaned = re.sub(r"\b(today|tomorrow)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" ,.;")
    date_only_words = {"date", "dates", "slot", "slots", "appointment", "available", "availability"}
    if not cleaned or cleaned.lower() in date_only_words:
        return None
    return cleaned


def is_end_session_request(text: str) -> bool:
    text_l = text.lower().strip()
    end_terms = ["end chat", "close chat", "close session", "end session", "start fresh", "reset chat", "new patient"]
    return any(term in text_l for term in end_terms)


def is_yes(text: str) -> bool:
    return text.lower().strip() in {"yes", "y", "yeah", "yep", "sure", "continue", "more", "yes please"}


def is_no(text: str) -> bool:
    return text.lower().strip() in {"no", "n", "nope", "nothing else", "end", "close", "done", "finish"}
