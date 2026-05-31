from __future__ import annotations

import re
from html import unescape
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import requests

from .config import DISCLAIMER


def _clean_html(text: str) -> str:
    text = unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _search_medlineplus(query: str) -> dict[str, object] | None:
    url = f"https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term={quote_plus(query)}"
    try:
        response = requests.get(url, timeout=4)
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except Exception:
        return None

    document = root.find(".//document")
    if document is None:
        return None
    title = _clean_html(document.findtext("content[@name='title']") or "MedlinePlus Result")
    summary = document.findtext("content[@name='FullSummary']") or document.findtext("content[@name='snippet']") or ""
    url_text = document.findtext("content[@name='url']")
    summary = _clean_html(summary)
    if not summary:
        return None
    return {
        "condition": title,
        "summary": _clean_html(summary)[:1200],
        "sources": [url_text] if url_text else ["https://medlineplus.gov/"],
        "disclaimer": DISCLAIMER,
        "source_type": "live_medlineplus",
    }


def search_medical_info(query: str) -> dict[str, object]:
    live_result = _search_medlineplus(query)
    if live_result:
        return live_result

    return {
        "condition": "Trusted Medical Information Unavailable",
        "summary": (
            "I could not retrieve a trusted live source for this topic right now. "
            "Please consult MedlinePlus, WHO, CDC, or a licensed clinician for condition-specific guidance."
        ),
        "sources": ["https://medlineplus.gov/", "https://www.who.int/health-topics"],
        "disclaimer": DISCLAIMER,
        "source_type": "trusted_source_unavailable",
    }
