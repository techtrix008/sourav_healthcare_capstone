"""Streamlit entrypoint for the healthcare assistant app.

The UI implementation lives in ``src.assistant_ui`` so this file stays focused
on startup order: configure Streamlit first, then render the application.
"""

from src.assistant_ui import configure_page, render_app


# Streamlit requires page configuration to happen before any visible elements.
configure_page()

# Hand off to the UI module where the chat, sidebar, tabs, and workflows live.
render_app()
