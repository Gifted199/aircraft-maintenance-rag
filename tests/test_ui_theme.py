"""The UI theme (ui/theme.py): WCAG AA contrast, no agency branding, the generated Streamlit config
is up to date, and the UI image contains the theme files."""

import re
import tomllib
from pathlib import Path

import pytest

import theme
from presenter import UI_VERSION

ROOT = Path(__file__).resolve().parents[1]
T = theme.TOKENS


def _luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("text, background", [
    ("base-darkest", "white"), ("base-darkest", "base-lightest"),
    ("base-darkest", "error-lighter"), ("base-darkest", "warning-lighter"),
    ("base-darkest", "info-lighter"), ("base-darkest", "success-lighter"),
    ("base-darkest", "primary-lighter"), ("base-dark", "white"),
    ("white", "primary-darker"), ("white", "primary"), ("white", "primary-dark"),
    ("primary", "white"), ("primary", "base-lightest"), ("primary-dark", "primary-lighter"),
])  # fmt: skip
def test_every_text_colour_pair_meets_wcag_aa(text, background):
    assert contrast(T[text], T[background]) >= 4.5


def test_focus_outline_and_alert_borders_are_visible_on_white():
    assert contrast(T["focus"], T["white"]) >= 3  # WCAG non-text contrast
    assert contrast(T["error"], T["white"]) >= 3


def test_streamlit_theme_uses_the_uswds_tokens_and_dark_alert_text():
    config = tomllib.loads(theme.config_toml())["theme"]
    assert config["primaryColor"] == "#005ea2" and config["textColor"] == "#1b1b1b"
    assert config["redColor"] == "#d54309" and config["yellowColor"] == "#ffbe2e"
    assert config["blueColor"] == "#00bde3" and config["greenColor"] == "#00a91c"
    assert config["font"].startswith("Public Sans:https://fonts.googleapis.com/")
    for colour in ("red", "yellow", "blue", "green"):
        assert config[f"{colour}TextColor"] == T["base-darkest"]


def test_generated_config_file_is_up_to_date():
    saved = (ROOT / "ui" / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert saved == theme.config_toml(), "run: uv run python ui/theme.py"


def test_alert_styles_cover_every_streamlit_alert_kind():
    css = theme.page_css()
    for kind, colour in (("Error", "error"), ("Warning", "warning"), ("Info", "info"),
                         ("Success", "success")):  # fmt: skip
        assert f'[data-testid="stAlertContent{kind}"]' in css
        assert f"border-left-color: {T[colour]}" in css
    assert "focus-visible" in css and T["focus"] in css


def test_header_and_footer_name_the_app_and_carry_no_agency_branding():
    header, footer = theme.header_html(), theme.footer_html()
    assert "Aircraft Maintenance Manual Assistant" in header
    assert "Synthetic training data" in footer and theme.REPO_URL in footer
    assert 'aria-hidden="true"' in header  # the plane icon is decorative, the name is text
    text = " ".join((header, footer, theme.page_css(), theme.config_toml())).lower()
    for forbidden in ("faa", "federal aviation", "official website", ".gov", "united states",
                      "u.s. government", "seal"):  # fmt: skip
        assert forbidden not in text
    assert re.search(r'rel="noopener noreferrer"', footer)


def test_ui_image_includes_the_theme_and_compose_uses_the_ui_version():
    allowed = (ROOT / "ui" / ".dockerignore").read_text(encoding="utf-8")
    dockerfile = (ROOT / "ui" / "Dockerfile").read_text(encoding="utf-8")
    for name in ("theme.py", ".streamlit/config.toml"):
        assert f"!{name}" in allowed and name in dockerfile
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert f"akinsds1126-aero-rag-ui:{UI_VERSION}" in compose
    assert UI_VERSION == "0.2.1"
