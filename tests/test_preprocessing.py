"""
Unit tests for the text preprocessing module.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_html, clean_text, preprocess_resume


def test_clean_html_basic():
    """Test HTML tag removal."""
    html = "<p>Hello <b>World</b></p>"
    result = clean_html(html)
    assert "Hello" in result
    assert "World" in result
    assert "<p>" not in result
    assert "<b>" not in result
    print("✅ test_clean_html_basic passed")


def test_clean_html_empty():
    """Test HTML cleaning with empty input."""
    assert clean_html("") == ""
    assert clean_html(None) == ""
    print("✅ test_clean_html_empty passed")


def test_clean_text_urls():
    """Test URL removal."""
    text = "Visit https://example.com for more info."
    result = clean_text(text)
    assert "https://example.com" not in result
    assert "Visit" in result
    print("✅ test_clean_text_urls passed")


def test_clean_text_emails():
    """Test email removal."""
    text = "Contact john@example.com for details."
    result = clean_text(text)
    assert "john@example.com" not in result
    print("✅ test_clean_text_emails passed")


def test_clean_text_special_chars():
    """Test special character removal."""
    text = "Python!!! & Java*** are great@@@"
    result = clean_text(text)
    assert "!!!" not in result
    assert "***" not in result
    assert "@@@" not in result
    print("✅ test_clean_text_special_chars passed")


def test_preprocess_resume_full():
    """Test the full preprocessing pipeline."""
    html_resume = """
    <html><body>
    <h1>John Doe</h1>
    <p>Email: john@example.com</p>
    <p>Skills: Python, Machine Learning, Data Analysis</p>
    <p>Experience: 5 years in software development</p>
    </body></html>
    """
    result = preprocess_resume(html_resume)

    # Should contain key terms
    assert "python" in result.lower() or "skill" in result.lower()
    # Should not contain HTML
    assert "<html>" not in result
    assert "<body>" not in result
    # Should not contain email
    assert "john@example.com" not in result

    print("✅ test_preprocess_resume_full passed")


def test_preprocess_resume_empty():
    """Test preprocessing with empty input."""
    assert preprocess_resume("") == ""
    assert preprocess_resume(None) == ""
    print("✅ test_preprocess_resume_empty passed")


if __name__ == "__main__":
    test_clean_html_basic()
    test_clean_html_empty()
    test_clean_text_urls()
    test_clean_text_emails()
    test_clean_text_special_chars()
    test_preprocess_resume_full()
    test_preprocess_resume_empty()
    print("\n✅ All preprocessing tests passed!")
