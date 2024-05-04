from pdfminer.high_level import extract_text
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    return extract_text(pdf_path)


def get_vader_score(text):
    """Calculates the VADER compound sentiment score."""
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']


def normalize_flesch_kincaid(text, max_scale=20):
    """Calculates the Flesch-Kincaid grade level, normalized to a maximum scale."""
    fk_grade = textstat.flesch_kincaid_grade(text)
    normalized_fk = min(fk_grade, max_scale)  # Cap the FK grade to max_scale
    return normalized_fk / max_scale  # Normalize to a range of 0-1


def calculate_acceptability(text):
    """Calculates the acceptability score based on text analysis."""
    fk_score = normalize_flesch_kincaid(text)
    vader_score = get_vader_score(text)
    print(fk_score)
    print(vader_score)
    # Weighted sum of normalized FK score (65%) and VADER score (35%)
    acceptability = 0.65 * fk_score + 0.35 * vader_score
    return acceptability


# Example usage:
pdf_path = 'sample.pdf'  # Replace with your PDF file path
extracted_text = extract_text_from_pdf(pdf_path)

acceptability_score = calculate_acceptability(extracted_text)
print("Acceptability Score:", acceptability_score)
