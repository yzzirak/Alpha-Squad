import os
import cv2
import hashlib
import numpy as np
import pytesseract
import pdfplumber
import pandas as pd
from PIL import Image
from rapidfuzz import fuzz
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class ExpenseValidator:
    def __init__(self):
        self.expense_records = []
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()
        self.train_non_business_classifier()

    def train_non_business_classifier(self):
        """Train a simple NLP classifier to detect non-business expenses."""
        business_samples = ["hotel", "travel", "flight", "taxi", "business dinner", "conference", "office supply"]
        non_business_samples = ["netflix", "amazon prime", "grocery", "electricity bill", "shopping"]

        X_train = business_samples + non_business_samples
        y_train = [1] * len(business_samples) + [0] * len(non_business_samples)

        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)

    def extract_text_from_image(self, image_path):
        """Extract text from an image file using OCR."""
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF, separating multiple bills."""
        extracted_texts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_texts.append(text)
        return extracted_texts

    def extract_text_from_csv(self, csv_path):
        """Extract text from a CSV file."""
        df = pd.read_csv(csv_path)
        return df.to_string()

    def get_image_hash(self, image_path):
        """Generate a perceptual hash for an image file."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (8, 8))
        avg = np.mean(image)
        return ''.join(['1' if pixel > avg else '0' for pixel in image.flatten()])

    def check_duplicate(self, extracted_text, file_hash):
        """Detect duplicate bills using fuzzy text matching and image hashing."""
        for record in self.expense_records:
            prev_text, prev_hash = record
            text_similarity = fuzz.ratio(extracted_text, prev_text)
            if text_similarity > 90 or prev_hash == file_hash:
                return True
        return False

    def validate_date(self, extracted_text, entered_date):
        """Validate if the extracted bill date matches the user-entered date."""
        extracted_date = self.extract_date(extracted_text)
        return extracted_date == entered_date

    def extract_date(self, text):
        """Extract date from text (dummy implementation, can be improved)."""
        import re
        match = re.search(r'\b(\d{2}/\d{2}/\d{4})\b', text)
        return match.group(1) if match else None

    def detect_non_business_expense(self, extracted_text):
        """Detect whether a bill contains non-business expenses."""
        text_vec = self.vectorizer.transform([extracted_text])
        prediction = self.model.predict(text_vec)
        return prediction[0] == 0

    def process_expense(self, file_path, entered_date):
        """Process an expense file and perform all four validations."""
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in ['.jpg', '.jpeg', '.png']:
            extracted_text = self.extract_text_from_image(file_path)
            file_hash = self.get_image_hash(file_path)
        elif file_extension == '.pdf':
            extracted_texts = self.extract_text_from_pdf(file_path)
            for text in extracted_texts:
                file_hash = hashlib.md5(text.encode()).hexdigest()
                self.handle_expense(text, file_hash, entered_date)
            return
        elif file_extension == '.csv':
            extracted_text = self.extract_text_from_csv(file_path)
            file_hash = hashlib.md5(extracted_text.encode()).hexdigest()
        else:
            print("Unsupported file format!")
            return

        self.handle_expense(extracted_text, file_hash, entered_date)

    def handle_expense(self, extracted_text, file_hash, entered_date):
        """Handles duplicate checking, date validation, and non-business detection."""
        if self.check_duplicate(extracted_text, file_hash):
            print("Duplicate bill detected! Rejecting expense.")
            return

        if not self.validate_date(extracted_text, entered_date):
            print("Date mismatch detected! Please verify the bill date.")
            return

        if self.detect_non_business_expense(extracted_text):
            print("Non-business expense detected! Flagging for review.")
            return

        print("Expense successfully recorded.")
        self.expense_records.append((extracted_text, file_hash))




if __name__ == "__main__":
    validator = ExpenseValidator()
    file_path = r"C:\Users\hp\Downloads\OIP.jpeg"
    entered_date = "10/03/2025"
    validator.process_expense(file_path, entered_date)
