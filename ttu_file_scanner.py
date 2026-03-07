import pdfplumber
import docx
import csv

def scan_file(file):
    name = file.name
    text = ""
    if name.endswith(".txt"):
        text = file.read().decode("utf-8")
    elif name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for p in pdf.pages:
                page_text = p.extract_text()
                if page_text:
                    text += page_text
    elif name.endswith(".docx"):
        doc = docx.Document(file)
        for p in doc.paragraphs:
            text += p.text + "\n"
    elif name.endswith(".csv"):
        content = file.read().decode("utf-8").splitlines()
        reader = csv.reader(content)
        for r in reader:
            text += " ".join(r) + "\n"
    return text