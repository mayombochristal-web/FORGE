# ======================================================
# TTU FILE SCANNER
# Text Transfer Unit
# ======================================================

import pandas as pd
import json
import PyPDF2
import docx

def scan_file(file):

    name = file.name.lower()

    try:

        # TXT
        if name.endswith(".txt"):

            return file.read().decode("utf-8", errors="ignore")

        # CSV
        if name.endswith(".csv"):

            df = pd.read_csv(file)

            return df.to_string()

        # JSON
        if name.endswith(".json"):

            data = json.load(file)

            return json.dumps(data)

        # PDF
        if name.endswith(".pdf"):

            reader = PyPDF2.PdfReader(file)

            text = ""

            for page in reader.pages:

                t = page.extract_text()

                if t:
                    text += t

            return text

        # DOCX
        if name.endswith(".docx"):

            doc = docx.Document(file)

            text = ""

            for p in doc.paragraphs:

                text += p.text + "\n"

            return text

        return ""

    except Exception as e:

        print("Scan error:", e)

        return ""