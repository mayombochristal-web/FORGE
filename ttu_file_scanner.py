import pdfplumber
import docx
import csv


def scan_file(file):

    name=file.name

    text=""

    if name.endswith(".txt"):

        text=file.read().decode()

    elif name.endswith(".pdf"):

        with pdfplumber.open(file) as pdf:

            for p in pdf.pages:

                text+=p.extract_text()

    elif name.endswith(".docx"):

        doc=docx.Document(file)

        for p in doc.paragraphs:

            text+=p.text

    elif name.endswith(".csv"):

        reader=csv.reader(file.read().decode().splitlines())

        for r in reader:

            text+=" ".join(r)

    return text