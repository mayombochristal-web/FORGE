import pandas as pd
import json
import PyPDF2
import docx

def scan_file(file):

    name=file.name.lower()

    try:

        if name.endswith(".txt"):

            return file.read().decode()

        if name.endswith(".csv"):

            df=pd.read_csv(file)

            return df.to_string()

        if name.endswith(".json"):

            data=json.load(file)

            return json.dumps(data)

        if name.endswith(".pdf"):

            reader=PyPDF2.PdfReader(file)

            text=""

            for p in reader.pages:
                text+=p.extract_text()

            return text

        if name.endswith(".docx"):

            doc=docx.Document(file)

            text=""

            for p in doc.paragraphs:
                text+=p.text+"\n"

            return text

        return ""

    except:

        return ""