# S+04 — INGESTION

import PyPDF2
import speech_recognition as sr

def read_pdf(file):

    reader=PyPDF2.PdfReader(file)
    text=[]

    for p in reader.pages:
        t=p.extract_text()
        if t:
            text.append(t)

    return " ".join(text)

def read_audio(file):

    r=sr.Recognizer()

    with sr.AudioFile(file) as source:
        audio=r.record(source)

    try:
        return r.recognize_google(audio,language="fr-FR")
    except:
        return ""