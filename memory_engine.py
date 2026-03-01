# S+02 — MEMORY ENGINE

import json, os

DATA_DIR="data"
LEXICON=f"{DATA_DIR}/lexicon.json"

os.makedirs(DATA_DIR,exist_ok=True)

def load_lex():
    if os.path.exists(LEXICON):
        return json.load(open(LEXICON))
    return {}

def save_lex(L):
    json.dump(L,open(LEXICON,"w"))

def learn(text,intensity):

    words=text.lower().split()
    if len(words)<2:
        return

    L=load_lex()

    for a,b in zip(words,words[1:]):
        L.setdefault(a,{})

        L[a][b]=L[a].get(b,0)+intensity

    save_lex(L)