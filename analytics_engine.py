import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

DB_PATH = "oracle_memory/oracle.db"

# ======================================================
# EXPORT CSV
# ======================================================

def export_csv(db=DB_PATH):

    conn = sqlite3.connect(db)

    df = pd.read_sql_query("SELECT * FROM memories", conn)

    df.to_csv("oracle_report.csv", index=False)

    return df


# ======================================================
# EXPORT JSON
# ======================================================

def export_json(df):

    df.to_json("oracle_report.json", orient="records")


# ======================================================
# EXPORT TXT
# ======================================================

def export_txt(df):

    with open("oracle_report.txt","w",encoding="utf-8") as f:

        f.write(df.to_string())


# ======================================================
# ANALYSE DES SOUVENIRS
# ======================================================

def analyze_memories(db=DB_PATH):

    conn = sqlite3.connect(db)

    df = pd.read_sql_query("SELECT * FROM memories", conn)

    if df.empty:

        return {
            "total_memories":0,
            "sources":{},
            "top_words":[]
        }

    # compteur sources
    sources = df["source"].value_counts().to_dict()

    # analyse lexicale simple
    text = " ".join(df["text"].astype(str).tolist())

    words = text.lower().split()

    freq = pd.Series(words).value_counts().head(10).to_dict()

    return {
        "total_memories": len(df),
        "sources": sources,
        "top_words": freq
    }


# ======================================================
# GRAPH PROGRESSION IA
# ======================================================

def graph_progress(db=DB_PATH):

    conn = sqlite3.connect(db)

    df = pd.read_sql_query("SELECT timestamp FROM memories", conn)

    if df.empty:

        return None

    # conversion date
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna()

    # colonne compteur
    df["count"] = 1

    # grouper par jour
    progress = df.groupby(df["timestamp"].dt.date)["count"].sum()

    fig, ax = plt.subplots()

    ax.plot(progress.index, progress.values, marker="o")

    ax.set_title("ORACLE Learning Progress")

    ax.set_xlabel("Date")

    ax.set_ylabel("Memories Learned")

    plt.xticks(rotation=45)

    os.makedirs("reports", exist_ok=True)

    path = "reports/oracle_progress.png"

    fig.savefig(path, bbox_inches="tight")

    return path


# ======================================================
# GRAPH TYPES DE CONNAISSANCE
# ======================================================

def graph_sources(db=DB_PATH):

    conn = sqlite3.connect(db)

    df = pd.read_sql_query("SELECT source FROM memories", conn)

    if df.empty:

        return None

    counts = df["source"].value_counts()

    fig, ax = plt.subplots()

    counts.plot(kind="bar", ax=ax)

    ax.set_title("Types de connaissances apprises")

    ax.set_xlabel("Source")

    ax.set_ylabel("Quantité")

    os.makedirs("reports", exist_ok=True)

    path = "reports/oracle_sources.png"

    fig.savefig(path, bbox_inches="tight")

    return path