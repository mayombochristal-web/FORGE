# ======================================================
# ORACLE ANALYTICS ENGINE
# ======================================================

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

DB_PATH = "oracle_memory/cosmos_memory.db"

# ======================================================
# EXPORT CSV
# ======================================================

def export_csv():

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("SELECT * FROM memory", conn)

    os.makedirs("reports", exist_ok=True)

    path = "reports/oracle_report.csv"

    df.to_csv(path, index=False)

    return path


# ======================================================
# EXPORT JSON
# ======================================================

def export_json():

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("SELECT * FROM memory", conn)

    os.makedirs("reports", exist_ok=True)

    path = "reports/oracle_report.json"

    df.to_json(path, orient="records")

    return path


# ======================================================
# EXPORT TXT
# ======================================================

def export_txt():

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("SELECT * FROM memory", conn)

    os.makedirs("reports", exist_ok=True)

    path = "reports/oracle_report.txt"

    with open(path,"w",encoding="utf-8") as f:

        f.write(df.to_string())

    return path


# ======================================================
# ANALYSE MEMOIRE
# ======================================================

def analyze_memories():

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("SELECT * FROM memory", conn)

    if df.empty:

        return {
            "total_memories":0,
            "sources":{},
            "top_words":[]
        }

    sources = df["source"].value_counts().to_dict()

    text = " ".join(df["text"].astype(str).tolist())

    words = text.lower().split()

    freq = pd.Series(words).value_counts().head(10).to_dict()

    return {
        "total_memories": len(df),
        "sources": sources,
        "top_words": freq
    }


# ======================================================
# GRAPH PROGRESSION
# ======================================================

def graph_progress():

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("SELECT date FROM memory", conn)

    if df.empty:

        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna()

    df["count"] = 1

    progress = df.groupby(df["date"].dt.date)["count"].sum()

    fig, ax = plt.subplots()

    ax.plot(progress.index, progress.values, marker="o")

    ax.set_title("ORACLE Learning Progress")

    ax.set_xlabel("Date")

    ax.set_ylabel("Memories")

    plt.xticks(rotation=45)

    os.makedirs("reports", exist_ok=True)

    path = "reports/oracle_progress.png"

    fig.savefig(path, bbox_inches="tight")

    return path


# ======================================================
# GRAPH SOURCES
# ======================================================

def graph_sources():

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("SELECT source FROM memory", conn)

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