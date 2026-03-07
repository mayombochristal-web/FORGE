import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def export_csv(db):

    conn=sqlite3.connect(db)

    df=pd.read_sql_query("SELECT * FROM memories",conn)

    df.to_csv("oracle_report.csv")

    return df


def export_json(df):

    df.to_json("oracle_report.json")


def export_txt(df):

    df.to_string("oracle_report.txt")


def graph_progress(db):

    conn=sqlite3.connect(db)

    df=pd.read_sql_query("SELECT timestamp FROM memories",conn)

    df["timestamp"]=pd.to_datetime(df["timestamp"])

    df["count"]=1

    df=df.groupby(df["timestamp"].dt.date).sum()

    fig,ax=plt.subplots()

    ax.plot(df.index,df["count"])

    ax.set_title("Oracle Learning Progress")

    fig.savefig("oracle_progress.png")

    return "oracle_progress.png"