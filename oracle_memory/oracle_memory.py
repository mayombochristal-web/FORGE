
import os, json, datetime
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)
MEM_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MEM_DIR, exist_ok=True)

FILES = {
    "fragments": os.path.join(MEM_DIR, "fragments.csv"),
    "relations": os.path.join(MEM_DIR, "relations.json"),
    "cortex":    os.path.join(MEM_DIR, "cortex.json"),
}

def load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_json(p, d):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def load_frag():
    if not os.path.exists(FILES["fragments"]):
        return pd.DataFrame(columns=["fragment", "count"])
    return pd.read_csv(FILES["fragments"])

def save_frag(df):
    df.to_csv(FILES["fragments"], index=False)

def init_memory():
    if not os.path.exists(FILES["fragments"]):
        save_frag(pd.DataFrame(columns=["fragment", "count"]))
    if not os.path.exists(FILES["relations"]):
        save_json(FILES["relations"], {})
    if not os.path.exists(FILES["cortex"]):
        save_json(FILES["cortex"], {
            "VS": 12, "age": 0, "new_today": 0,
            "last_day": str(datetime.date.today()), "timeline": [],
        })

def merge_fragments(local_df, incoming_df):
    return (
        pd.concat([local_df, incoming_df])
        .groupby("fragment", as_index=False)["count"].sum()
    )

def merge_relations(local_rel, incoming_rel):
    for a, links in incoming_rel.items():
        local_rel.setdefault(a, {})
        for b, w in links.items():
            local_rel[a][b] = local_rel[a].get(b, 0) + w
    return local_rel

def merge_cortex(local_ctx, incoming_ctx):
    local_ctx["age"] += incoming_ctx.get("age", 0)
    local_ctx["new_today"] += incoming_ctx.get("new_today", 0)
    local_ctx.setdefault("timeline", [])
    local_ctx["timeline"].extend(incoming_ctx.get("timeline", []))
    local_ctx["timeline"] = local_ctx["timeline"][-5000:]
    local_ctx["VS"] = 10 + float(np.log1p(local_ctx["age"] * 1000))
    return local_ctx