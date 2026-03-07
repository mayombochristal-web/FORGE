import os
import json
from datetime import datetime

BASE="oracle_memory"

MEM="oracle_memory/memories"

def init():

    os.makedirs(MEM,exist_ok=True)


def save(text,vector,source):

    init()

    t=datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    data={

    "text":text,
    "vector":vector,
    "source":source,
    "time":t

    }

    path=f"{MEM}/memory_{t}.json"

    with open(path,"w",encoding="utf8") as f:

        json.dump(data,f)

    return path