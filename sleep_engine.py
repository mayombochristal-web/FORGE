# S+05 — SLEEP ENGINE

from core.memory_engine import load_lex, save_lex

def sleep_cycle(threshold=0.2,decay=0.85):

    L=load_lex()

    new_L={
        w:{t:p*decay for t,p in c.items()
           if p*decay>threshold}
        for w,c in L.items()
        if any(p*decay>threshold for p in c.values())
    }

    save_lex(new_L)

    return "Mémoire consolidée."