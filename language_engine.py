# S+03 — LANGUAGE ENGINE

import random
from core.memory_engine import load_lex

def oracle_reply(phi,seed=None):

    L=load_lex()

    if not L:
        return "Je commence à apprendre."

    if not seed or seed not in L:
        seed=random.choice(list(L.keys()))

    M,C,D=phi["phi_m"],phi["phi_c"],phi["phi_d"]

    words=[seed]

    length=int(5+M*25)

    for _ in range(length):

        current=words[-1]

        if current not in L:
            break

        options=L[current]

        if random.random()>C:
            nxt=max(options,key=options.get)
        else:
            nxt=random.choices(
                list(options.keys()),
                weights=list(options.values())
            )[0]

        words.append(nxt)

        if random.random()<D*0.1:
            break

    return " ".join(words).capitalize()+"."