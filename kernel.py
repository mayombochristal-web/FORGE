import numpy as np
import json
from scipy.integrate import solve_ivp

class TTUKernel:
    def __init__(self, memory_file="memory_phase.json"):
        self.params = {'alpha': 0.0001, 'beta': 0.5, 'gamma': 1.2, 'lambda_': 4.0, 'mu': 3.0}
        self.memory_file = memory_file
        try:
            with open(memory_file, 'r') as f: self.history = json.load(f)
        except: self.history = []
        self.state = self.history[-1]['state'] if self.history else [15.0, 0.5, 0.2]

    def engine(self, t, phi, v_t):
        pm, pc, pd = phi
        d_pm = -self.params['alpha'] * pm + self.params['beta'] * pd
        d_pc = self.params['gamma'] * v_t - self.params['lambda_'] * pc * pd
        d_pd = 0.1 * pc**2 - self.params['mu'] * (pd**3)
        return [d_pm, d_pc, d_pd]

    def process(self, data):
        v_t = np.sin(len(data) * 0.1)
        sol = solve_ivp(self.engine, (0, 40), self.state, args=(v_t,), method='BDF', t_eval=np.linspace(0, 40, 500))
        self.state = sol.y[:, -1].tolist()
        entropy = np.std(np.diff(sol.y[1])) * 20
        substance = "".join([chr(int(abs(p) % 26) + 65) for p in sol.y[0][::100]])
        return sol, {"temp": np.clip(entropy, 0.4, 1.2), "substance": substance}

    def save(self, query, substance):
        self.history.append({"q": query, "s": substance, "state": self.state})
        with open(self.memory_file, 'w') as f: json.dump(self.history[-10:], f)
