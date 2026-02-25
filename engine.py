import numpy as np
from scipy.integrate import ode

class TTUEngine:
    def __init__(self, gamma=1.35, mu=0.75, lambd=0.08):
        self.gamma = gamma
        self.mu = mu
        self.lambd = lambd
        self.state = [1.0, 0.0, 0.0] 

    def system(self, t, y, impulse):
        phi_c, phi_d, phi_m = y
        instability = 0.1 * np.cos(15 * t) * np.exp(-0.1 * t)
        d_phi_c = (self.gamma * (impulse + instability)) - phi_d
        d_phi_d = phi_c - (self.mu * (phi_d**3))
        d_phi_m = phi_d - (self.lambd * phi_m)
        return [d_phi_c, d_phi_d, d_phi_m]

    def process_signal(self, impulse):
        solver = ode(self.system).set_integrator('vode', method='bdf', order=15)
        solver.set_initial_value(self.state, 0).set_f_params(impulse)
        self.state = solver.integrate(0.1)
        self.state = [np.clip(s, -100, 100) for s in self.state]
        if abs(self.state[1]) > 10: self.mu = 2.5
        else: self.mu = 0.6
        return self.state