import numpy as np
from scipy.integrate import ode

class TTUEngine:
    def __init__(self):
        self.gamma = 1.1    
        self.mu = 0.5       
        self.lambd = 0.015  
        self.state = [1.0, 0.0, 0.0] 

    def system(self, t, y, impulse):
        c, d, m = y
        dc = (self.gamma * impulse) - d
        dd = c - (self.mu * (d**3))
        dm = d - (self.lambd * m)
        return [dc, dd, dm]

    def process_signal(self, impulse):
        solver = ode(self.system).set_integrator('vode', method='bdf')
        solver.set_initial_value(self.state, 0).set_f_params(impulse)
        self.state = solver.integrate(0.1)
        
        # Auto-régulation automatique
        if self.state[2] > 50: self.mu = 2.0  
        if self.state[2] < 5:  self.mu = 0.2  
        return self.state
