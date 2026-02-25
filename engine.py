import numpy as np
from scipy.integrate import ode

class TTUEngine:
    def __init__(self):
        self.gamma = 1.2    # Gain optimisé pour la complexité
        self.mu = 0.6       # Friction cubique pour la texture sémantique
        self.lambd = 0.05   # Couplage renforcé contre la laminarité
        self.state = [1.0, 0.0, 0.0] 

    def system(self, t, y, impulse):
        phi_c, phi_d, phi_m = y
        # Injection d'un bruit rose interne pour forcer la complexité
        noise = 0.05 * np.sin(10 * t) 
        
        # Triade fondamentale avec modulation chaotique
        d_phi_c = (self.gamma * (impulse + noise)) - phi_d
        d_phi_d = phi_c - (self.mu * (phi_d**3))
        d_phi_m = phi_d - (self.lambd * phi_m)
        return [d_phi_c, d_phi_d, d_phi_m]

    def process_signal(self, impulse):
        # Utilisation du solveur BDF pour les systèmes raides
        solver = ode(self.system).set_integrator('vode', method='bdf')
        solver.set_initial_value(self.state, 0).set_f_params(impulse)
        self.state = solver.integrate(0.1)
        
        # Auto-régulation (Hémostase)
        if self.state[2] > 50: self.mu = 2.0  
        if self.state[2] < 5:  self.mu = 0.2  
        return self.state
