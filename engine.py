def process_signal(self, impulse):
    solver = ode(self.system).set_integrator('vode', method='bdf')
    solver.set_initial_value(self.state, 0).set_f_params(impulse)
    self.state = solver.integrate(0.1)
    
    # Écrêtage Triadique : On maintient les variables dans le monde réel
    self.state = [np.clip(s, -100, 100) for s in self.state]
    return self.state
