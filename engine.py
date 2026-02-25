# Dans engine.py, modifiez la fonction process_signal
def process_signal(self, impulse):
    solver = ode(self.system).set_integrator('vode', method='bdf')
    solver.set_initial_value(self.state, 0).set_f_params(impulse)
    self.state = solver.integrate(0.1)
    
    # BRIDAGE DE SÉCURITÉ (Évite les nombres géants)
    for i in range(3):
        if self.state[i] > 100: self.state[i] = 100 # Plafond à 100
        if self.state[i] < -100: self.state[i] = -100 # Plancher à -100
        
    return self.state
