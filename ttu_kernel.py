import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Tuple, Dict

class TTU_Master_Kernel:
    def __init__(self, params: Optional[Dict[str, float]] = None, initial_state: Optional[list] = None):
        self.params = params or {
            'alpha': 0.0001,
            'beta': 0.5,
            'gamma': 1.2,
            'lambda_': 4.0,
            'mu': 3.0
        }
        self.phi_init = initial_state or [15.0, 0.5, 0.2]
        self.solution = None
        self.substance = ""

    def engine(self, t: float, phi: list, v_t: float) -> list:
        pm, pc, pd = phi
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params['gamma']
        lambda_ = self.params['lambda_']
        mu = self.params['mu']

        d_pm = -alpha * pm + beta * pd
        d_pc = gamma * v_t - lambda_ * pc * pd
        d_pd = 0.1 * pc**2 - mu * (pd**3)
        return [d_pm, d_pc, d_pd]

    def run_sequence(self,
                     t_span: Tuple[float, float] = (0, 500),
                     t_eval: Optional[np.ndarray] = None,
                     signal_func=None,
                     method: str = 'BDF'):
        if signal_func is None:
            signal_func = lambda t: 0.0

        def dynamics(t, phi):
            v_t = signal_func(t)
            return self.engine(t, phi, v_t)

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 10000)

        sol = solve_ivp(dynamics, t_span, self.phi_init,
                        method=method, t_eval=t_eval)
        self.solution = sol
        return sol

    def extract_substance(self, sampling_rate: int = 500) -> str:
        if self.solution is None:
            return ""
        pm = self.solution.y[0]
        sampled = pm[::sampling_rate]
        ascii_codes = [32 + (int(abs(val) * 10) % 94) for val in sampled]
        substance = "".join(chr(code) for code in ascii_codes)
        self.substance = substance
        return substance

    def get_attractor_data(self):
        if self.solution is None:
            return None, None
        return self.solution.y[1], self.solution.y[2]