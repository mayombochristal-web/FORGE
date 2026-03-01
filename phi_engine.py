# S+01 — PHI ENGINE

import math

def init_phi():
    return {
        "phi_m":0.5,
        "phi_c":0.5,
        "phi_d":0.5
    }

def evolve_phi(phi, excitation=0.1):

    phi["phi_m"]=max(0.1,min(1.0,
        phi["phi_m"]+(excitation*0.2)-0.01))

    phi["phi_c"]=max(0.1,min(1.0,
        phi["phi_c"]+(excitation*0.5)-0.05))

    phi["phi_d"]=max(0.1,min(1.0,
        phi["phi_d"]+(excitation*0.1)-0.02))

    return phi

def phi_intensity(phi):
    return math.sqrt(
        phi["phi_m"]**2 +
        phi["phi_c"]**2 +
        phi["phi_d"]**2
    )