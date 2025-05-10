
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from math import exp, sqrt, tanh

class Barrier:
    def __init__(self, params):
        self.params = params

    def A(self):
        raise NotImplementedError

    def sigma2(self):
        raise NotImplementedError

    @property
    def sigma(self):
        return sqrt(self.sigma2())

# 1. Wasteform barrier
class Wasteform(Barrier):
    def __init__(self, params, f_irf):
        super().__init__(params)
        self.k = params['k']                # dissolution rate constant (yr^-1)
        self.lam = params['lambda']         # decay constant (yr^-1)
        self.f_irf = f_irf                  # instant release fraction

    def A_irf(self):
        return 1.0                          # area for instant release component

    def sigma2_irf(self):
        return 0.0                          # zero variance for instant release

    def A_diss(self):
        return self.k / (self.k + self.lam) # area for dissolution component

    def sigma2_diss(self):
        return 1.0 / (self.k + self.lam)**2 # variance for dissolution component

    def combine(self, A_rest, sigma2_rest):
        # combine instant release and dissolution with other barriers
        term1 = (1.0 - self.f_irf) * self.A_diss() * A_rest / sqrt(self.sigma2_diss() + sigma2_rest)
        term2 = self.f_irf * A_rest / sqrt(sigma2_rest)
        return term1 + term2

# 2. Container barrier
class Container(Barrier):
    def __init__(self, params):
        super().__init__(params)
        self.T0 = params['T0']              # initial failure time (yr)
        self.Tp = params['Tp']              # failure duration (yr)
        self.lam = params['lambda']         # decay constant (yr^-1)

    def A(self):
        # area for uniformly distributed failure
        return exp(-self.lam * self.T0) * (1 - exp(-self.lam * self.Tp)) / (self.lam * self.Tp)

    def sigma2(self):
        # variance for uniformly distributed failure
        lam, Tp = self.lam, self.Tp
        term = exp(-lam * Tp) / (1 - exp(-lam * Tp))
        return 1/lam**2 - Tp**2 * term**2

# 3. Buffer barrier
class Buffer(Barrier):
    def __init__(self, params):
        super().__init__(params)
        self.Q = params['Q_eq']             # equivalent flow rate (m3/yr)
        self.phi = params['phi']            # porosity
        self.rho = params['rho']            # dry density (kg/m3)
        self.Kd = params['Kd']              # sorption coefficient (m3/kg)
        self.V = params['V']                # buffer volume (m3)
        self.lam = params['lambda']         # decay constant (yr^-1)

    def beta(self):
        # effective removal rate
        alpha = self.phi + self.Kd * self.rho
        return self.Q / (alpha * self.V)

    def A(self):
        # area for buffer barrier
        b = self.beta()
        return b / (self.lam + b)

    def sigma2(self):
        # variance for buffer barrier
        b = self.beta()
        return 1.0 / (self.lam + b)**2

# 4. Diffusive geosphere barrier
class DiffusiveGeosphere(Barrier):
    def __init__(self, params):
        super().__init__(params)
        self.phi = params['phi']            # porosity
        self.rho = params['rho']            # dry density (kg/m3)
        self.Kd = params['Kd']              # sorption coefficient (m3/kg)
        self.De = params['De']              # effective diffusion coefficient (m2/s)
        self.L = params['L']                # travel distance (m)
        self.lam = params['lambda']         # decay constant (yr^-1)

    def tau(self):
        # characteristic diffusion time
        R = 1 + self.rho * self.Kd / self.phi
        return self.phi * R * self.L**2 / self.De

    def A(self):
        # area for diffusive barrier
        return 1.0 / np.cosh(sqrt(self.lam * self.tau()))

    def sigma2(self):
        # variance for diffusive barrier
        t = self.tau()
        x = sqrt(self.lam * t)
        return t/(4*self.lam) * (tanh(x)/x - (1/np.cosh(x))**2)

# 5. Advective matrix geosphere barrier
class AdvectiveMatrixGeosphere(Barrier):
    def __init__(self, params):
        super().__init__(params)
        self.R = 1 + params['rho'] * params['Kd'] / params['phi']
        self.T = params['T']                # travel time (yr)
        self.alpha = params['alpha']        # dispersivity (m)
        self.L = params['L']                # travel distance (m)
        self.lam = params['lambda']         # decay constant (yr^-1)

    def psi(self):
        # helper factor for advective transport
        return sqrt(1 + 4*self.alpha*self.R*self.T*self.lam/self.L)

    def A(self):
        # area for advective matrix barrier
        return exp(-self.L*(self.psi()-1)/(2*self.alpha))

    def sigma2(self):
        # variance for advective matrix barrier
        p = self.psi()
        return 2*self.R**2*self.T**2*self.alpha/(self.L * p**3)

# 6. Advective fracture geosphere barrier
class AdvectiveFractureGeosphere(Barrier):
    def __init__(self, params):
        super().__init__(params)
        self.R = 1 + params['rho'] * params['Kd'] / params['phi']
        self.v = params['v']                # fracture flow velocity (m/yr)
        self.D = params['DL']               # fracture dispersivity (m2/yr)
        self.L = params['L']                # travel distance (m)
        self.lam = params['lambda']         # decay constant (yr^-1)
        self.phiF = params['phiF']          # fracture porosity
        self.alphaM = params['alphaM']      # matrix dispersivity parameter (m2/yr)
        self.d = params['d']                # matrix characteristic length (m)

    def TM(self):
        # matrix diffusion time
        return self.alphaM * self.d**2 / self.D

    def eta(self):
        # matrix-fracture interaction parameter
        return 2*self.alphaM / self.phiF

    def psi(self):
        # helper factor for fracture transport
        x = sqrt(self.lam * self.TM())
        return sqrt(1 + 4*self.R*self.D*self.lam/self.v**2 * (1 + self.eta()*tanh(x)/x))

    def A(self):
        # area for advective fracture barrier
        return exp(self.L*self.v/(2*self.D) * (1 - self.psi()))

    def sigma2(self):
        # variance for advective fracture barrier
        p = self.psi(); R,v,L,D,lam = self.R,self.v,self.L,self.D,self.lam
        TM,eta = self.TM(), self.eta()
        x = sqrt(lam*TM)
        term1 = (2*D*R/(v**2*p**2)*(1+eta/2*tanh(x)/x + 1/np.cosh(x)**2))**2
        term2 = eta/(4*lam)*(tanh(x)/x - 1/np.cosh(x)**2)
        term3 = eta/2*sqrt(TM/lam)*1/np.cosh(x)**2*tanh(x)
        return R*L/(v*p