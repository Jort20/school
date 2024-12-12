import numpy as np
import matplotlib.pyplot as plt

class TumorGrowthModels:
    def __init__(self, t_data, V_data):
        self.t_data = t_data
        self.V_data = V_data

    # Functie om nepdata te genereren
    @staticmethod
    def generate_fake_data():
        t_data = [0, 13, 20, 32, 42, 55, 65, 75, 85, 88, 95, 98, 107, 115, 120]
        V_data = [250, 255, 550, 575, 576, 800, 1050, 1250, 1750, 2000, 2550, 2750, 3000, 3500, 4000]
        return np.array(t_data), np.array(V_data)


    # Runga methode voor numerieke integratie
    @staticmethod
    def Runga_method(model_growth, t, V0, c, V_max, dt):
        V = [V0]
        for i in range(1, len(t)):
            t_current = t[i - 1]
            V_current = V[-1]
            y1 = dt * model_growth(t_current, V_current, c, V_max)
            y2 = dt * model_growth(t_current + dt / 2, V_current + y1 / 2, c, V_max)
            y3 = dt * model_growth(t_current + dt / 2, V_current + y2 / 2, c, V_max)
            y4 = dt * model_growth(t_current + dt, V_current + y3, c, V_max)
            V_new = V_current + (y1 + 2 * y2 + 2 * y3 + y4) / 6
            V.append(V_new)
        return np.array(V)
    
    def montroll_Runga(self, t, V0, c, V_max, d, dt):
        V = [V0]  # Beginwaarde van het volume
        for i in range(1, len(t)):
            t_current = t[i - 1]
            V_current = V[-1]

            # Bereken de groeisnelheid voor Montroll's model
            y1 = dt * c * V_current * (V_max**d - V_current**d)
            y2 = dt * c * (V_current + y1 / 2) * (V_max**d - (V_current + y1 / 2)**d)
            y3 = dt * c * (V_current + y2 / 2) * (V_max**d - (V_current + y2 / 2)**d)
            y4 = dt * c * (V_current + y3) * (V_max**d - (V_current + y3)**d)

            # Bereken de nieuwe waarde van V
            V_new = V_current + (y1 + 2 * y2 + 2 * y3 + y4) / 6
            V.append(V_new)

        return np.array(V)
    
    def allee_Runga(self, t, V0, c, V_min, V_max, dt):
        V = [V0]  # Beginwaarde van het volume
        for i in range(1, len(t)):
            t_current = t[i - 1]
            V_current = V[-1]

            # Bereken de groeisnelheid voor het Allee-effect model (direct in de Runga methode)
            y1 = dt * c * (V_current - V_min) * (V_max - V_current)
            y2 = dt * c * (V_current + y1 / 2 - V_min) * (V_max - (V_current + y1 / 2))
            y3 = dt * c * (V_current + y2 / 2 - V_min) * (V_max - (V_current + y2 / 2))
            y4 = dt * c * (V_current + y3 - V_min) * (V_max - (V_current + y3))

            # Bereken de nieuwe waarde van V
            V_new = V_current + (y1 + 2 * y2 + 2 * y3 + y4) / 6
            V.append(V_new)

        return np.array(V)
    
    @staticmethod
    def fit_model_brute_force(model_wrapper, t_data, V_data, p0, num_iterations=10000, step_size=0.01):
        def model(t, *params):
            return model_wrapper(t, *params, V_data)

        best_params = p0
        best_cost = np.sum((model(t_data, *best_params) - V_data) ** 2)

        for _ in range(num_iterations):
            new_params = best_params + np.random.uniform(-step_size, step_size, len(p0))
            cost = np.sum((model(t_data, *new_params) - V_data) ** 2)

            # Gradients can be used here for more informed steps, rather than random perturbation
            if cost < best_cost:
                best_params = new_params
                best_cost = cost

        return best_params
    
    @staticmethod
    def calculate_aic(n, rss, k):
        return n * np.log(rss / n) + 2 * k

    @staticmethod
    def calculate_bic(n, rss, k):
        return n * np.log(rss / n) + k * np.log(n)

    @staticmethod
    def calculate_residuals(V_data, V_sim):
        residuals = V_data - V_sim
        rss = np.sum(residuals**2)
        return rss        
    
class TumorGrowthModels:
    # Runga-Kutta method for solving differential equations
    @staticmethod
    def Runga_method(growth_function, t, V0, *params):
        V = [V0]
        dt = t[1] - t[0]
        for i in range(1, len(t)):
            k1 = growth_function(t[i-1], V[-1], *params)
            k2 = growth_function(t[i-1] + dt/2, V[-1] + dt*k1/2, *params)
            k3 = growth_function(t[i-1] + dt/2, V[-1] + dt*k2/2, *params)
            k4 = growth_function(t[i-1] + dt, V[-1] + dt*k3, *params)
            V_new = V[-1] + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            V.append(V_new)
        return np.array(V)

class LogisticModel(TumorGrowthModels):
    @staticmethod
    def logistic_growth(t, V, c, V_max):
        return c * V * (V_max - V)

    def logistic_Runga(self, t, V0, c, V_max, dt):
        return TumorGrowthModels.Runga_method(self.logistic_growth, t, V0, c, V_max)

    @staticmethod
    def logistic_wrapper(t, c, V_max, V_data):
        V0 = V_data[0]
        dt = t[1] - t[0]
        V = [V0]
        for i in range(1, len(t)):
            V_new = V[-1] + dt * LogisticModel.logistic_growth(t[i], V[-1], c, V_max)
            V.append(V_new)
        return np.array(V)

class GompertzModel(TumorGrowthModels):
    @staticmethod
    def gompertz_growth(t, V, c, V_max):
        return c * V * np.log(V_max / V)

    def gompertz_Runga(self, t, V0, c, V_max, dt):
        return TumorGrowthModels.Runga_method(self.gompertz_growth, t, V0, c, V_max)

    @staticmethod
    def gompertz_wrapper(t, c, V_max, V_data):
        V0 = V_data[0]
        dt = t[1] - t[0]
        V = [V0]
        for i in range(1, len(t)):
            V_new = V[-1] + dt * GompertzModel.gompertz_growth(t[i], V[-1], c, V_max)
            V.append(V_new)
        return np.array(V)

class VonBertalanffyModel(TumorGrowthModels):
    @staticmethod
    def von_bertalanffy_growth(t, V, c, d):
        return c * V**(2/3) - d * V

    def von_bertalanffy_runga(self, t, V0, c, d, dt):
        return TumorGrowthModels.Runga_method(self.von_bertalanffy_growth, t, V0, c, d)

    @staticmethod
    def von_bertalanffy_wrapper(t, c, d, V_data):
        V0 = V_data[0]
        dt = t[1] - t[0]
        V = [V0]
        for i in range(1, len(t)):
            V_new = V[-1] + dt * VonBertalanffyModel.von_bertalanffy_growth(t[i], V[-1], c, d)
            V.append(V_new)
        return np.array(V)

class MendelsohnModel(TumorGrowthModels):
    @staticmethod
    def mendelsohn_growth(t, V, c, D):
        return c * V**D

    def mendelsohn_Runga(self, t, V0, c, D, dt):
        return TumorGrowthModels.Runga_method(self.mendelsohn_growth, t, V0, c, D)

    @staticmethod
    def mendelsohn_wrapper(t, c, D, V_data):
        V0 = V_data[0]
        dt = t[1] - t[0]
        V = [V0]
        for i in range(1, len(t)):
            V_new = V[-1] + dt * MendelsohnModel.mendelsohn_growth(t[i], V[-1], c, D)
            V.append(V_new)
        return np.array(V)

class MontrollModel(TumorGrowthModels):
    @staticmethod
    def montroll_growth(t, V, c, V_max, d):
        return c * V * (V_max**d - V**d)

    def montroll_Runge(self, t, V0, c, V_max, d, dt):
        return TumorGrowthModels.Runga_method(self.montroll_growth, t, V0, c, V_max, d)

    @staticmethod
    def montroll_wrapper(t, c, V_max, d, V_data):
        V0 = V_data[0]
        dt = t[1] - t[0]
        V = [V0]
        for i in range(1, len(t)):
            V_new = V[-1] + dt * MontrollModel.montroll_growth(t[i], V[-1], c, V_max, d)
            V.append(V_new)
        return np.array(V)

class AlleeEffectModel(TumorGrowthModels):
    @staticmethod
    def allee_growth(t, V, c, V_min, V_max):
        if V <= V_min or V >= V_max:
            return 0
        return c * (V - V_min) * (V_max - V)

    def runga_allee(self, t, V0, c, V_min, V_max, dt):
        return TumorGrowthModels.Runga_method(self.allee_growth, t, V0, c, V_min, V_max)

    @staticmethod
    def allee_wrapper(t, c, V_min, V_max, V_data):
        V0 = V_data[0]
        dt = t[1] - t[0]
        V = [V0]
        for i in range(1, len(t)):
            V_new = V[-1] + dt * AlleeEffectModel.allee_growth(t[i], V[-1], c, V_min, V_max)
            V.append(V_new)
        return np.array(V)
    
