# test.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

    # Groeimodellen
    @staticmethod
    def logistic_growth(t, V, c, V_max):
        return c * V * (1 - V / V_max)

    @staticmethod
    def gompertz_growth(t, V, c, V_max):
        return c * V * np.log(V_max / V)

    @staticmethod
    def von_bertalanffy_growth(t, V, c, d):
        return c * V**(2/3) - d * V

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

    def gompertz_Runga(self, t, V0, c, V_max, dt):
        return self.Runga_method(self.gompertz_growth, t, V0, c, V_max, dt)

    def logistic_Runga(self, t, V0, c, V_max, dt):
        return self.Runga_method(self.logistic_growth, t, V0, c, V_max, dt)

    def von_bertalanffy_runga(self, t, V0, c, d, dt):
        return self.Runga_method(self.von_bertalanffy_growth, t, V0, c, d, dt)

    # Wrapper functies voor curve fitting
    @staticmethod
    def gompertz_wrapper(t, c, V_max):
        V0 = 250  
        dt = t[1] - t[0]
        V = [V0]
        for i in range(1, len(t)):
            V_new = V[-1] + dt * TumorGrowthModels.gompertz_growth(t[i], V[-1], c, V_max)
            V.append(V_new)
        return np.array(V)

    @staticmethod
    def logistic_wrapper(t, c, V_max):
        V0 = 250  
        dt = t[1] - t[0]
        V = [V0]
        for i in range(1, len(t)):
            V_new = V[-1] + dt * TumorGrowthModels.logistic_growth(t[i], V[-1], c, V_max)
            V.append(V_new)
        return np.array(V)

    @staticmethod
    def von_bertalanffy_wrapper(t, c, d):
        V0 = 250  
        dt = t[1] - t[0]
        V = [V0]
        for i in range(1, len(t)):
            V_new = V[-1] + dt * TumorGrowthModels.von_bertalanffy_growth(t[i], V[-1], c, d)
            V.append(V_new)
        return np.array(V)

    # Model fitting met curve_fit
    @staticmethod
    def fit_model(model_wrapper, t_data, V_data, p0):
        popt, _ = curve_fit(model_wrapper, t_data, V_data, p0=p0)
        return popt

    # AIC en BIC berekeningen
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

    # Model evaluatie, selectie en visualisatie
    def evaluate_models(self, t_vooruit=None):
        # Als t_vooruit niet wordt meegegeven, gebruik de standaardwaarde
        if t_vooruit is None:
            t_vooruit = np.linspace(0, 120, 100)  # Standaardtijdspanne van 0 tot 120 dagen met 100 punten

        initial_params = [0.1, 4000]
        initial_params_von_bertalanffy = [0.1, 0.01]

        # Pas modellen aan
        params_gompertz = self.fit_model(self.gompertz_wrapper, self.t_data, self.V_data, p0=initial_params)
        params_logistic = self.fit_model(self.logistic_wrapper, self.t_data, self.V_data, p0=initial_params)
        params_von_bertalanffy = self.fit_model(self.von_bertalanffy_wrapper, self.t_data, self.V_data, p0=initial_params_von_bertalanffy)

        # Simuleer data
        dt = t_vooruit[1] - t_vooruit[0]
        V_sim_gompertz = self.gompertz_Runga(t_vooruit, 250, *params_gompertz, dt)
        V_sim_logistic = self.logistic_Runga(t_vooruit, 250, *params_logistic, dt)
        V_sim_von_bertalanffy = self.von_bertalanffy_runga(t_vooruit, 250, *params_von_bertalanffy, dt)

        # Visualisatie
        plt.figure(figsize=(10, 6))
        plt.scatter(self.t_data, self.V_data, color="red", label="Data")
        plt.plot(t_vooruit, V_sim_gompertz, label=f"Gompertz Model", color="blue")
        plt.plot(t_vooruit, V_sim_logistic, label=f"Logistic Model", color="green")
        plt.plot(t_vooruit, V_sim_von_bertalanffy, label=f"Von Bertalanffy Model", color="purple")
        plt.title("Tumorgroei Modellen vs. Data")
        plt.xlabel("Tijd (dagen)")
        plt.ylabel("Tumorvolume (mm³)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Bereken AIC en BIC
        rss_gompertz = self.calculate_residuals(self.V_data, self.gompertz_wrapper(self.t_data, *params_gompertz))
        rss_logistic = self.calculate_residuals(self.V_data, self.logistic_wrapper(self.t_data, *params_logistic))
        rss_von_bertalanffy = self.calculate_residuals(self.V_data, self.von_bertalanffy_wrapper(self.t_data, *params_von_bertalanffy))

        n = len(self.V_data)
        k_gompertz = len(params_gompertz)
        k_logistic = len(params_logistic)
        k_von_bertalanffy = len(params_von_bertalanffy)

        aic_gompertz = self.calculate_aic(n, rss_gompertz, k_gompertz)
        bic_gompertz = self.calculate_bic(n, rss_gompertz, k_gompertz)
        aic_logistic = self.calculate_aic(n, rss_logistic, k_logistic)
        bic_logistic = self.calculate_bic(n, rss_logistic, k_logistic)
        aic_von_bertalanffy = self.calculate_aic(n, rss_von_bertalanffy, k_von_bertalanffy)
        bic_von_bertalanffy = self.calculate_bic(n, rss_von_bertalanffy, k_von_bertalanffy)

        print("AIC en BIC resultaten:")
        print(f"Gompertz: AIC = {aic_gompertz:.2f}, BIC = {bic_gompertz:.2f}")
        print(f"Logistic: AIC = {aic_logistic:.2f}, BIC = {bic_logistic:.2f}")
        print(f"Von Bertalanffy: AIC = {aic_von_bertalanffy:.2f}, BIC = {bic_von_bertalanffy:.2f}")
