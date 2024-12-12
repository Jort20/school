import numpy as np
import matplotlib.pyplot as plt
from test import *


def evaluate_models(self, t_vooruit=None):
        # Als t_vooruit niet wordt meegegeven, gebruik de standaardwaarde
        if t_vooruit is None:
            t_vooruit = np.linspace(0, 120, len(self.ts))  # Standaardtijdspanne van 0 tot 120 dagen met 100 punten



        # Pas modellen aan
        params_mendelsohn = Tumorgrowthmodel.fit_model_brute_force(self.mendelsohn_wrapper, self.t_data, self.V_data, p0=[0.01, 0.1], num_iterations=10000)
        params_gompertz = self.fit_model_brute_force(self.gompertz_wrapper, self.t_data, self.V_data, p0=[0.11, 7.7], num_iterations=10000)
        params_logistic = self.fit_model_brute_force(self.logistic_wrapper, self.t_data, self.V_data, p0=[0.01, 7], num_iterations=10000)
        params_von_bertalanffy = self.fit_model_brute_force(self.von_bertalanffy_wrapper, self.t_data, self.V_data, p0=[0.5,0.2], num_iterations=10000)
        params_montroll = self.fit_model_brute_force(self.montroll_wrapper, self.t_data, self.V_data, p0=[0.01, 8, 0.1], num_iterations=10000)
        params_allee = self.fit_model_brute_force(self.allee_wrapper, self.t_data, self.V_data, p0=[0.05, 0, 7.5], num_iterations=10000)
        # Simuleer data
        dt = t_vooruit[1] - t_vooruit[0]
        V_sim_gompertz = self.gompertz_Runga(t_vooruit, self.V_data[0], *params_gompertz, dt)
        V_sim_logistic = self.logistic_Runga(t_vooruit, self.V_data[0], *params_logistic, dt)
        V_sim_von_bertalanffy = self.von_bertalanffy_runga(t_vooruit, self.V_data[0], *params_von_bertalanffy, dt)
        V_sim_mendelsohn = self.mendelsohn_Runga(t_vooruit, self.V_data[0], *params_mendelsohn, dt)
        V_sim_montroll = self.montroll_Runga(t_vooruit, self.V_data[0], *params_montroll, dt)
        V_sim_allee = self.allee_Runga(t_vooruit, self.V_data[0], *params_allee, dt)
        # Fit Montroll's model met brute force
        


        # Visualisatie
        plt.figure(figsize=(10, 6))
        plt.scatter(self.t_data, self.V_data, color="red", label="Data")
        plt.plot(t_vooruit, V_sim_mendelsohn, label=f"Mendelsohn Model, params={params_mendelsohn}", color="orange")
        plt.plot(t_vooruit, V_sim_gompertz, label=f"Gompertz Model, params={params_gompertz}", color="blue")
        plt.plot(t_vooruit, V_sim_logistic, label=f"Logistic Model, params={params_logistic}", color="green")
        plt.plot(t_vooruit, V_sim_von_bertalanffy, label=f"Von Bertalanffy Model, params={params_von_bertalanffy}", color="purple")
        plt.plot(t_vooruit, V_sim_montroll, label=f"Montroll Model, params={params_montroll}", color="pink")
        plt.plot(t_vooruit, V_sim_allee, label=f"Allee Effect Model, params={params_allee}", color="brown")
        plt.title("Tumorgroei Modellen vs. Data")
        plt.xlabel("Tijd (dagen)")
        plt.ylabel("Tumorvolume (mm³)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Bereken AIC en BIC
        rss_gompertz = self.calculate_residuals(self.V_data,self.gompertz_wrapper(self.t_data, *params_gompertz, self.V_data))
        rss_logistic = self.calculate_residuals(self.V_data, self.logistic_wrapper(self.t_data, *params_logistic,self.V_data))
        rss_von_bertalanffy = self.calculate_residuals(self.V_data, self.von_bertalanffy_wrapper(self.t_data, *params_von_bertalanffy,self.V_data))

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

        rss_mendelsohn = self.calculate_residuals(self.V_data, self.mendelsohn_wrapper(self.t_data, *params_mendelsohn, self.V_data))
        aic_mendelsohn = self.calculate_aic(n, rss_mendelsohn, len(params_mendelsohn))
        bic_mendelsohn = self.calculate_bic(n, rss_mendelsohn, len(params_mendelsohn))

        rss_montroll = self.calculate_residuals(self.V_data, self.montroll_wrapper(self.t_data, *params_montroll,self.V_data))
        aic_montroll = self.calculate_aic(n,rss_montroll,len(params_montroll))
        bic_montroll = self.calculate_bic(n,rss_montroll,len(params_montroll))

        rss_allee = self.calculate_residuals(self.V_data, self.allee_wrapper(self.t_data, *params_allee,self.V_data))
        aic_allee = self.calculate_aic(n,rss_allee,len(params_allee))
        bic_allee = self.calculate_bic(n,rss_allee,len(params_allee))



        import pandas as pd

        # Voorbeeld van AIC en BIC waarden
        models = ['Gompertz', 'Logistic', 'Von Bertalanffy', 'Mendelsohn', 'Montroll','allee']
        aic_values = [aic_gompertz, aic_logistic, aic_von_bertalanffy, aic_mendelsohn, aic_montroll,aic_allee]
        bic_values = [bic_gompertz, bic_logistic, bic_von_bertalanffy, bic_mendelsohn, bic_montroll,bic_allee]

        # Creëer een DataFrame
        df = pd.DataFrame({
            'Model': models,
            'AIC': aic_values,
            'BIC': bic_values
        })

        # Sorteer de tabel op AIC (of BIC)
        df_sorted_aic = df.sort_values(by='AIC', ascending=True)  # Sorteer van hoog naar laag AIC
        df_sorted_bic = df.sort_values(by='BIC', ascending=True)  # Sorteer van hoog naar laag BIC

        # Print de gesorteerde tabellen
        print("AIC gesorteerd van laag naar hoog:")
        print(df_sorted_aic)

        print("\nBIC gesorteerd van laag naar hoog:")
        print(df_sorted_bic)