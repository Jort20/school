import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def generate_fake_data():
    t_data = [0, 13, 20, 32, 42, 55, 65, 75, 85, 88, 95, 98, 107, 115, 120]
    V_data = [250, 255, 550, 575, 576, 800, 1050, 1250, 1750, 2000, 2550, 2750, 3000, 3500, 4000]
    return np.array(t_data), np.array(V_data)

def logistic_growth(t, V, c, V_max):
    return c * V * (1 - V / V_max)

def gompertz_growth(t, V, c, V_max):
    return c * V * np.log(V_max / V)

def Hean_method(model_growth, t, V0, c, V_max, dt):
    V = [V0]
    for i in range(1, len(t)):
        t_current = t[i-1]
        V_current = V[-1]
        
        y1 = model_growth(t_current, V_current, c, V_max)
        V_euler = V_current + dt * y1
        y2 = model_growth(t_current + dt, V_euler, c, V_max)
        
        V_new = V_current + dt * 0.5 * (y1 + y2)
        V.append(V_new)
    
    return np.array(V)

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

def gompertz_hean(t, V0, c, V_max, dt):
    return Hean_method(gompertz_growth, t, V0, c, V_max, dt)

def logistic_hean(t, V0, c, V_max, dt):
    return Hean_method(logistic_growth, t, V0, c, V_max, dt)

def gompertz_Runga(t, V0, c, V_max, dt):
    return Runga_method(gompertz_growth, t, V0, c, V_max, dt)

def logistic_Runga(t, V0, c, V_max, dt):
    return Runga_method(logistic_growth, t, V0, c, V_max, dt)

def gompertz_wrapper(t, c, V_max):
    """Wrapper om Gompertz-growth compatibel te maken met curve_fit."""
    V0 = 250  
    dt = t[1] - t[0]  
    V = [V0]
    for i in range(1, len(t)):
        V_new = V[-1] + dt * gompertz_growth(t[i], V[-1], c, V_max)
        V.append(V_new)
    return np.array(V)

def logistic_wrapper(t, c, V_max):
    """Wrapper om Logistic-growth compatibel te maken met curve_fit."""
    V0 = 250  
    dt = t[1] - t[0]  
    V = [V0]
    for i in range(1, len(t)):
        V_new = V[-1] + dt * logistic_growth(t[i], V[-1], c, V_max)
        V.append(V_new)
    return np.array(V)

def fit_model(model_wrapper, t_data, V_data, p0):
    """
    Past het model aan de data aan en retourneert de optimale parameters.
    """
    popt, pcov = curve_fit(model_wrapper, t_data, V_data, p0=p0)
    return popt

if __name__ == "__main__":
    t_data, V_data = generate_fake_data()
    tijd = np.linspace(0, 120, 100)

    initial_params = [0.1, 4000]  

    params_gompertz = fit_model(gompertz_wrapper, t_data, V_data, p0=initial_params)
    params_logistic = fit_model(logistic_wrapper, t_data, V_data, p0=initial_params)

    dt = tijd[1] - tijd[0]  
    V_sim_gompertz = gompertz_hean(tijd, 250, *params_gompertz, dt)
    V_sim_logistic = logistic_hean(tijd, 250, *params_logistic, dt)

    plt.figure(figsize=(10, 6))

    plt.scatter(t_data, V_data, color="red", label="Nepdata (observaties)")
    plt.plot(tijd, V_sim_gompertz, label=f"Gompertz Model\n(c={params_gompertz[0]:.3f}, V_max={params_gompertz[1]:.1f})", color="blue")
    plt.plot(tijd, V_sim_logistic, label=f"Logistic Model\n(c={params_logistic[0]:.3f}, V_max={params_logistic[1]:.1f})", color="green")

    plt.title("Tumorgroei Modellen vs. Nepdata")
    plt.xlabel("Tijd (dagen)")
    plt.ylabel("Tumorvolume (mm³)")
    plt.legend()
    plt.grid(True)
    plt.show()

    V_sim_gompertz = gompertz_Runga(tijd, 250, *params_gompertz, dt)
    V_sim_logistic = logistic_Runga(tijd, 250, *params_logistic, dt)

    plt.figure(figsize=(10, 6))

    plt.scatter(t_data, V_data, color="red", label="Nepdata (observaties)")
    plt.plot(tijd, V_sim_gompertz, label=f"Gompertz Model\n(c={params_gompertz[0]:.3f}, V_max={params_gompertz[1]:.1f})", color="blue")
    plt.plot(tijd, V_sim_logistic, label=f"Logistic Model\n(c={params_logistic[0]:.3f}, V_max={params_logistic[1]:.1f})", color="green")

    plt.title("Tumorgroei Modellen vs. Nepdata")
    plt.xlabel("Tijd (dagen)")
    plt.ylabel("Tumorvolume (mm³)")
    plt.legend()
    plt.grid(True)
    plt.show()
