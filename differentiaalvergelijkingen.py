import numpy as np
import matplotlib.pyplot as plt


class GrowthModel:
    def __init__(self, c, V_max):
        self.c = c
        self.V_max = V_max
    
    def growth(self, t, V):
        raise NotImplementedError("This method should be overridden by subclasses.")


class GompertzGrowth(GrowthModel):
    def growth(self, t, V):
        return self.c * V * np.log(self.V_max / V)


class LogisticGrowth(GrowthModel):
    def growth(self, t, V):
        return self.c * V * (1 - V / self.V_max)


def ODE_solver(model, t, V0, dt=0.1):
    V = [V0]  
    for i in range(1, len(t)):
        V_new = V[-1] + dt * model.growth(t[i], V[-1])  
        V.append(V_new)
    return V

def generate_fake_data():
    t_data = [0, 13, 20, 32, 42, 55, 65, 75, 85, 88, 95, 98, 107, 115, 120]
    V_data = [250, 255, 550, 575, 576, 800, 1050, 1250, 1750, 2000, 2550, 2750, 3000, 3500, 4000]
    return t_data, V_data

if __name__ == "__main__":

    t_data, V_data = generate_fake_data()

    V0 = 250
    params = {"c": 0.1, "V_max": 4000}

  
    gompertz_model = GompertzGrowth(c=params["c"], V_max=params["V_max"])
    logistic_model = LogisticGrowth(c=params["c"], V_max=params["V_max"])

    tijd = np.linspace(0, 120, 100)


    V_sim_gompertz = ODE_solver(gompertz_model, tijd, V0)
    V_sim_logistic = ODE_solver(logistic_model, tijd, V0)

    plt.figure(figsize=(10, 6))

    plt.plot(tijd, V_sim_gompertz, label="Gompertz Model", color="blue")
    plt.plot(tijd, V_sim_logistic, label="Logistisch Model", color="green")

    plt.scatter(t_data, V_data, color="red", label="Nepdata (observaties)")

    plt.title("Tumorgroei Modellen vs. Nepdata")
    plt.xlabel("Tijd (dagen)")
    plt.ylabel("Tumorvolume (mm³)")
    plt.legend()
    plt.grid(True)
    plt.show()
