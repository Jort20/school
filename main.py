# main.py
from test import TumorGrowthModels  # Importeer de class uit test.py
import numpy as np

# Stel zelf tijd en volume data in
t_data = [0, 13, 20, 32, 42, 55, 65, 75, 85, 88, 95, 98, 107, 115, 120]
V_data = [250, 255, 550, 575, 576, 800, 1050, 1250, 1750, 2000, 2550, 2750, 3000, 3500, 4000]

# Maak een model aan met je eigen data
model = TumorGrowthModels(t_data, V_data)

# Pas de tijdspanne en aantal punten aan
t_vooruit = np.linspace(0, 120, 100)  # Van 0 tot 150 dagen met 150 punten

# Voer model evaluatie uit en visualiseer de resultaten
model.evaluate_models(t_vooruit)
