import pandas as pd 
import numpy as np

data = {
         'Hour' : [8,10,12,14,16,18],
         'Temp' : [18, 22, 26, 28, 25, 20],
         'Cloudy_%' : [10, 5, 0, 20, 40, 10],
         'Radiation_W/m^2' : [200, 550, 900, 750, 400, 100],
         'Generation_kW (Target)' : [15, 45, 85, 70, 35, 8]
       }

df = pd.DataFrame(data)
print("An example of dataset for the model")
print(df)
