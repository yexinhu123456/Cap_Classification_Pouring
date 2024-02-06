import pickle 
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score







# name = ["final_water_estimation_30", "final_water_estimation_40", "final_water_estimation_50", "final_water_estimation_60", "final_water_estimation_70",
#         "final_water_estimation_80", "final_water_estimation_90", "final_water_estimation_100"]

# name = ["final_vinegar_estimation_30", "final_vinegar_estimation_40", "final_vinegar_estimation_50", "final_vinegar_estimation_60", "final_vinegar_estimation_70",
#         "final_vinegar_estimation_80", "final_vinegar_estimation_90", "final_vinegar_estimation_100"]

# name = ["final_lentils_estimation_30", "final_lentils_estimation_40", "final_lentils_estimation_50", "final_lentils_estimation_60", "final_lentils_estimation_70",
#           "final_lentils_estimation_80", "final_lentils_estimation_90", "final_lentils_estimation_100"]

# name = ["final_rice_estimation_30", "final_rice_estimation_40", "final_rice_estimation_50", "final_rice_estimation_60", "final_rice_estimation_70",
#           "final_rice_estimation_80", "final_rice_estimation_90", "final_rice_estimation_100"]


name = ["final_oil_estimation_30", "final_oil_estimation_40", "final_oil_estimation_50", "final_oil_estimation_60", "final_oil_estimation_70",
          "final_oil_estimation_80", "final_oil_estimation_90", "final_oil_estimation_100"]

weight_overpour = []
weight = []
for i in range(8):
    k = name[i]


    # os.chdir(f"C:/Users/yexin/Desktop/liquid/data/data_collection_1012{i:02}")
    # os.chdir(f"C:/Users/yexin/Desktop/liquid/data/data_collection_09140{i}")
    os.chdir(f"C:/Users/yexin/Desktop/liquid/data/data_collection_{k}")
    for f in os.listdir():
        data = pickle.load(open(f, "rb"))
        
        
        w_mean = np.mean((data[1][-200 : -1]).astype(np.float32))
        w = data[2][-1].astype(np.float32)
        
        weight.append(w)
        weight_overpour.append(w_mean - w)
        
        
weight_overpour = np.array(weight_overpour)[:, np.newaxis]
weight = np.array(weight)[:, np.newaxis]





plt.scatter(weight, weight_overpour)
plt.show()
        


poly_features = PolynomialFeatures(degree=2)
weight_1 = poly_features.fit_transform(weight.reshape(-1, 1))
print(weight_1.shape)
model = LinearRegression().fit(weight_1, weight_overpour)
poly_pred = model.predict(weight_1)

# a = np.linspace(30, 100)[:, np.newaxis]
# a_1 = poly_features.fit_transform(a.reshape(-1, 1))
# a_pred = model.predict(a_1)
# plt.plot(a, a_pred.reshape(-1), c = "r")

plt.scatter(weight, weight_overpour)

weight_plot = np.linspace(30, 110, 81)[:, np.newaxis]
weight_plot_1 = poly_features.transform(weight_plot.reshape(-1, 1))
print(weight_plot.shape)
plot_pred = model.predict(weight_plot_1)
plt.plot(weight_plot, plot_pred)

plt.title("Overpouring Weight VS Stop Weight (Oil)")
plt.xlabel("stop weight [g]")
plt.ylabel("overpouring weight [g]")
plt.show()  

with open(r'C:\Users\yexin\Desktop\liquid\ckpts\poly_features_oil_final.pkl', 'wb') as file:
    pickle.dump(poly_features, file)


with open(r'C:\Users\yexin\Desktop\liquid\ckpts\linear_regression_model_oil_final.pkl', 'wb') as file:
    pickle.dump(model, file)

