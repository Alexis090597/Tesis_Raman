import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from numpy.polynomial import Polynomial
from pybaselines import Baseline
from sklearn.decomposition import PCA
from itertools import combinations


#Funcion para obtener la interpolacion cúbica de un segmento
def Inter_cubica(x, y, start, end, margen=20, grado=3):
    y_corregido = y.copy()
    n = len(x)
    start_pad = max(start - margen, 0)
    end_pad = min(end + margen, n)
    x_fit = np.concatenate((x[start_pad:start], x[end + 1:end_pad]))
    y_fit = np.concatenate((y[start_pad:start], y[end + 1:end_pad]))

    if len(x_fit) < grado + 1:
        raise ValueError("No hay suficientes puntos para ajustar un polinomio de grado {}".format(grado))

    # Ajuste polinomial
    p = Polynomial.fit(x_fit, y_fit, deg=grado)

    # Evaluar el polinomio en la zona dañada
    x_interp = x[start:end + 1]
    y_interp = p(x_interp)

    # Sustituir en la señal original
    y_corregido[start:end + 1] = y_interp

    return y_corregido


# Leer los datos del archivo
datos = []
with open('Muestras.txt', 'r', encoding='utf-8') as archivo:
    lista = [line.strip() for line in archivo]
for i in range(len(lista)):
    datos.append(np.loadtxt(lista[i], delimiter=','))

# Separar los datos en variables X e Y #870-2048
x = datos[1][:, 0]
a = datos[0][:, 0]
b = datos[0][:, 1]
Res = []

#Constantes para el ruido de cada espectro
""""
#Constantes
Cons=[0.131, 0.22, 0.21, 0.21, 0.221, 0.22, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.151, 0.123,
0.123, 0.123, 0.123, 0.14, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18,
0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.178, 0.178, 0.178, 0.22, 0.23, 0.23, 0.23,
0.23, 0.23, 0.23, 0.24, 0.24, 0.24, 0.25, 0.25, 0.25, 0.25]"""
Cons=[0.131, 0.22, 0.21, 0.21, 0.221, 0.22, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.151, 0.123,
0.123, 0.123, 0.123, 0.14, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18]
# Creacion de vectores Y
for i in range(1, len(datos)):
    Res.append(datos[i][:, 1])
for i in range (len(Res)):
    Res[i]=Res[i]-(b*Cons[i])
for i in range(len(a)):
    a[i]=a[i]-5.75
# Creacion de vector X
FLaser = 532
VLaser = (pow(10, 7)) / FLaser
R1 = []
for i in range(len(x)):
    VRaman = (pow(10, 7)) / (a[i])
    Vdes = VLaser - VRaman
    Vdes = Vdes - 200-5.75#200
    R1.append(Vdes)

# Vectores a usar
Res1 = []
Res2 = []
for i in range(len(Res)):
    Res1.append(Res[i][900:1500])
    Res2.append(Res[i][1501:2048])

# Interpolación cúbica

# Intervalo del ruido del sensor
start, end = 990,1020#299, 309
start2, end2 = 900,920#399, 409
start3, end3 = 1100, 1140
#start4, end4 = 0, 10
#start5, end5 = 820, 900
y = []
ReSmo = []
ReSmo2 = []
corrected = []
corrected2 = []

for i in range(len(Res)):
    y.append(Inter_cubica(R1[900:2048], Res[i][900:2048], start, end))
    y[i] = Inter_cubica(R1[900:2048], y[i], start2, end2)
    y[i] = Inter_cubica(R1[900:2048], y[i], start3, end3)
    #y[i] = Inter_cubica(R1, y[i], start4, end4)
    #y[i] = Inter_cubica(R1, y[i], start5, end5)

# Filtro Savtzky-Goley
    ReSmo.append(savgol_filter(Res1[i], window_length=11, polyorder=2))
    ReSmo2.append(savgol_filter(y[i], window_length=11, polyorder=2))
# Filtro para la flourescencia
    baseline_obj = Baseline()
    baseline, _ = baseline_obj.asls(ReSmo2[i], lam=1e6, p=0.01)
    corrected.append(ReSmo2[i] - baseline)
corrected=corrected+ReSmo2
X = np.array(corrected)

y2 = np.array(['Aceite de Canola']*15 + ['Aceite de Oliva']*15 + ["Aceite de Canola con fluorescencia"]*15 + ["Aceite de Oliva con fluorescencia"]*15)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Cálculo de la razón de Fisher
clases = np.unique(y2)
fisher_ratios_pc1 = {}
fisher_ratios_pc2 = {}

for clase1, clase2 in combinations(clases, 2):
    # PC1
    pc1_1 = X_pca[y2 == clase1, 0]
    pc1_2 = X_pca[y2 == clase2, 0]
    mu1_pc1, mu2_pc1 = np.mean(pc1_1), np.mean(pc1_2)
    var1_pc1, var2_pc1 = np.var(pc1_1), np.var(pc1_2)
    fisher_pc1 = (mu1_pc1 - mu2_pc1)**2 / (var1_pc1 + var2_pc1)
    fisher_ratios_pc1[f"{clase1} vs {clase2}"] = fisher_pc1

    # PC2
    pc2_1 = X_pca[y2 == clase1, 1]
    pc2_2 = X_pca[y2 == clase2, 1]
    mu1_pc2, mu2_pc2 = np.mean(pc2_1), np.mean(pc2_2)
    var1_pc2, var2_pc2 = np.var(pc2_1), np.var(pc2_2)
    fisher_pc2 = (mu1_pc2 - mu2_pc2)**2 / (var1_pc2 + var2_pc2)
    fisher_ratios_pc2[f"{clase1} vs {clase2}"] = fisher_pc2

# Mostrar resultados
print("=== Razón de Fisher para PC1 ===")
for par, valor in fisher_ratios_pc1.items():
    print(f"{par}: {valor:.2f}")

print("\n=== Razón de Fisher para PC2 ===")
for par, valor in fisher_ratios_pc2.items():
    print(f"{par}: {valor:.2f}")

# Plot

plt.figure(figsize=(8, 6))
for etiqueta in np.unique(y2):
    plt.scatter(X_pca[y2 == etiqueta, 0], X_pca[y2 == etiqueta, 1],label=etiqueta)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Aceites")
plt.legend()
plt.grid(True)
plt.tight_layout()
# Mostrar el gráfico
plt.show()
#Este plot sirve para poder mostrar los espectros fuera del PCA
#plt.show()
#plt.plot(a,datos[2],linestyle='-',color='black')
#plt.plot(R1[900:2048], ReSmo2[15], linestyle='-', color='blue')
#plt.plot(R1[900:2048],ReSmo2[18],linestyle='-', color='gray')
#Este plot sirve para mostrar las lineas que limitan el efecto del filtro notch en longitud de onda y en numero de onda
#plt.axvline(x=61 - 200, ymin=0, ymax=1, linestyle='--', color='red')
#plt.axvline(x=337 - 200, ymin=0, ymax=1, linestyle='--', color='red')
#plt.axvline(x=528, ymin=0, ymax=1, linestyle='--', color='red')
#plt.axvline(x=536, ymin=0, ymax=1, linestyle='--', color='red')
# Añadir títulos y etiquetas
#plt.title('Espectro Raman')
#plt.ylabel('Intensidad')
#plt.xlabel('Número de onda cm-1')
