import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# CONFIGURACIÓN DE RUTA DINÁMICA
ruta_raiz = os.path.dirname(os.path.abspath(__file__))
ruta_dataset = os.path.join(ruta_raiz, 'dataset', 'iris.csv')

# Control de existencia de archivo
if not os.path.exists(ruta_dataset):
    print(f" Error crítico: No se encuentra 'iris.csv' en la ruta {ruta_dataset}")
    print(" Por favor, ejecuta primero el script 'crear_dataset.py' adentro de la carpeta 'dataset'.")
    exit()

# Carga del dataset original
df = pd.read_csv(ruta_dataset)

# =========================================================================
# PASO 1: VERIFICACIÓN Y VALIDACIÓN DEL PERCEPTRÓN (SUPERVISADO)
# =========================================================================

umbral = 0.5  
opciones_pesos = {
    'A': [1.2, -0.15, -0.47],
    'B': [5.9, -1.21, -2.48],
    'C': [6.3, -1.08, -2.05]
}

datos_examen = []
for _, fila in df.iterrows():
    entradas = (1.0, float(fila["petal.length"]), float(fila["petal.width"]))
    salida_deseada = 1 if fila["variety"] == "Setosa" else 0
    datos_examen.append((entradas, salida_deseada))

def evaluar_perceptron(valores, pesos):
    suma_neta = sum(v * w for v, w in zip(valores, pesos))
    return 1 if suma_neta > umbral else 0, suma_neta

print("=========================================================================")
print(" VERIFICACIÓN DE PESOS Y CÁLCULO DE OPTIMALIDAD GEOMÉTRICA")
print("=========================================================================")
print(f"{'Opcion':<10}{'w(0)':>8}{'w(1)':>8}{'w(2)':>8}{'Precision':>13}{'Válido':>10}{'Margen Seg.':>14}")
print("-------------------------------------------------------------------------")

mejor_opcion = None
max_margen = -1
pesos_optimos = []

for opcion, pesos in opciones_pesos.items():
    aciertos = 0
    margenes_distancia = []

    for entradas, objetivo in datos_examen:
        prediccion, suma_neta = evaluar_perceptron(entradas, pesos)
        if prediccion == objetivo:
            aciertos += 1
            margenes_distancia.append(abs(suma_neta - umbral))

    precision = (aciertos / len(datos_examen)) * 100
    valido = "[SI]" if precision == 100.0 else "[NO]"
    margen_promedio = np.mean(margenes_distancia) if precision == 100.0 else 0.0

    print(f"  {opcion:<8}{pesos[0]:>8.2f}{pesos[1]:>8.2f}{pesos[2]:>8.2f}{precision:>11.1f}%{valido:>10}{margen_promedio:>13.2f}")

    if precision == 100.0 and margen_promedio > max_margen:
        max_margen = margen_promedio
        mejor_opcion = opcion
        pesos_optimos = pesos

print("=========================================================================")

if mejor_opcion:
    print("\n==================================================")
    print(f"🎯 Pesos finales óptimos: {[round(w, 2) for w in pesos_optimos]}")
    print("==================================================")


# =========================================================================
# PASO 2: AGRUPAMIENTO AUTOMÁTICO K-MEANS (NO SUPERVISADO)
# =========================================================================

print("\nEjecutando agrupamiento K-Means...")
X = df[['petal.length', 'petal.width']].values

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
grupos = kmeans.labels_

plt.figure(figsize=(7, 4.5))
colores_especies = ['red', 'green', 'blue']  

for i in range(3):
    puntos_cluster = X[grupos == i]
    plt.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], c=colores_especies[i], s=50, alpha=0.8, label=f'Cluster {i}')

plt.xlabel('PetalLengthCm (Eje X)')
plt.ylabel('PetalWidthCm (Eje Y)')
plt.title('Agrupamiento Automático K-Means')
plt.grid(True)
plt.legend()

# Guardado automático en disco 
ruta_grafico = os.path.join(ruta_raiz, 'resultado_kmeans.png')
plt.savefig(ruta_grafico, dpi=300)
plt.close()

print("\n==================================================")
print(f"¡Gráfico vectorial generado!")
print(f"Imagen exportada en: 'resultado_kmeans.png'")
print("El archivo se encuentra disponible en la raíz de su proyecto.")
print("==================================================")