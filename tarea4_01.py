import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


###Paso 1: Leer el archivo CSV
print("=======================================================================")
print("-----------Paso 1: Cargar los valores desde un CSV---------------------")
print("-----------------------------------------------------------------------")

df = pd.read_csv('titanik.csv')
print(df.head())
print()
print()

print("=======================================================================")
print("-----------Paso 2: Corregir los valores vacíos de la edad--------------")
print("-----------------------------------------------------------------------")


# Almacenar una copia del DataFrame antes de la corrección
df_before = df.copy()

# Calcular la media de las edades por género
mean_age_male = df[df['gender'] == 'male']['age'].mean()
mean_age_female = df[df['gender'] == 'female']['age'].mean()

# Rellenar los valores faltantes
df.loc[(df['age'].isnull()) & (df['gender'] == 'male'), 'age'] = mean_age_male
df.loc[(df['age'].isnull()) & (df['gender'] == 'female'), 'age'] = mean_age_female

# Verificar y mostrar el número de valores faltantes
missing_count = df['age'].isnull().sum()
if missing_count == 0:
    print("Todos los valores faltantes de la columna 'age' han sido corregidos.")
else:
    print(f"Aún hay {missing_count} valores faltantes en la columna 'age'.")

# Identificar y mostrar las filas que fueron modificadas
print("Estos son parte de los registros corregidos:")
modified_rows = df[df['age'] != df_before['age']]
print(modified_rows)
print()

###Paso 3: Calculos estadísticos
print("=======================================================================")
print("Paso 3: media, mediana, moda, rango, varianza y desv. estándar edades")
print("-----------------------------------------------------------------------")

# Estadísticas descriptivas de la columna 'age'
mean_age = df['age'].mean()
median_age = df['age'].median()
mode_age = df['age'].mode()[0]
range_age = df['age'].max() - df['age'].min()
variance_age = df['age'].var()
std_dev_age = df['age'].std()

print(f"Media: {mean_age}")
print(f"Mediana: {median_age}")
print(f"Moda: {mode_age}")
print(f"Rango: {range_age}")
print(f"Varianza: {variance_age}")
print(f"Desviación estándar: {std_dev_age}")

###Paso 4: Tasa de Supervivencia General
print("=======================================================================")
print("Paso 4: Tasa de Supervivencia General")
print("-----------------------------------------------------------------------")

survival_rate = df['survived'].mean()
print(f"Tasa de supervivencia general: {survival_rate * 100:.2f}%")

###Paso 5: Tasa de Supervivencia por Género
print("=======================================================================")
print("Paso 5: Tasa de Supervivencia por Género")
print("-----------------------------------------------------------------------")

survival_rate_male = df[df['gender'] == 'male']['survived'].mean()
survival_rate_female = df[df['gender'] == 'female']['survived'].mean()

print(f"Tasa de supervivencia hombres: {survival_rate_male * 100:.2f}%")
print(f"Tasa de supervivencia mujeres: {survival_rate_female * 100:.2f}%")

###Paso 6: Histograma de Edades por Clase
print("=======================================================================")
print("Paso 6: Histograma de Edades por Clase")
print("-----------------------------------------------------------------------")

# Función para calcular estadísticas descriptivas
def calcular_estadisticas(df, clase):
    clase_df = df[df['p_class'] == clase]['age']
    mean_age = clase_df.mean()
    median_age = clase_df.median()
    mode_age = clase_df.mode()[0] if not clase_df.mode().empty else 'N/A'
    range_age = clase_df.max() - clase_df.min()
    variance_age = clase_df.var()
    std_dev_age = clase_df.std()

    return mean_age, median_age, mode_age, range_age, variance_age, std_dev_age

# Crear histogramas por clase con estadísticas descriptivas
plt.figure(figsize=(15, 5))

colors = ['blue', 'green', 'red']
titles = ['Primera Clase', 'Segunda Clase', 'Tercera Clase']
for i, (clase, color, title) in enumerate(zip([1, 2, 3], colors, titles), start=1):
    plt.subplot(1, 3, i)
    sns.histplot(df[df['p_class'] == clase]['age'], bins=20, kde=True, color=color)
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.title(title)

    mean_age, median_age, mode_age, range_age, variance_age, std_dev_age = calcular_estadisticas(df, clase)

    # Mostrar estadísticas en la gráfica
    stats_text = (f"Media: {mean_age:.2f}\n"
                  f"Mediana: {median_age:.2f}\n"
                  f"Moda: {mode_age}\n"
                  f"Rango: {range_age:.2f}\n"
                  f"Varianza: {variance_age:.2f}\n"
                  f"Desv. Estándar: {std_dev_age:.2f}")
    plt.gca().text(0.95, 0.95, stats_text, ha='right', va='top', transform=plt.gca().transAxes,
                   bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()


###Paso 7: Diagramas de Cajas de Edades de Supervivientes y No Supervivientes
print("=======================================================================")
print("Paso 7: Diagrama de caja de edad de supervivientes y no supervivientes")
print("-----------------------------------------------------------------------")

# Crear un diagrama de cajas para las edades de los supervivientes y los no supervivientes
plt.figure(figsize=(12, 6))

# Diagrama de caja para los supervivientes
plt.subplot(1, 2, 1)
df[df['survived'] == 1]['age'].plot.box()
plt.title('Edades de los Supervivientes')
plt.ylabel('Edad')

# Diagrama de caja para los no supervivientes
plt.subplot(1, 2, 2)
df[df['survived'] == 0]['age'].plot.box()
plt.title('Edades de los No Supervivientes')
plt.ylabel('Edad')

plt.tight_layout()
plt.show()

print("=======================================================================")
print("=======================================================================")
print("=======================================================================")
print("-----------PARTE 2---------------------")
print("-----------------------------------------------------------------------")

print("=======================================================================")
print("Paso 1: Intervalo de Confianza para la Edad Promedio")
print("-----------------------------------------------------------------------")
import numpy as np
from scipy.stats import t

# Cargar los datos y calcular la edad promedio y el tamaño de la muestra
mean_age = df['age'].mean()
std_dev = df['age'].std()
n = len(df['age'].dropna())

# Calcular el error estándar de la media
stderr_mean = std_dev / np.sqrt(n)

# Definir el nivel de confianza (95%)
confidence_level = 0.95

# Calcular el valor crítico de t para el intervalo de confianza
t_critical = t.ppf((1 + confidence_level) / 2, df=n-1)

# Calcular el intervalo de confianza
lower_bound = mean_age - t_critical * stderr_mean
upper_bound = mean_age + t_critical * stderr_mean

print(f"Intervalo de confianza del 95% para la edad promedio: ({lower_bound}, {upper_bound})")

#Paso 2
print("=======================================================================")
print("Paso 2: Pruebas de Hipótesis de edad")
print("-----------------------------------------------------------------------")
from scipy.stats import ttest_1samp

# Filtrar datos por género y edad
women_age = df[(df['gender'] == 'female') & (df['age'].notnull())]['age']
men_age = df[(df['gender'] == 'male') & (df['age'].notnull())]['age']

# Prueba de una muestra para mujeres
t_stat_women, p_value_women = ttest_1samp(women_age, 56)
print(f"Para mujeres: t-statistic = {t_stat_women}, p-value = {p_value_women}")

# Prueba de una muestra para hombres
t_stat_men, p_value_men = ttest_1samp(men_age, 56)
print(f"Para hombres: t-statistic = {t_stat_men}, p-value = {p_value_men}")

print("=======================================================================")
print("Paso 3: Pruebas de supervivencia entre h y m")
print("-----------------------------------------------------------------------")
from scipy.stats import chi2_contingency

# Conteo de supervivencia por género
survival_counts = pd.crosstab(df['gender'], df['survived'])

# Prueba de independencia de Chi-cuadrado
chi2_gender, p_value_gender, _, _ = chi2_contingency(survival_counts)
print(f"Para diferencias de género: Chi-square statistic = {chi2_gender}, p-value = {p_value_gender}")

print("=======================================================================")
print("Paso 4: Diferencia en la Tasa de Supervivencia entre Clases")
print("-----------------------------------------------------------------------")

# Conteo de supervivencia por clase
class_survival_counts = pd.crosstab(df['p_class'], df['survived'])

# Prueba de independencia de Chi-cuadrado
chi2_class, p_value_class, _, _ = chi2_contingency(class_survival_counts)
print(f"Para diferencias de clase: Chi-square statistic = {chi2_class}, p-value = {p_value_class}")

print("=======================================================================")
print("Paso 5: Comparación de edad entre H y M")
print("-----------------------------------------------------------------------")

# Prueba de dos muestras para comparar la edad entre mujeres y hombres
from scipy.stats import ttest_ind

t_stat_gender, p_value_gender = ttest_ind(women_age, men_age)
print(f"t-statistic para comparación de edad entre mujeres y hombres: {t_stat_gender}, p-value: {p_value_gender}")

