# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Función para cargar los datos
def load_data():
    # Leer el archivo CSV desde el enlace proporcionado y eliminar la columna "Unnamed: 0"
    return pd.read_csv(
        "https://media.githubusercontent.com/media/fowardelcac/Tp2_sem/refs/heads/main/winemag-data-130k-v2.csv"
    ).drop("Unnamed: 0", axis=1)


# Función para generar gráficos de análisis exploratorio de datos (EDA)
def graphs(df: pd.DataFrame):
    # Sub-función para mostrar los principales países productores de vino
    def biggest_producers(df: pd.DataFrame):
        # Contar la cantidad de vinos por país y seleccionar los 15 principales
        pais_counts = df["country"].value_counts().head(15)

        # Crear gráfico de barras para los 15 principales países productores
        plt.figure(figsize=(10, 6))
        sns.barplot(y=pais_counts.index, x=pais_counts.values, hue=pais_counts.index)

        # Configurar títulos y etiquetas del gráfico
        plt.title(
            "Los 15 Principales Productores de Vino del Mundo.",
            fontsize=18,
            weight="bold",
        )
        plt.xlabel("Cantidad de Vinos", fontsize=16)
        plt.ylabel("País", fontsize=16)
        plt.xticks(rotation=45)

        # Mejorar el estilo visual del gráfico
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Mostrar el gráfico
        plt.show()

    # Sub-función para mostrar las principales provincias productoras de vino
    def biggest_provinces(df: pd.DataFrame):
        # Agrupar y contar por provincia y país, seleccionar las 15 principales
        province_counts = (
            df.groupby(["province", "country"])["province"]
            .count()
            .sort_values(ascending=False)
            .head(15)
            .reset_index(name="Count")
        )

        # Crear gráfico de barras coloreado por país
        plt.figure(figsize=(10, 6))
        sns.barplot(y="province", x="Count", hue="country", data=province_counts)

        # Personalizar el gráfico
        plt.title(
            "Las 15 Principales Provincias por Número de Vinos",
            fontsize=16,
            weight="bold",
        )
        plt.xlabel("Cantidad de Vinos", fontsize=12)
        plt.ylabel("Provincia", fontsize=12)
        plt.legend(title="País", fontsize=10, title_fontsize=12)

        # Mejorar el layout y mostrar el gráfico
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

    # Sub-función para mostrar la distribución de precios de los vinos
    def price_distribution(df: pd.DataFrame):
        # Filtrar precios menores o iguales a 65 para mejorar la visualización
        price_filter = df.loc[df.price <= 65]["price"]

        # Crear el gráfico de histograma para la distribución de precios
        plt.figure(figsize=(10, 6))
        sns.histplot(
            price_filter,
            kde=True,
            color="skyblue",
            bins=30,
            label="Distribución del precio (<= 65)",
        )

        # Configurar títulos y etiquetas del gráfico
        plt.title(
            "Distribución del Precio (Filtrando valores hasta el decil 0.9)",
            fontsize=16,
        )
        plt.xlabel("Precio", fontsize=14)
        plt.ylabel("Frecuencia", fontsize=14)

        # Mostrar leyenda y gráfico
        plt.legend()
        plt.show()

    # Sub-función para mostrar el diagrama de caja de los precios
    def price_boxplot(df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        filtered_data = df[df["price"] < 500]  # Excluir precios mayores a 500
        plt.title("Diagrama de caja del precio", fontsize=16)
        plt.xlabel("Precio", fontsize=14)
        sns.boxplot(x="price", data=filtered_data)
        plt.show()

    # Sub-función para mostrar la distribución de los puntos
    def points_distribution(df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.histplot(df.points, kde=True, color="skyblue", bins=30)

        # Configurar títulos y etiquetas del gráfico
        plt.title("Distribución de los puntos", fontsize=16)
        plt.xlabel("Puntos", fontsize=14)

        # Mostrar leyenda y gráfico
        plt.legend()
        plt.show()

    # Mostrar gráficos generados por las sub-funciones
    biggest_producers(df)
    biggest_provinces(df)

    # Mostrar los resultados de análisis específicos en lugar de solo imprimirlos como texto
    print("Las cinco regiones con mayor producción:")
    print(
        df.region_1.value_counts().to_frame().head()
    )  # Mostrar DataFrame en lugar de imprimir
    print("*" * 150)

    print("Variedades de uva más comunes: ")
    print(
        df.variety.value_counts().sort_values(ascending=False).to_frame().head(20)
    )  # Mostrar DataFrame
    print("*" * 150)

    print("Bodegas con mayor producción: ")
    print(
        df.winery.value_counts().sort_values(ascending=False).to_frame().head(20)
    )  # Mostrar DataFrame
    print("*" * 150)

    # Generar gráficos adicionales
    price_distribution(df)
    price_boxplot(df)

    print("Países con vinos más caros: ")
    print(
        df.groupby("country")["price"]
        .max()
        .sort_values(ascending=False)
        .to_frame()
        .head(15)
    )  # Mostrar DataFrame
    print("*" * 150)

    points_distribution(df)

    print("Países con vinos de mejor puntuación en promedio: ")
    print(
        df.groupby("country")["points"]
        .mean()
        .sort_values(ascending=False)
        .to_frame()
        .head(15)
    )  # Mostrar DataFrame
    print("*" * 150)


# Función para procesar y limpiar los datos
def process(df: pd.DataFrame):
    # Función para imputar valores faltantes de forma aleatoria
    def random_imputer(df_filter: pd.DataFrame, col: str):
        # Seleccionar valores no nulos de la columna para la imputación
        non_null_values = df_filter[col].dropna().values

        # Aplicar un valor aleatorio de non_null_values donde hay NaN en la columna
        df_filter.loc[:, col] = df_filter[col].apply(
            lambda x: np.random.choice(non_null_values) if pd.isnull(x) else x
        )
        return df_filter

    # Crear una copia del DataFrame original
    df = df.copy()
    df.drop_duplicates(inplace=True)  # Eliminar duplicados
    df_filter = df.dropna(
        subset=["country", "designation"]
    )  # Filtrar filas con valores nulos en las columnas clave
    df_filter = random_imputer(
        df_filter, "price"
    )  # Imputar valores faltantes en la columna "price"
    return df_filter


# Función principal para ejecutar el análisis completo
def main():
    # Cargar los datos desde la fuente
    df = load_data()

    # Mostrar información básica del DataFrame
    print("Valores nulos por columna:")
    print(df.isnull().sum())  # Mostrar la suma de valores nulos por columna
    print("*" * 150)

    print("Información general del dataset:")
    print(df.info())  # Mostrar información básica del DataFrame
    print("*" * 150)
    df_processed = process(df)
    # Realizar el análisis exploratorio de datos (EDA)
    graphs(df_processed)


# Ejecutar la función principal
main()
