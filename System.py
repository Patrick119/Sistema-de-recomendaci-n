import pandas as pd
import numpy as np
import time
from collections import defaultdict

def precalcular_matrices(data, movies):
    usuarios_items = defaultdict(dict)
    indices_invertidos = defaultdict(dict)

    for _, row in data.iterrows():
        usuario = row['userId']
        pelicula = row['movieId']
        rating = row['rating']

        usuarios_items[usuario][pelicula] = rating
        indices_invertidos[pelicula][usuario] = rating

    return usuarios_items, indices_invertidos, movies

def knn(usuario, tipo_distancia, usuarios_items, indices_invertidos, k):
    distancias = []
    peliculas_usuario = usuarios_items.get(usuario, {})

    if not peliculas_usuario:
        return distancias

    usuarios_comunes = set()

    for pelicula in peliculas_usuario:
        usuarios_comunes.update(indices_invertidos.get(pelicula, {}).keys())

    usuarios_comunes.discard(usuario)

    for vecino in usuarios_comunes:
        distancia = calcular_distancia(usuario, vecino, tipo_distancia, usuarios_items)
        if distancia is not None:
            distancias.append((vecino, distancia))

    return sorted(distancias, key=lambda x: x[1])[:k]

def calcular_distancia(usuario1, usuario2, tipo_distancia, usuarios_items):
    peliculas_usuario1 = usuarios_items.get(usuario1, {})
    peliculas_usuario2 = usuarios_items.get(usuario2, {})

    if not peliculas_usuario1 or not peliculas_usuario2:
        return None

    peliculas_comunes = set(peliculas_usuario1.keys()) & set(peliculas_usuario2.keys())

    if not peliculas_comunes:
        return None

    if tipo_distancia == 'manhattan':
        distancia = sum(abs(peliculas_usuario1[pelicula] - peliculas_usuario2[pelicula]) for pelicula in peliculas_comunes)
    elif tipo_distancia == 'euclidiana':
        distancia = np.sqrt(sum((peliculas_usuario1[pelicula] - peliculas_usuario2[pelicula]) ** 2 for pelicula in peliculas_comunes))
    elif tipo_distancia == 'pearson':
        distancia = pearson_correlation(usuario1, usuario2, peliculas_usuario1, peliculas_usuario2, peliculas_comunes)
    elif tipo_distancia == 'coseno':
        distancia = cosine_similarity(usuario1, usuario2, peliculas_usuario1, peliculas_usuario2, peliculas_comunes)

    return distancia

def pearson_correlation(usuario1, usuario2, peliculas_usuario1, peliculas_usuario2, peliculas_comunes):
    n = len(peliculas_comunes)
    if n == 0:
        return None

    sum_xy = sum(peliculas_usuario1[pelicula] * peliculas_usuario2[pelicula] for pelicula in peliculas_comunes)
    sum_x = sum(peliculas_usuario1[pelicula] for pelicula in peliculas_comunes)
    sum_y = sum(peliculas_usuario2[pelicula] for pelicula in peliculas_comunes)
    sum_x_sq = sum(peliculas_usuario1[pelicula] ** 2 for pelicula in peliculas_comunes)
    sum_y_sq = sum(peliculas_usuario2[pelicula] ** 2 for pelicula in peliculas_comunes)

    numerador = sum_xy - (sum_x * sum_y) / n
    denominador = np.sqrt((sum_x_sq - (sum_x ** 2) / n) * (sum_y_sq - (sum_y ** 2) / n))

    if denominador == 0:
        return None
    else:
        return numerador / denominador



def knnRecomendation(usuario, usuarios_items, indices_invertidos, k, umbral_similitud):
    distancias = []
    peliculas_usuario = usuarios_items.get(usuario, {})

    if not peliculas_usuario:
        return distancias

    usuarios_comunes = set()

    for pelicula in peliculas_usuario:
        usuarios_comunes.update(indices_invertidos.get(pelicula, {}).keys())

    usuarios_comunes.discard(usuario)

    for vecino in usuarios_comunes:
        distancia = cosine_similarity(usuario, vecino, peliculas_usuario, usuarios_items[vecino])
        if distancia is not None and distancia >= umbral_similitud:
            distancias.append((vecino, distancia))

    return sorted(distancias, key=lambda x: x[1], reverse=True)[:k]

def cosine_similarity(usuario1, usuario2, peliculas_usuario1, peliculas_usuario2, peliculas_comunes=None):
    if peliculas_comunes is None:
        numerador = sum(peliculas_usuario1[pelicula] * peliculas_usuario2.get(pelicula, 0) for pelicula in peliculas_usuario1)
        denominador = np.sqrt(sum(peliculas_usuario1[pelicula] ** 2 for pelicula in peliculas_usuario1)) * np.sqrt(sum(peliculas_usuario2[pelicula] ** 2 for pelicula in peliculas_usuario2))
    else:
        numerador = sum(peliculas_usuario1[pelicula] * peliculas_usuario2[pelicula] for pelicula in peliculas_comunes)
        denominador = np.sqrt(sum(peliculas_usuario1[pelicula] ** 2 for pelicula in peliculas_comunes)) * np.sqrt(sum(peliculas_usuario2[pelicula] ** 2 for pelicula in peliculas_comunes))

    if denominador == 0:
        return None
    else:
        return numerador / denominador


def recomendar_peliculas(usuario, usuarios_items, indices_invertidos, k, umbral_similitud, data, movies):
    print("-------------------------------------------------------------")
    print(f"\nRECOMENDACIÓN DE PELÍCULAS PARA EL USUARIO {usuario}\n")
    print("-------------------------------------------------------------")
    start_time = time.time()
    vecinos_cercanos = knnRecomendation(usuario, usuarios_items, indices_invertidos, k, umbral_similitud)

    if not vecinos_cercanos:
        print("No se encontraron vecinos cercanos para recomendar películas.")
        return

    print("Sus 10 vecinos más cercanos son:")
    for vecino, distancia in vecinos_cercanos:
        print(f"Usuario: {vecino}, Distancia: {distancia}")

    peliculas_recomendadas = defaultdict(list)
    for vecino, _ in vecinos_cercanos:
        peliculas_vecino = usuarios_items.get(vecino, {})
        for pelicula, rating in peliculas_vecino.items():
            if pelicula not in usuarios_items[usuario] and rating >= 3:  # No se ha visto la película y la calificación es al menos 3
                peliculas_recomendadas[pelicula].append(rating)

    if not peliculas_recomendadas:
        print("No se encontraron películas para recomendar.")
        return

    # Filtro de coincidencias exactas y positivas
    peliculas_coincidentes_exactas = defaultdict(int)
    peliculas_coincidentes_positivas = defaultdict(int)

    for pelicula, ratings in peliculas_recomendadas.items():
        for rating in ratings:
            if rating >= 4:  # Cambio de 5 a 4 estrellas para considerar como coincidencia positiva
                peliculas_coincidentes_positivas[pelicula] += 1

    # Filtro de popularidad
    popularidad_peliculas = data.groupby('movieId')['rating'].mean().sort_values(ascending=False)

    # Filtro de diversidad
    generos_vistos_vecinos = defaultdict(int)
    for pelicula_id, _ in peliculas_recomendadas.items():
        generos = movies.loc[movies['movieId'] == pelicula_id, 'genres'].iloc[0]
        for genero in generos.split('|'):
            generos_vistos_vecinos[genero] += 1

    # Ordenar los géneros por cantidad de películas vistas por los vecinos
    generos_ordenados_vecinos = sorted(generos_vistos_vecinos.items(), key=lambda x: x[1], reverse=True)

    print("\nGéneros más vistos por los vecinos:")
    for genero, cantidad in generos_ordenados_vecinos:
        print(f"Género: {genero}, Cantidad de películas: {cantidad}")

    # Géneros vistos por el usuario
    generos_vistos_usuario = defaultdict(int)
    for pelicula_id, rating in usuarios_items[usuario].items():
        generos = movies.loc[movies['movieId'] == pelicula_id, 'genres'].iloc[0]
        for genero in generos.split('|'):
            generos_vistos_usuario[genero] += 1

    # Ordenar los géneros por cantidad de películas vistas por el usuario
    generos_ordenados_usuario = sorted(generos_vistos_usuario.items(), key=lambda x: x[1], reverse=True)

    print("\nGéneros más vistos por el usuario:")
    for genero, cantidad in generos_ordenados_usuario:
        print(f"Género: {genero}, Cantidad de películas: {cantidad}")

    # Generar un ranking de películas recomendadas basado en la calificación promedio de los vecinos cercanos
    ranking_exactas_positivas = [(pelicula, peliculas_coincidentes_positivas[pelicula]) for pelicula in peliculas_coincidentes_positivas]
    ranking_exactas_positivas.sort(key=lambda x: x[1], reverse=True)

    ranking_popularidad = [(pelicula, popularidad_peliculas[pelicula]) for pelicula, _ in ranking_exactas_positivas]
    ranking_popularidad.sort(key=lambda x: x[1], reverse=True)

    ranking_diversidad = [(pelicula, diversidad_generos(pelicula, movies, generos_ordenados_vecinos, generos_vistos_usuario)) for pelicula, _ in ranking_exactas_positivas]
    ranking_diversidad.sort(key=lambda x: x[1], reverse=True)

    print("\nPelículas recomendadas basadas en coincidencias exactas y positivas:")
    for pelicula_id, _ in ranking_exactas_positivas[:2]:
        mostrar_info_pelicula(pelicula_id, movies)

    print("\nPelículas recomendadas basadas en popularidad:")
    for pelicula_id, _ in ranking_popularidad[:2]:
        mostrar_info_pelicula(pelicula_id, movies)

    print("\nPelículas recomendadas basadas en diversidad de géneros:")
    for pelicula_id, _ in ranking_diversidad[:2]:
        mostrar_info_pelicula(pelicula_id, movies)

    end_time = time.time()  # Tiempo de finalización
    elapsed_time = end_time - start_time  # Tiempo transcurrido
    print(f"\nTiempo de recomendación: {elapsed_time:.2f} segundos")

def diversidad_generos(pelicula_id, movies, generos_ordenados_vecinos, generos_vistos_usuario):
    generos_pelicula = movies.loc[movies['movieId'] == pelicula_id, 'genres'].iloc[0].split('|')
    diversidad = 0
    for i, (genero, _) in enumerate(generos_ordenados_vecinos):
        if genero in generos_pelicula:
            diversidad += (len(generos_ordenados_vecinos) - i) * generos_vistos_usuario.get(genero, 0)  # Ponderar por la cantidad de películas vistas por el usuario en ese género
    return diversidad

def mostrar_info_pelicula(pelicula_id, movies):
    pelicula_info = movies[movies['movieId'] == pelicula_id]
    titulo = pelicula_info['title'].values[0]
    generos = pelicula_info['genres'].values[0]
    print(f"Película: {titulo}, Géneros: {generos}")

def agregar_nuevos_usuarios(data):
    print("-------------------------------------------------------------")
    print("\nAGREGAR NUEVOS USUARIOS\n")
    print("-------------------------------------------------------------")

    nuevos_usuarios = []  # Aquí se almacenan las nuevas filas
    for i in range(2):  # Pedir dos usuarios
        while True:
            nuevo_usuario = input(f"Ingrese el ID de usuario {i+1}: ").strip()
            if not nuevo_usuario.isdigit() or int(nuevo_usuario) == 0:
                print("El ID de usuario no es válido. Debe ser un número entero mayor que cero. Intente nuevamente.")
                continue
            if int(nuevo_usuario) in data['userId'].unique():
                print("El ID de usuario ya existe. Intente nuevamente con un ID diferente.")
                continue
            break

        num_peliculas = 0
        while num_peliculas <= 0:
            num_peliculas = int(input(f"Ingrese cuántas películas ha visto el usuario {i+1}: ").strip())
            if num_peliculas <= 0:
                print("La cantidad de películas debe ser mayor que cero. Intente nuevamente.")

        nuevas_filas = []  # Aquí se almacenan las nuevas entradas para cada usuario
        for j in range(num_peliculas):
            while True:
                movie_id = input(f"Ingrese el ID de la película {j+1} para el usuario {i+1}: ").strip()
                if not movie_id.isdigit() or int(movie_id) == 0:
                    print("El ID de la película no es válido. Debe ser un número entero mayor que cero. Intente nuevamente.")
                    continue
                if int(movie_id) not in data['movieId'].unique():
                    print("El ID de la película no existe en la base de datos. Intente nuevamente.")
                    continue
                break
            while True:
                rating = input(f"Ingrese la calificación para la película {movie_id}: ").strip()
                try:
                    rating_float = float(rating)
                    if rating_float < 0 or rating_float > 5:
                        print("La calificación debe ser un número entre 0 y 5. Intente nuevamente.")
                        continue
                    break
                except ValueError:
                    print("La calificación debe ser un número decimal entre 0 y 5. Intente nuevamente.")
                    continue
            nuevas_filas.append({'userId': int(nuevo_usuario), 'movieId': int(movie_id), 'rating': rating_float, 'timestamp': None})

        nuevos_usuarios.extend(nuevas_filas)  # Se agregan las nuevas entradas a la lista de nuevas filas

    return nuevos_usuarios


def main():
    data = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    usuarios_items, indices_invertidos, movies = precalcular_matrices(data, movies)

    while True:
        print("-------------------------------------------------------------")
        print("\nBIENVENIDO A LA CALCULADORA DE DISTANCIA\n")
        print("-------------------------------------------------------------")
        print("\nSelecciona el tipo de cálculo:")
        print("1. Calcular distancia de Manhattan")
        print("2. Calcular distancia Euclidiana")
        print("3. Calcular correlación de Pearson)")
        print("4. Calcular similitud del coseno")
        print("5. Encontrar vecinos más cercanos a un usuario (KNN)")
        print("6. Agregar nuevos usuarios")
        print("7. Recomendar películas a un usuario")
        print("8. Salir")

        opcion = input("Ingresa el número de la opción deseada: ")

        if opcion in ['1', '2', '3', '4']:
            usuario1 = int(input("Ingrese el usuario 1: ").strip())
            usuario2 = int(input("Ingrese el usuario 2: ").strip())
            tipo_calculo = 'manhattan' if opcion == '1' else 'euclidiana' if opcion == '2' else 'pearson' if opcion == '3' else 'coseno'
            start_time = time.time()
            resultado = calcular_distancia(usuario1, usuario2, tipo_calculo, usuarios_items)
            end_time = time.time()
            if resultado is not None:
                print(resultado)
            else:
                print(f"No se puede calcular la distancia utilizando {tipo_calculo} para los usuarios especificados.")
            print(f"Tiempo de ejecución: {end_time - start_time} segundos")

        elif opcion == '5':
            user = int(input("Ingrese el usuario: ").strip())
            distancia = input("Ingrese el tipo de distancia (manhattan, euclidiana, pearson, coseno): ").strip().lower()
            k = int(input("Ingrese cuántos vecinos más cercanos desea obtener: "))
            start_time = time.time()
            print(f"\nLos {k} vecinos más cercanos a {user} son:")
            vecinos = knn(user, distancia, usuarios_items, indices_invertidos, k)
            for vecino, distancia in vecinos:
                print(f"{vecino}: {distancia}")
            end_time = time.time()
            print(f"Tiempo de ejecución: {end_time - start_time} segundos")

        elif opcion == '6':
            nuevas_filas = agregar_nuevos_usuarios(data)
            if nuevas_filas:
                if data.empty:
                    data = pd.DataFrame(nuevas_filas)
                else:
                    data = pd.concat([data, pd.DataFrame(nuevas_filas)], ignore_index=True)
                data.to_csv('ratings.csv', index=False)  # Guardar el DataFrame en el archivo CSV
                print("Usuarios agregados exitosamente.")
            else:
                print("No hay nuevas filas para agregar.")


        elif opcion == '7':
            usuario = int(input("Ingrese el ID de usuario al que desea recomendar películas: ").strip())
            recomendar_peliculas(usuario, usuarios_items, indices_invertidos, k=10, umbral_similitud=0.1, data=data, movies=movies)
            
        elif opcion == '8':
            print("¡Hasta luego!")
            break

if __name__ == "__main__":
    main()

