#!/usr/bin/env python
import json
import sys

# Definir en cuántas fechas se agruparán los grupos
num_fechas_por_grupo = 2  # Puedes cambiar este valor según la cantidad de fechas por grupo

# Datos de ejemplo
datos = []
#     [1, 1, "2025-05-18", 1],
#     [1, 1, "2025-05-19", 2],
#     [1, 1, "2025-05-20", 3],
#     [2, 1, "2025-05-21", 9],
#     [3, 1, "2025-05-18", 11],
#     [3, 1, "2025-05-19", 12],
#     [3, 1, "2025-05-20", 13],
#     [4, 1, "2025-05-18", 16],
#     [4, 1, "2025-05-19", 17],
#     [4, 1, "2025-05-20", 18],
#     [5, 1, "2025-05-18", 21],
#     [5, 1, "2025-05-19", 22],
#     [5, 1, "2025-05-20", 23],
#     [10, 1, "2025-05-18", 46],
#     [10, 1, "2025-05-19", 47],
#     [10, 1, "2025-05-20", 48],
#     [16, 1, "2025-05-20", 78],
#     [20, 1, "2025-05-18", 96],
#     [20, 1, "2025-05-19", 97],
#     [20, 1, "2025-05-20", 98]
# ]

entrada = sys.argv[1]
num_fechas_por_grupo = int(sys.argv[2])
datos = json.loads(entrada)




# Agrupar registros por fecha
registros_por_fecha = {}

for registro in datos:
    fecha = registro[2]
    if fecha not in registros_por_fecha:
        registros_por_fecha[fecha] = []
    registros_por_fecha[fecha].append(registro)

# Obtener las fechas ordenadas
fechas_ordenadas = sorted(registros_por_fecha.keys())

# Determinar el número total de grupos
total_grupos = len(fechas_ordenadas) // num_fechas_por_grupo + (1 if len(fechas_ordenadas) % num_fechas_por_grupo != 0 else 0)

# Agrupar fechas en bloques de tamaño num_fechas_por_grupo y calcular resultados
grupos = {}
nombres_grupos = []

for i in range(0, len(fechas_ordenadas), num_fechas_por_grupo):
    grupo_key = f"Grupo_{i // num_fechas_por_grupo + 1}"  # Nombre del grupo
    nombres_grupos.append(grupo_key)  # Guardar el nombre del grupo
    total_ventas = 0
    sumatoria_rendimientos = 0

    for fecha in fechas_ordenadas[i:i + num_fechas_por_grupo]:
        for registro in registros_por_fecha[fecha]:
            rendimiento = float(registro[1]) * float(registro[3])  # Multiplicar cantidad * precio
            sumatoria_rendimientos += rendimiento
            total_ventas += 1  # Contar cada venta

    # Calcular el resultado dividido y luego dividirlo por el número total de grupos
    resultado_dividido = round(sumatoria_rendimientos / total_ventas if total_ventas > 0 else 0, 1)
    resultado_final = round(resultado_dividido / total_grupos if total_grupos > 0 else 0, 1)

    grupos[grupo_key] = resultado_final  # Guardar el resultado final en cada grupo

# Estructurar el JSON final con los grupos y los nombres de los grupos
json_data = {
    "grupos": grupos,
    "nombres_grupos": nombres_grupos
}

# Convertir a formato JSON y mostrar
json_resultado = json.dumps(json_data, indent=4)

print(json_resultado)
