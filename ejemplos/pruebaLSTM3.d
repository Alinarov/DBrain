#!/usr/bin/env dmd
import std;

import DBrain.DBrain;

void main () {


    // 1. Datos y normalización
	// Secuencias ascendentes (+1) y descendentes (-1) con variaciones
	float[] secuencia = [
	    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 
	    6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
	    11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
	    16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
	    21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
	    26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
	    31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
	    36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
	    41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
	    46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
	    51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
	    56.0f, 57.0f, 58.0f, 59.0f, 60.0f,
	    61.0f, 62.0f, 63.0f, 64.0f, 65.0f,
	    66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
	    71.0f, 72.0f, 73.0f, 74.0f, 75.0f,
	    76.0f, 77.0f, 78.0f, 79.0f, 80.0f,
	    81.0f, 82.0f, 83.0f, 84.0f, 85.0f,
	    86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
	    91.0f, 92.0f, 93.0f, 94.0f, 95.0f,
	    96.0f, 97.0f, 98.0f, 99.0f, 100.0f
	];

	// Targets (siguiente número en la secuencia)
	float[] objetivos = [
	    2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
	    7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
	    12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
	    17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 
	    22.0f, 23.0f, 24.0f, 25.0f,
	    26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
	    31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
	    36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
	    41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
	    46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
	    51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
	    56.0f, 57.0f, 58.0f, 59.0f, 60.0f,
	    61.0f, 62.0f, 63.0f, 64.0f, 65.0f,
	    66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
	    71.0f, 72.0f, 73.0f, 74.0f, 75.0f,
	    76.0f, 77.0f, 78.0f, 79.0f, 80.0f,
	    81.0f, 82.0f, 83.0f, 84.0f, 85.0f,
	    86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
	    91.0f, 92.0f, 93.0f, 94.0f, 95.0f,
	    96.0f, 97.0f, 98.0f, 99.0f, 100.0f, 101.0f
	];    
	float min_val = 1.0f;  // El mínimo valor en la secuencia
	float max_val = 21.0f; // El máximo valor en la secuencia

	auto secuencia_norm = secuencia.map!(x => (x - min_val) / (max_val - min_val)).array;
	auto objetivos_norm = objetivos.map!(x => (x - min_val) / (max_val - min_val)).array;
    print("secuencia_norm",secuencia);
    print("objetivos_norm",objetivos);


	//size_t tamanoEntrada, size_t[] arquitectura_red, float tasaAprendizaje = 0.01f, float tasaDecaimiento = 0.95f
    Network_LSTM red = new Network_LSTM(1, [1]);


    red.entrenar(secuencia, objetivos, maxEpochs: 1);

    float[] datos_prueba = [2,3,4,5,6];

    float[] resultados_predicciones;
	float[] predicciones_reales; 

	while (true) {
	    string prueba;
	    resultados_predicciones = red.predecir(datos_prueba, 1);
	    predicciones_reales = red.desnormalizarArray(resultados_predicciones, 1.0f, 21.0f);
	    
	    print("---------------------------------");
	    print("resultados_predicciones: ", resultados_predicciones);
	    print("Predicciones desnormalizadas: ", predicciones_reales);
	    print("---------------------------------");
	    print("objetivos: ", objetivos);
	    print("datos_prueba: ", datos_prueba);

	    print("¿Está correcto? (si/no)");
	    prueba = readln().strip().toLower();  // Limpia y convierte a minúsculas
	    
	    if (prueba == "no") {
	        print("¿Deseas agregar otro número? (si/no)");
	        string agregar = readln().strip().toLower();
	        
	        if (agregar == "si") {
	            print("Ingresa el número:");
	            try {
	                string siguiente = readln().strip();
	                datos_prueba ~= to!float(siguiente);
	            } catch (Exception e) {
	                print("Error: Ingresa un número válido");
	                continue;
	            }
	        }
	        red.entrenar(secuencia, objetivos, 10);    
	    } else if (prueba == "si") {
	        break;
	    } else {
	        print("Opción no válida. Por favor responde 'si' o 'no'");
	    }
	}
    // Le pasamos una lista de prueba y los pasos a predecir 


}
