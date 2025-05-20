#!/usr/bin/env dmd
import std;
import DBrain.DBrain;

void main() {

    // 1. Datos y normalización
	// Secuencias ascendentes (+1) y descendentes (-1) con variaciones
	float[] secuencia = [
	    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 
	    6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
	    11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
	    16.0f, 17.0f, 18.0f, 19.0f, 20.0f
	];

	// Targets (siguiente número en la secuencia)
	float[] objetivos = [
	    2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
	    7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
	    12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
	    17.0f, 18.0f, 19.0f, 20.0f, 21.0f
	];    
	float min_val = 1.0f;  // El mínimo valor en la secuencia
	float max_val = 21.0f; // El máximo valor en la secuencia

	auto secuencia_norm = secuencia.map!(x => (x - min_val) / (max_val - min_val)).array;
	auto objetivos_norm = objetivos.map!(x => (x - min_val) / (max_val - min_val)).array;
    print("secuencia_norm",secuencia_norm);
    print("objetivos_norm",objetivos_norm);

    // 2. Inicialización LSTM con Xavier
    auto lstm = new LSTM_celda(
        pesos: [xavierInit(to!float(secuencia_norm.length), to!float(objetivos.length)), xavierInit(to!float(secuencia_norm.length), to!float(objetivos.length)), xavierInit(to!float(secuencia_norm.length), to!float(objetivos.length)), xavierInit(to!float(secuencia_norm.length), to!float(objetivos.length))],
        valores_ponderados: [xavierInit(to!float(secuencia_norm.length), to!float(objetivos.length)), xavierInit(to!float(secuencia_norm.length), to!float(objetivos.length)), xavierInit(to!float(secuencia_norm.length), to!float(objetivos.length)), xavierInit(to!float(secuencia_norm.length), to!float(objetivos.length))]
    );

	// Definimos las funciones de activacion y de ipervolica o gradiente
	lstm.funcion_activacion = &DBrain.funciones_activacion.sigmoid;
	lstm.funcion_hiperbolica = &DBrain.funciones_activacion.tanh_hiperbolica;
	lstm.funcion_derivada = &DBrain.funciones_activacion.tanh_derivada;


    //lstm.guardarParametros(lstm, "DBrain/experiencias/lstm.json");
    lstm.cargarParametros(lstm, "DBrain/experiencias/lstm.json");


	float mejor_error = float.max;
	int sin_mejoria = 0;

	void entreno() {
		print("entrenando..");
		foreach (epoch; 0..10000) {
		    float error = lstm.entrenar(lstm, secuencia_norm, objetivos_norm);
		    
		    if (error < mejor_error) {
		        mejor_error = error;
		        sin_mejoria = 0;
		    } else {
		        sin_mejoria++;
		        if (sin_mejoria > 500) break; // Detener si no mejora
		    }
		}

	}

	entreno();

	void predecir(LSTM_celda lstm, float ultimo_valor, int pasos) {
	    // "Calentar" la LSTM con la secuencia completa
	    float estado = 0.0f;
	    foreach (x; secuencia_norm) {
	        estado = lstm.fordward(x, estado);
	    }
	    
	    // Predicciones
	    foreach (i; 0..pasos) {
	        estado = lstm.fordward(ultimo_valor, estado);
	        float prediccion = round(estado) * 10.0f; // Desnormalizar
	        writeln("Predicción ", i+1, ": ", prediccion);
	        ultimo_valor = estado; // Usar la predicción como entrada
	    }
	}
	

	float ultimo_valor = 3.0f; // valor a iniciar la prediccion
	int pasos = 100; // numero de epocas futuras 
	bool prediccion_correcta = false; // control de errores
	int intentos_maximos = 10; // Para evitar bucles infinitos
	int intentos = 0; 

	while (!prediccion_correcta && intentos < intentos_maximos) {
	    // 1. Realizar predicciones
	    print("Prediciendo: ", ultimo_valor, " | pasos: ", pasos);
	    float[] predicciones;
	    float estado = 0.0f;
	    
	    float entrada_actual = ultimo_valor / 10.0f; // Normalizar
	    // probando
	    foreach (i; 0..pasos) {

	        estado = lstm.fordward(entrada_actual, estado); // Operamos 
	        // Desnormalizamos el resultado
	        float prediccion = estado * 10.0f; // Desnormalizar
	        predicciones ~= prediccion;
	        entrada_actual = estado;
	        print("Predicción ", i+1, ": ", prediccion);
	    }

	    // 2. Verificar si las predicciones son correctas (ej: secuencia creciente)
	    prediccion_correcta = true;
	    for (int i = 1; i < predicciones.length; ++i) {
	        if (predicciones[i] <= predicciones[i-1]) { // Deben ser crecientes
	            prediccion_correcta = false;
	            break;
	        }
	    }

	    // 3. Si no es correcta, entrenar más
	    if (!prediccion_correcta) {
	        writeln("Entrenando adicionalmente...");
	        sin_mejoria = 0;
	        mejor_error = float.max;
	        
	        foreach (epoch; 0..5000) { // Epocas adicionales
	            float error = lstm.entrenar(lstm, secuencia_norm, objetivos_norm);
	            
	            if (error < mejor_error) {
	                mejor_error = error;
	                sin_mejoria = 0;
	            } else {
	                sin_mejoria++;
	                if (sin_mejoria > 300) break;
	            }
	        }
	        intentos++;
	    }
	}

	if (prediccion_correcta) {
	    writeln("¡Predicciones correctas alcanzadas!");
	} else {
	    writeln("No se lograron predicciones correctas después de ", intentos_maximos, " intentos.");
		print("¿Deseas reentrenar los pesos? (si/no): ");
		string reentreno = to!string(readln().strip());

	    if (reentreno == "si") {
			entreno();
		}
	}


	// --- Predicción más allá del 7 ---
	print("¿Deseas guardar los pesos? (si/no): ");
	string guardar = to!string(readln().strip());

	if (guardar == "si") {
	    lstm.guardarParametros(lstm, "DBrain/experiencias/lstm.json");
	} else {
		print("¿Deseas reentrenar los pesos? (si/no): ");
		string reentreno = to!string(readln().strip());
		if (reentreno == "si") {
			entreno();
		}

	}
}


