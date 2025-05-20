#!/usr/bin/env dmd
import std;
/* ------------------------------------------ */
import DBrain.DBrain;
import secuencias: secuencias;
/* ------------------------------------------ */

void main () {


	float[] promedios = obtener_datos();



	Tuple!(float[], float[], float[]) muestras_divididas = divisor_datos(promedios);

	float[] muestras_entrenamiento = muestras_divididas[0];
	float[] muestras_prueba = muestras_divididas[1];
	float[] muestras_validacion = muestras_divididas[2];


	size_t caracteristicas_X_registro = 1;

	// 1. Datos y normalización
	float[][] secuencias;
	// Targets (siguiente número en la secuencia)
	float[][] objetivos;


	foreach (i; 0 .. muestras_entrenamiento.length - caracteristicas_X_registro) {
		// Secuencia: [i, i+1, ..., i+n-1]
		secuencias ~= muestras_entrenamiento[i .. i + caracteristicas_X_registro].dup;

		// Objetivo: el siguiente valor
		objetivos ~= [muestras_entrenamiento[i + caracteristicas_X_registro]];
	}
	//print(objetivos);

	//float min_val = 1.0f;  // El mínimo valor en la secuencia
	//float max_val = 21.0f; // El máximo valor en la secuencia

	//auto secuencia_norm = secuencia.map!(x => (x - min_val) / (max_val - min_val)).array;
	//auto objetivos_norm = objetivos.map!(x => (x - min_val) / (max_val - min_val)).array;
	//print("secuencia_norm",secuencia);
	//print("objetivos_norm",objetivos);


	//size_t tamanoEntrada, size_t[] arquitectura_red, float tasaAprendizaje = 0.01f, float tasaDecaimiento = 0.95f
	RedLSTM red = new RedLSTM(1, [3,2,1], tasa_aprendizaje : 1f, tasa_decaimiento : 0.01f);

	int max_epocas = 40;

	red.entrenar(secuencias, objetivos, max_epocas: max_epocas);
		
	red.actualizarPesos();

	float[][] datos_prueba;

	foreach (i; 0 .. muestras_prueba.length - caracteristicas_X_registro) {
		// Secuencia: [i, i+1, ..., i+n-1]
		datos_prueba ~= muestras_prueba[i .. i + caracteristicas_X_registro].dup;

	}


	print(datos_prueba);
	float[] resultados_predicciones;
	float[] predicciones_reales; 

	while (true) {
		string prueba;
		resultados_predicciones = red.predecir(datos_prueba, 6);
		//predicciones_reales = red.desnormalizarArray(resultados_predicciones, 1.0f, 21.0f);
		
		print("---------------------------------");
		print("resultados_predicciones: ", resultados_predicciones);
		//print("Predicciones desnormalizadas: ", predicciones_reales);
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
					datos_prueba ~= [to!float(siguiente)];
				} catch (Exception e) {
					print("Error: Ingresa un número válido");
					continue;
				}
			}
			red.entrenar(secuencias, objetivos, max_epocas: max_epocas);    
		} else if (prueba == "si") {
			break;
		} else {
			print("Opción no válida. Por favor responde 'si' o 'no'");
		}
	}
	// Le pasamos una lista de prueba y los pasos a predecir 


}






Tuple!(float[], float[], float[]) divisor_datos(float[] muestras_entrada) {
	// Calcular los tamaños
	int total = cast(int)muestras_entrada.length;
	int entrenamiento_len = cast(int)(total * 0.8);
	int prueba_len = cast(int)(total * 0.15);
	int validacion_len = total - entrenamiento_len - prueba_len;

	// Cortar las muestras
	float[] muestras_entrenamiento = muestras_entrada[0 .. entrenamiento_len];
	float[] muestras_prueba = muestras_entrada[entrenamiento_len .. entrenamiento_len + prueba_len];
	float[] muestras_validacion = muestras_entrada[entrenamiento_len + prueba_len .. $];

	return tuple(muestras_entrenamiento, muestras_prueba, muestras_validacion);
}



float[] obtener_datos () {

	string numero_fechas_por_grupo = "2";


	// 2. Convertir a formato string
	string secuencias_entrada = "[\n" ~
		secuencias
			.map!(fila => "    [" ~ fila.map!(s => `"` ~ s ~ `"`).join(", ") ~ "]")
			.join(",\n") ~
		"\n]";

	//JSONValue secuencias = parseJSON(secuencias);

	// Ejecutar el proceso Python
	auto secuencias_ordenadas = pipeProcess(
		["python", "prueba8.py", secuencias_entrada, numero_fechas_por_grupo],
		Redirect.stdout
	);

	string resultado;
	foreach (line; secuencias_ordenadas.stdout.byLine) resultado ~= line.idup;

	// Esperar la finalización del proceso
	auto result = wait(secuencias_ordenadas.pid);
	
	
	if (result != 0) {
		print("Error: El proceso Python falló con código " ~ result.to!string);
	}


	JSONValue resultado_json = parseJSON(resultado);


	string[] nombre_grupos;
	foreach (i, grupo; resultado_json["nombres_grupos"].array) {
		nombre_grupos ~= grupo.str;
	}
		
	float[] promedios;

	foreach (i, promedios_py; resultado_json["grupos"].object) {            
		promedios ~= redondear_string(to!string(promedios_py));
	}

	return promedios;

}
