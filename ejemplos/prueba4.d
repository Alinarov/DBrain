//#!/usr/bin/env dmd
//import std;
//import std.algorithm : map;
//import std.array : array;
//import DBrain.herramientas.calculaciones.estadistica;
//import DBrain.herramientas.obtener_datos;
//import std.random : Random, uniform;
//import std.range : take, drop;
//import std.stdio : writeln;

//// Función para dividir los datos en entrenamiento y prueba
//float[][][] dividirDatos(float[][] datos, float porcentajeEntrenamiento) {
//	Random rnd;
//	rnd.seed(42); // Semilla para reproducibilidad

//	// Mezclar los datos
//	auto datosMezclados = datos.dup;
//	for (ulong i = datosMezclados.length - 1; i > 0; i--) {
		
//		int x = to!int(i);
//		int j = uniform(0, x + 1, rnd);
//		auto temp = datosMezclados[x];
//		datosMezclados[x] = datosMezclados[j];
//		datosMezclados[j] = temp;
//	}

//	// Dividir los datos
//	int indiceDivision = cast(int)(datosMezclados.length * porcentajeEntrenamiento);
//	float[][] entrenamiento = datosMezclados[0 .. indiceDivision];
//	float[][] prueba = datosMezclados[indiceDivision .. $];

//	return [entrenamiento, prueba];
//}

//void main() {
//	// Datos de ejemplo
//	float[][] datos = [
//		[27, 4, 2023, 0, 0, 3, 1461.85, 4385.55],
//		[8, 4, 2023, 0, 0, 2, 1263.27, 2526.54],
//		[11, 9, 2023, 2, 1, 3, 176.69, 530.08],
//		[18, 5, 2023, 1, 2, 5, 189.78, 948.90],
//		[16, 2, 2023, 5, 6, 5, 59.71, 48.30],
//		[25, 12, 2023, 3, 3, 5, 392.89, 1964.46]
//	];

//	// Dividir los datos (80% entrenamiento, 20% prueba)
//	auto entrenamiento_prueba = dividirDatos(datos, 0.8);



//	// Mostrar resultados
//	writeln("Conjunto de entrenamiento: ", entrenamiento_prueba[0]);
//	writeln("Conjunto de prueba: ", entrenamiento_prueba[1]);


//	float[][] X_entrenamiento = entrenamiento_prueba[0].map!(fila => fila[0 .. $-1]).array; // Características
//	float[][] y_entrenamiento = entrenamiento_prueba[0].map!(fila => [fila[$-1]]).array; // Etiquetas
//	print();
//	print(X_entrenamiento);
//	print();
//	print(y_entrenamiento);

//	float[][] X_prueba = entrenamiento_prueba[1].map!(fila => fila[0 .. $-1]).array; // Características
//	float[][] y_prueba = entrenamiento_prueba[1].map!(fila => [fila[$-1]]).array; // Etiquetas
//	print();
//	print(X_prueba);
//	print();
//	print(y_prueba);

//	/*
//	// Entrenar la red neuronal
//	auto redNeuronal = new RedNeuronal([4, 10, 1]); // Ejemplo: 4 entradas, 10 neuronas ocultas, 1 salida
//	redNeuronal.entrenar(X_entrenamiento, y_entrenamiento, 1000); // 1000 épocas

//	// Evaluar la red neuronal
//	float error = redNeuronal.evaluar(X_prueba, y_prueba);
//	writeln("Error en el conjunto de prueba: ", error); 


//	source > rdmd prueba4.d 
//	Conjunto de entrenamiento: [[16, 2, 2023, 5, 6, 5, 59.71, 48.3], [18, 5, 2023, 1, 2, 5, 189.78, 948.9], [8, 4, 2023, 0, 0, 2, 1263.27, 2526.54], [25, 12, 2023, 3, 3, 5, 392.89, 1964.46]]
//	Conjunto de prueba: [[11, 9, 2023, 2, 1, 3, 176.69, 530.08], [27, 4, 2023, 0, 0, 3, 1461.85, 4385.55]]
//	[[16, 2, 2023, 5, 6, 5, 59.71], [18, 5, 2023, 1, 2, 5, 189.78], [8, 4, 2023, 0, 0, 2, 1263.27], [25, 12, 2023, 3, 3, 5, 392.89]]

//	[[48.3], [948.9], [2526.54], [1964.46]]
//	 source > rdmd prueba4.d 
//	Conjunto de entrenamiento: [[16, 2, 2023, 5, 6, 5, 59.71, 48.3], [18, 5, 2023, 1, 2, 5, 189.78, 948.9], [8, 4, 2023, 0, 0, 2, 1263.27, 2526.54], [25, 12, 2023, 3, 3, 5, 392.89, 1964.46]]
//	Conjunto de prueba: [[11, 9, 2023, 2, 1, 3, 176.69, 530.08], [27, 4, 2023, 0, 0, 3, 1461.85, 4385.55]]

//	[[16, 2, 2023, 5, 6, 5, 59.71], [18, 5, 2023, 1, 2, 5, 189.78], [8, 4, 2023, 0, 0, 2, 1263.27], [25, 12, 2023, 3, 3, 5, 392.89]]

//	[[48.3], [948.9], [2526.54], [1964.46]]
//	 source > rdmd prueba4.d 
//	Conjunto de entrenamiento: [[16, 2, 2023, 5, 6, 5, 59.71, 48.3], [18, 5, 2023, 1, 2, 5, 189.78, 948.9], [8, 4, 2023, 0, 0, 2, 1263.27, 2526.54], [25, 12, 2023, 3, 3, 5, 392.89, 1964.46]]
//	Conjunto de prueba: [[11, 9, 2023, 2, 1, 3, 176.69, 530.08], [27, 4, 2023, 0, 0, 3, 1461.85, 4385.55]]

//	[[16, 2, 2023, 5, 6, 5, 59.71], [18, 5, 2023, 1, 2, 5, 189.78], [8, 4, 2023, 0, 0, 2, 1263.27], [25, 12, 2023, 3, 3, 5, 392.89]]

//	[[48.3], [948.9], [2526.54], [1964.46]]
//	 source > rdmd prueba4.d 
//	Conjunto de entrenamiento: [[16, 2, 2023, 5, 6, 5, 59.71, 48.3], [18, 5, 2023, 1, 2, 5, 189.78, 948.9], [8, 4, 2023, 0, 0, 2, 1263.27, 2526.54], [25, 12, 2023, 3, 3, 5, 392.89, 1964.46]]
//	Conjunto de prueba: [[11, 9, 2023, 2, 1, 3, 176.69, 530.08], [27, 4, 2023, 0, 0, 3, 1461.85, 4385.55]]

//	[[16, 2, 2023, 5, 6, 5, 59.71], [18, 5, 2023, 1, 2, 5, 189.78], [8, 4, 2023, 0, 0, 2, 1263.27], [25, 12, 2023, 3, 3, 5, 392.89]]

//	[[48.3], [948.9], [2526.54], [1964.46]]

//	[[11, 9, 2023, 2, 1, 3, 176.69], [27, 4, 2023, 0, 0, 3, 1461.85]]

//	[[530.08], [4385.55]]
//	 source > 

//	*/

//}