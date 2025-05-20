//#!/usr/bin/env dmd
//module preparacion_datos;
//import std;
///* ---------------------------------- */
//import DBrain.herramientas.calculaciones.estadistica;
//import DBrain.herramientas.limpieza_datos;
//import DBrain.herramientas.obtener_datos;
//import DBrain.herramientas.app_herramientas;
///* ---------------------------------- */

//float[][][] division_entrenamiento(string archivo, float porcentajeEntrenamiento = 0.8) {

//	// Obtener datos del CSV
//	string[][] tablaOriginal = get_datos_csv(archivo);

//	// Filtrar los productos y estaciones
//	string[] productos = columna_filtrada(4, tablaOriginal);
//	string[] estaciones = columna_filtrada(3, tablaOriginal);

//	// Codificar y normalizar productos
//	list_datos_Z productos_bin = codificarYNormalizar(productos);
//	writeln("Productos: ", productos);
//	writeln("Productos binarios normalizados: ", get_productos_binarios(productos_bin));

//	// Codificar y normalizar estaciones
//	list_datos_Z estaciones_bin = codificarYNormalizar(estaciones);
//	writeln("Estaciones: ", estaciones);
//	writeln("Estaciones binarios normalizados: ", get_productos_binarios(estaciones_bin));

//	//Esta es la matriz en donde tendremos todos los datos organizados
//	float[][] datos_divisor;

//	// Procesar los datos originales
//	foreach (i, ref dato; tablaOriginal) {
//		float[] buffer;
//		// Nos saltamos el primer registro porque contiene los nombres de las columnas
//		if (i == 0) continue;

//		// Estaciones codificadas
//		int index_est = to!int(estaciones.countUntil(dato[3]));
//		float estacion_normalizada = get_productos_binarios(estaciones_bin)[index_est];
//		buffer ~= estacion_normalizada;
//		//write([estacion_normalizada]);

//		// Productos normalizados
//		int index_prod = to!int(productos.countUntil(dato[4]));
//		float producto_normalizado = get_productos_binarios(productos_bin)[index_prod];
//		buffer ~= producto_normalizado;
//		//write([producto_normalizado]);

//		// Normalizar las columnas numéricas
//		float[] columnaNormalizada = normalizarColumnaZScore([
//			to!float(dato[6]), // Cantidad
//			to!float(dato[7]), // Precio Unitario
//			to!float(dato[8])  // Total
//		]);
//		buffer ~= columnaNormalizada;
//		//write(columnaNormalizada);

//		// Nueva línea para la siguiente fila
//		//writeln();

//		datos_divisor ~= [buffer];
//	}
//	foreach(i, ref d; datos_divisor) {
//		print(d);
//	}
//	print();

//	// Dividir los datos (80% entrenamiento, 20% prueba)
//	auto entrenamiento_prueba = dividirDatos(datos_divisor, porcentajeEntrenamiento);

//	return entrenamiento_prueba;


//}


//float[][][] separador_x_y (float[][][] entrenamiento_prueba){
//	// Mostrar resultados
//	//writeln("Conjunto de entrenamiento: ", entrenamiento_prueba[0]);
//	//writeln("Conjunto de prueba: ", entrenamiento_prueba[1]);

//	float[][] X_entrenamiento = entrenamiento_prueba[0].map!(fila => fila[0 .. $-1]).array; // Características
//	float[][] y_entrenamiento = entrenamiento_prueba[0].map!(fila => [fila[$-1]]).array; // Etiquetas
	
//	float[][] X_prueba = entrenamiento_prueba[1].map!(fila => fila[0 .. $-1]).array; // Características
//	float[][] y_prueba = entrenamiento_prueba[1].map!(fila => [fila[$-1]]).array; // Etiquetas
	
//	return [X_entrenamiento , y_entrenamiento , X_prueba , y_prueba];


//}