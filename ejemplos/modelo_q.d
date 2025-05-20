//#!/usr/bin/env dmd
//module modelo_q;
//import std;
//import std.random: uniform;
//import core.thread;
///* ------------------------- */
//// importaciones locales 
//import DBrain.DBrain;
//import preparacion_datos;
///* -------------------------- */
//alias print = writeln;

///**
// * Para entrenar el modelo debes de comentar las lineas asi: 
//	//herramientas.cargarPesos(network, "experiencias/xor.memoria");

//	//// Entrenar la red
//	//foreach (epoch; 0 .. 20000) {
//	//    foreach (i, input; inputs) {
//	//        network.think(input);
//	//        network.learn(targets[i]);
//	//    }
//	//}
// * 
// * y esta linea que esta al final: 
// * //herramientas.guardarPesos(network, "experiencias/xor.memoria");
// * */

//// Renombramos para que no sea tan largo
//alias funcion_activacion = DBrain.modelos.network_decision_q.funcion_activacion;
//alias funcion_gradiente = DBrain.modelos.network_decision_q.funcion_gradiente;
//alias herramientas = DBrain.herramientas.herramientas_network;

//void main (string [] args) {
//	print("
// +*****************************+
//+* Predicción Q AI LEARNING :D *+
// +*****************************+        ");

//	// entrada, n neuronas capas, salidas  
//	// 15, 32, 16, 1 - 3
//	Neuron[][] layers = createNeuralNetwork(4, [8,10], 1);

//	Network network = new Network_Q(layers);

//	funcion_activacion = &DBrain.funciones_activacion.sigmoid;
//	funcion_gradiente = &DBrain.funciones_activacion.tanh;


//	float[][][]entrenamiento_prueba = division_entrenamiento("ventas_1000.csv");

//	float[][][]datos_separados = separador_x_y(entrenamiento_prueba);

//	 float[][] x_entrenamiento = datos_separados[0];
//	 float[][] y_entrenamiento = datos_separados[1]; 
//	 float[][] x_prueba = datos_separados[2];
//	 float[][] y_prueba = datos_separados[3];
	
//	print();
//	print("X_entrenamiento ", x_entrenamiento);
//	print();
//	print("y_entrenamiento", y_entrenamiento);

//	print();
//	print("X_prueba", x_prueba);
//	print();
//	print("y_prueba", y_prueba);

//	// Datos de entrenamiento (ejemplo simple)
//	float[][] inputs = x_entrenamiento;
//	float[][] targets = y_entrenamiento;

//	herramientas.cargarPesos(network, "DBrain/experiencias/xor.memoria");

//	// Entrenar la red
//	print("Entrenando ");
//	foreach (epoch; 0 .. 20000) {
//	    foreach (i, input; inputs) {
//	        network.think(input);
//	        network.learn(targets[i]);
//	    }
//	}



	//// PROCESO DE PREDICCION
    //print(" -------------------- ");
	//print("Prediciendo");
	//float[] lastQvalues;
	//int episode = 40;
	//float[][] newInput = x_prueba;
	//float[] prediction;
	///* ------------------ */
	//int epocas_necesarias; // esto lo obtendremos una vez este entrenado y evaluado el modelo
	//float margen_error = 0.02; // Esto ya esta en porcentaje 
	//float mape_calculado = 0; // Esto va a cambiar con las epocas
    //float[][] predicciones; // Lista para almacenar todas las predicciones

    //foreach (i, ref x_entrada; newInput) {
    //    foreach (j; 0 .. episode) {
    //        prediction = network.think(x_entrada);

    //        lastQvalues = prediction.dup; // Guardamos los últimos Q-values

    //        // Calcular el error porcentual absoluto
    //        auto errores = calcular_mae(y_prueba[i][0], prediction[0]);
    //        float error_porcentual = errores[1];

    //        // Actualizar el MAPE calculado
    //        mape_calculado = error_porcentual;

    //        // Si el error es mayor o igual al margen de error
    //        if (error_porcentual >= margen_error) {
    //            float error = -1.0; // Penalización negativa
    //            prediction[0] += error;
    //        }
    //        // Si el error es menor o igual al margen de error
    //        else if (error_porcentual <= margen_error) {
    //            writeln("La red eligió una acción correcta.");
    //            float error = 0.55; // Recompensa
    //            prediction[0] += error;
    //            epocas_necesarias += 1; // Incrementar el número de épocas necesarias
    //            network.updateQValues(x_entrada, 0, prediction[0], x_entrada);

    //            // Guardar la predicción en la lista
    //            predicciones ~= [prediction[0]];
    //            break;
    //        }
    //    }
    //}


//    // Mostrar resultados
//    print(" -------------------- ");
//    foreach(i, ref dato; predicciones) {
//    	print("Predicciones: ", redondearDosDecimales(dato[0]));
//    };
//    print("Mape calculado: ", mape_calculado, "% con un margen de error: ", margen_error, "%");
//    print("Número de épocas necesarias: ", epocas_necesarias);


//	herramientas.guardarPesos(network, "DBrain/experiencias/xor.memoria");
//}
