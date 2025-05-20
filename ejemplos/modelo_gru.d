//#!/usr/bin/env dmd
//module modelo_gru;
//import std;
//import std.random: uniform;
//import core.thread;
///* ------------------------- */
//// importaciones locales 
//import DBrain.herramientas.calculaciones.errores;
//import DBrain.herramientas.calculaciones.math;
//import DBrain.herramientas.herramientas_network;
//import DBrain.herramientas.app_herramientas;
//import DBrain.modelos.network_decision_q;
//import DBrain.funciones_activacion;
//import DBrain.modelos.network_predictor_gru;         // Nueva importación
//import preparacion_datos;
///* -------------------------- */
//alias print = writeln;

//// Renombramos para que no sea tan largo
//alias funcion_activacion = DBrain.modelos.network_decision_q.funcion_activacion;
//alias funcion_gradiente = DBrain.modelos.network_decision_q.funcion_gradiente;
//alias herramientas = DBrain.herramientas.herramientas_network;

//void main(string[] args) {
//    print("
// +*************************************+
//+* Predicción GRU+Q AI LEARNING :D    *+
// +*************************************+        ");

//    // Parámetros de la red
//    int inputFeatures = 4;         // Número de características de entrada
//    int hiddenSize = 16;           // Tamaño de la capa oculta GRU
//    int outputSize = 1;            // Tamaño de la salida (predicción)
//    int sequenceLength = 10;       // Longitud de la secuencia temporal
    
//    // Crear la red GRU+Q
//    auto gruQNetwork = new GRUQNetwork(inputFeatures, hiddenSize, outputSize, sequenceLength);
    
//    // Cargar datos
//    float[][][]entrenamiento_prueba = division_entrenamiento("ventas_1000.csv");
//    float[][][]datos_separados = separador_x_y(entrenamiento_prueba);

//    float[][] x_entrenamiento = datos_separados[0];
//    float[][] y_entrenamiento = datos_separados[1]; 
//    float[][] x_prueba = datos_separados[2];
//    float[][] y_prueba = datos_separados[3];
    
//    print();
//    print("X_entrenamiento ", x_entrenamiento);
//    print();
//    print("y_entrenamiento", y_entrenamiento);
//    print();
//    print("X_prueba", x_prueba);
//    print();
//    print("y_prueba", y_prueba);

//    // Intentar cargar pesos (si existen)
//    herramientas.cargarPesos("DBrain/experiencias/gru_q.memoria");

//    // Preparar secuencias para GRU
//    float[][][] secuencias_entrenamiento = [];
//    foreach (i; 0..x_entrenamiento.length-sequenceLength) {
//        float[][] secuencia = x_entrenamiento[i..i+sequenceLength];
//        secuencias_entrenamiento ~= secuencia;
//    }
    
//    // Entrenamiento supervisado (pre-entrenamiento)
//    print("Pre-entrenando el modelo GRU...");
//    foreach (epoch; 0..5000) {
//        float errorTotal = 0;
//        foreach (i, secuencia; secuencias_entrenamiento) {
//            if (i + sequenceLength < y_entrenamiento.length) {
//                float[] target = y_entrenamiento[i + sequenceLength - 1];
//                float[] prediction = gruQNetwork.process(secuencia);
                
//                // Calcular error
//                float error = (prediction[0] - target[0])^^2;
//                errorTotal += error;
                
//                // Entrenar con este ejemplo
//                gruQNetwork.trainStep(secuencia, target);
//            }
//        }
        
//        if (epoch % 100 == 0) {
//            print("Época ", epoch, " - Error: ", errorTotal / secuencias_entrenamiento.length);
//        }
//    }
    
//    // PROCESO DE PREDICCIÓN CON Q-LEARNING
//    print(" -------------------- ");
//    print("Prediciendo con Q-Learning");
//    int episode = 40;
//    float margen_error = 0.02; // Margen de error como porcentaje
//    float mape_calculado = 0;
//    float[][] predicciones;
//    int epocas_necesarias = 0;
    
//    // Preparar secuencias de prueba
//    float[][][] secuencias_prueba = [];
//    foreach (i; 0..x_prueba.length-sequenceLength+1) {
//        if (i + sequenceLength <= x_prueba.length) {
//            float[][] secuencia = x_prueba[i..i+sequenceLength];
//            secuencias_prueba ~= secuencia;
//        }
//    }
    
//    // Si no hay suficientes datos, usar lo que hay
//    if (secuencias_prueba.length == 0 && x_prueba.length > 0) {
//        // Rellenar con ceros si no hay suficientes datos
//        float[][] secuencia = new float[][](sequenceLength, inputFeatures);
//        foreach (i; 0..x_prueba.length) {
//            secuencia[sequenceLength - x_prueba.length + i] = x_prueba[i];
//        }
//        secuencias_prueba ~= secuencia;
//    }
    
//    foreach (i, ref secuencia; secuencias_prueba) {
//        float[] prediction;
//        float[] actual;
        
//        if (i < y_prueba.length) {
//            actual = y_prueba[i];
//        } else {
//            // Si no hay más datos reales, usar el último disponible
//            actual = y_prueba[$-1];
//        }
        
//        foreach (j; 0..episode) {
//            // Realizar predicción
//            prediction = gruQNetwork.predictWithExploration(secuencia);
            
//            // Calcular error porcentual absoluto
//            auto errores = calcular_mae(actual[0], prediction[0]);
//            float error_porcentual = errores[1];
            
//            // Actualizar MAPE calculado
//            mape_calculado = error_porcentual;
            
//            // Recompensa basada en error
//            if (error_porcentual >= margen_error) {
//                // Penalización para error alto
//                float reward = -1.0;
//                gruQNetwork.applyReward(reward);
//            } else {
//                // Recompensa para error bajo
//                print("La red eligió una predicción correcta.");
//                float reward = 0.55;
//                gruQNetwork.applyReward(reward);
//                epocas_necesarias += 1;
                
//                // Guardar predicción exitosa
//                predicciones ~= [prediction];
//                break;
//            }
            
//            // Si es el último intento, guardar la mejor predicción disponible
//            if (j == episode - 1) {
//                predicciones ~= [prediction];
//            }
//        }
//    }
    
//    // Mostrar resultados
//    print(" -------------------- ");
//    foreach (i, ref dato; predicciones) {
//        print("Predicción ", i, ": ", redondearDosDecimales(dato[0]));
//    }
//    print("Mape calculado: ", mape_calculado, "% con un margen de error: ", margen_error, "%");
//    print("Número de épocas necesarias: ", epocas_necesarias);
    
//    // Guardar modelo
//    herramientos.guardarPesos("DBrain/experiencias/gru_q.memoria");
//}