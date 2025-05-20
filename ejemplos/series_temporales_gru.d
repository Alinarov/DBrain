//#!/usr/bin/env dmd
//module series_temporales_gru;
//import std;
//import DBrain.modelos.network_gru;
//import DBrain.funciones_activacion;
//import std.math : sin, PI;

//// Función para generar datos de series temporales (onda sinusoidal)
//float[][] generarSeriesTiempo(int numSecuencias, int longitudSecuencia, int tamanioCaracteristicas) {
//    float[][] series = new float[][](numSecuencias, tamanioCaracteristicas);
    
//    for (int i = 0; i < numSecuencias; i++) {
//        float t = i * 0.1;
//        series[i] = new float[tamanioCaracteristicas];
        
//        // Primera característica: seno
//        series[i][0] = sin(t);
        
//        // Segunda característica: seno desplazado
//        if (tamanioCaracteristicas > 1)
//            series[i][1] = sin(t + PI/4);
            
//        // Tercera característica: seno con frecuencia doble
//        if (tamanioCaracteristicas > 2)
//            series[i][2] = sin(2*t);
//    }
    
//    return series;
//}

//// Función para crear ventanas deslizantes de datos para entrenamiento
//auto crearVentanasSecuencia(float[][] series, int longitudSecuencia) {
//    int numSecuencias = cast(int)(series.length);
//    int numVentanas = numSecuencias - longitudSecuencia;
    
//    // Ventanas de entrada (X)
//    float[][][] X = new float[][][](numVentanas, longitudSecuencia, series[0].length);
//    // Valores objetivo (y) - el siguiente valor en la serie
//    float[][] y = new float[][](numVentanas, series[0].length);
    
//    for (int i = 0; i < numVentanas; i++) {
//        // Crear ventana deslizante
//        for (int j = 0; j < longitudSecuencia; j++) {
//            X[i][j] = series[i + j].dup;
//        }
//        // El valor objetivo es el siguiente después de la ventana
//        y[i] = series[i + longitudSecuencia].dup;
//    }
    
//    return tuple(X, y);
//}

//void main() {
//    // Configurar las funciones de activación globales
//    funcion_activacion_tanh = (float x) => tanh(x);
//    funcion_activacion_sigmoid = (float x) => 1.0 / (1.0 + exp(-x));
//    funcion_gradiente_tanh = (float x) => 1.0 - tanh(x) * tanh(x);
//    funcion_gradiente_sigmoid = (float x) {
//        float sigm = 1.0 / (1.0 + exp(-x));
//        return sigm * (1.0 - sigm);
//    };

//    // Parámetros
//    int longitudSecuencia = 10;
//    int tamanioCaracteristicas = 3;
//    int tamanioOculto = 8;
    
//    // Generar datos de series temporales
//    int totalDatos = 500;
//    auto seriesCompletas = generarSeriesTiempo(totalDatos, longitudSecuencia, tamanioCaracteristicas);
    
//    // Crear ventanas para entrenamiento
//    auto datos = crearVentanasSecuencia(seriesCompletas, longitudSecuencia);
//    auto X = datos[0];
//    auto y = datos[1];
    
//    writefln("Datos generados: %d secuencias de entrenamiento", X.length);
    
//    // Crear la red GRU
//    auto redGRU = new RedGRU(tamanioCaracteristicas, tamanioOculto, longitudSecuencia, tamanioCaracteristicas);
    
//    // Entrenamiento
//    writeln("Iniciando entrenamiento...");
//    int numEpocas = 100;
//    int reportarCada = 10;
    
//    for (int epoca = 0; epoca < numEpocas; epoca++) {
//        float errorPromedio = 0.0;
        
//        // Para cada ventana de datos
//        for (int i = 0; i < X.length; i++) {
//            // Propagar hacia adelante
//            float[] prediccion = redGRU.avanzar(X[i]);
            
//            // Calcular error
//            float errorSecuencia = 0.0;
//            foreach (j; 0..tamanioCaracteristicas) {
//                errorSecuencia += (prediccion[j] - y[i][j]) * (prediccion[j] - y[i][j]);
//            }
//            errorSecuencia /= tamanioCaracteristicas;
//            errorPromedio += errorSecuencia;
            
//            // Retropropagación
//            redGRU.aprender(y[i]);
//        }
        
//        errorPromedio /= X.length;
        
//        // Reportar progreso
//        if (epoca % reportarCada == 0 || epoca == numEpocas - 1) {
//            writefln("Época %d/%d: Error promedio = %.6f", 
//                     epoca + 1, numEpocas, errorPromedio);
//        }
//    }
    
//    // Evaluación: hacer predicciones a futuro
//    writeln("\nPredicciones a futuro:");
    
//    // Usar los últimos datos como punto de partida
//    float[][] ultimosConocidos = X[$ - 1].dup;
    
//    // Hacer varias predicciones consecutivas
//    int numPredicciones = 20;
    
//    for (int i = 0; i < numPredicciones; i++) {
//        // Predecir el siguiente valor
//        float[] siguientePrediccion = redGRU.avanzar(ultimosConocidos);
        
//        // Mostrar la predicción
//        writef("Predicción %2d: [", i + 1);
//        foreach (j; 0..tamanioCaracteristicas) {
//            writef(" %.4f", siguientePrediccion[j]);
//        }
//        writeln(" ]");
        
//        // Actualizar la ventana deslizante eliminando el primer elemento y agregando la predicción
//        for (int t = 0; t < longitudSecuencia - 1; t++) {
//            ultimosConocidos[t] = ultimosConocidos[t + 1].dup;
//        }
//        ultimosConocidos[longitudSecuencia - 1] = siguientePrediccion.dup;
//    }
    
//    // Guardar el modelo
//    string rutaModelo = "modelo_series_temporales.dat";
//    redGRU.guardarModelo(rutaModelo);
//    writefln("\nModelo guardado en: %s", rutaModelo);
//}