#!/usr/bin/env dmd 
module DBrain.modelos.times_series_network;

//import std;
//import std.math : exp;
//import std.math.exponential: log;
//import std.algorithm : map, clamp, reduce, max;
//import std.array : array;
//import std.range : iota, chunks;
//import std.stdio;

///* ------------------ */
//// Importaciones locales
//import DBrain.funciones_activacion;
//import DBrain.herramientas.herramientas_network;
//import DBrain.modelos.network_predictor_gru;

///* ------------------ */

//alias print = writeln;

///* DATOS GLOBALES DE CONFIGURACION */

//float learningRate = 0.09;
//float discountFactor = 0.95;


//// Esto va a jalar de la biblioteca de funciones 
//// funciones.d
//float function (float x) funcion_activacion;
//float function (float x) funcion_gradiente;

///* ------------------------------------------------------- */


//class TimeSeriesNetwork {
//    // Usamos GRU para procesamiento de secuencias
//    GRUCell[] gruLayer;
    
//    // Capa de salida
//    Neuron[] outputLayer;
//    Synapse[][] outputSynapses;
    
//    int inputFeatures;
//    int hiddenSize;
//    int outputSize;
//    int sequenceLength;
    
//    this(int inputFeatures, int hiddenSize, int outputSize, int sequenceLength) {
//        this.inputFeatures = inputFeatures;
//        this.hiddenSize = hiddenSize;
//        this.outputSize = outputSize;
//        this.sequenceLength = sequenceLength;
        
//        // Inicializar capa GRU
//        gruLayer = new GRUCell[1]; // Usamos una sola celda GRU para toda la capa
//        gruLayer[0] = new GRUCell(inputFeatures, hiddenSize);
        
//        // Inicializar capa de salida para predicción
//        outputLayer = new Neuron[outputSize];
//        outputSynapses = new Synapse[][](hiddenSize, outputSize);
        
//        // Crear neuronas de salida
//        for (int i = 0; i < outputSize; i++) {
//            outputLayer[i] = new Neuron("Output " ~ i.to!string);
//        }
        
//        // Conectar capa GRU con capa de salida
//        import std.random : uniform;
        
//        for (int i = 0; i < hiddenSize; i++) {
//            for (int j = 0; j < outputSize; j++) {
//                float initialWeight = uniform(-0.1, 0.1);
//                // Crear neurona ficticia para representar la salida de GRU
//                Neuron gruOutput = new Neuron("GRU_Output_" ~ i.to!string);
//                outputSynapses[i][j] = new Synapse(initialWeight, gruOutput, outputLayer[j]);
//                outputLayer[j].backwardSynapses ~= outputSynapses[i][j];
//            }
//        }
//    }
    
//    // Predicción para serie temporal
//    float[] predict(float[][] inputSequence) {
//        // Reset estados internos
//        resetInternalStates();
        
//        // Procesar secuencia
//        float[] hiddenState;
//        foreach (input; inputSequence) {
//            hiddenState = processTimeStep(input);
//        }
        
//        // Predicción final basada en último estado
//        return computeOutput(hiddenState);
//    }
    
//    float[] processTimeStep(float[] input) {
//        // Procesar un paso temporal en la secuencia usando GRU
//        return gruLayer[0].forward(input);
//    }
    
//    float[] computeOutput(float[] hiddenState) {
//        // Calcular salida a partir del estado oculto
//        foreach (i, ref neuron; outputLayer) {
//            neuron.input = 0.0;
            
//            // Conectar la salida de GRU a la capa de salida
//            for (int h = 0; h < hiddenSize; h++) {
//                neuron.input += hiddenState[h] * outputSynapses[h][i].weight;
//            }
            
//            // Aplicar función de activación
//            neuron.output = funcion_activacion(neuron.input);
//            neuron.input = 0.0;
//        }
        
//        // Devolver las salidas
//        return outputLayer.map!(n => n.output).array;
//    }
    
//    void train(float[][] inputSequence, float[] expectedOutput, float learningRate) {
//        // Forward pass
//        float[] prediction = predict(inputSequence);
        
//        // Calcular error en la capa de salida
//        foreach (i, ref neuron; outputLayer) {
//            neuron.error = expectedOutput[i] - neuron.output;
//        }
        
//        // Backpropagation para la capa de salida
//        foreach (ref neuron; outputLayer) {
//            neuron.delta = neuron.error * funcion_gradiente(neuron.output);
            
//            // Actualizar pesos de la capa de salida
//            foreach (ref synapse; neuron.backwardSynapses) {
//                // En este caso, backwardNeuron es una neurona ficticia que representa la salida GRU
//                float gruOutput = synapse.backwardNeuron.name.startsWith("GRU_Output_") ? 
//                    gruLayer[0].output[synapse.backwardNeuron.name["GRU_Output_".length].to!int] : 0.0;
                
//                synapse.weight += gruOutput * neuron.delta * learningRate;
//                synapse.weight = clamp(synapse.weight, -20.0, 20.0);
//            }
//        }
        
//        // Backpropagation para la capa GRU (simplificado)
//        // Nota: Una implementación completa requeriría backpropagation a través del tiempo (BPTT)
//        // que es más compleja y requiere mantener un historial de estados y gradientes
        
//        // Podemos simplificar asumiendo que el gradiente se propaga hacia atrás en el tiempo
//        // y afecta solo al último estado GRU
//        float[] outputGradient = new float[hiddenSize];
//        foreach (i, ref neuron; outputLayer) {
//            for (int h = 0; h < hiddenSize; h++) {
//                outputGradient[h] += neuron.delta * outputSynapses[h][i].weight;
//            }
//        }
        
//        // Llamar al método backward de GRU (que habría que implementar completamente)
//        // gruLayer[0].backward(outputGradient, learningRate);
//    }
    
//    void trainBPTT(float[][][] batchInputSequences, float[][] batchExpectedOutputs, int epochs, float learningRate) {
//        // Implementación de Backpropagation Through Time (BPTT)
//        // Esta es una versión simplificada
        
//        for (int epoch = 0; epoch < epochs; epoch++) {
//            float totalError = 0.0;
            
//            for (int b = 0; b < batchInputSequences.length; b++) {
//                float[][] inputSequence = batchInputSequences[b];
//                float[] expectedOutput = batchExpectedOutputs[b];
                
//                // Forward pass
//                float[] prediction = predict(inputSequence);
                
//                // Calcular error
//                float error = 0.0;
//                foreach (i; 0 .. prediction.length) {
//                    float diff = expectedOutput[i] - prediction[i];
//                    error += diff * diff;
//                }
//                totalError += error;
                
//                // Backward pass
//                train(inputSequence, expectedOutput, learningRate);
//            }
            
//            import std.stdio : writeln;
//            if (epoch % 10 == 0) {
//                writeln("Epoch ", epoch, " - Error: ", totalError / batchInputSequences.length);
//            }
//        }
//    }
    
//}