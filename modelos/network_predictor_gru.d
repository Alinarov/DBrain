#!/usr/bin/env dmd
module DBrain.modelos.modelo_gru_q;
import std;
import std.random: uniform;
import core.thread;
/* ------------------------- */
// importaciones locales 
import DBrain.herramientas.calculaciones.errores;
import DBrain.herramientas.calculaciones.math;
import DBrain.herramientas.herramientas_network;
import DBrain.herramientas.app_herramientas;
import DBrain.funciones_activacion;
import preparacion_datos;
/* -------------------------- */
alias print = writeln;

// Renombramos para que no sea tan largo
alias funcion_activacion = DBrain.funciones_activacion.sigmoid;
alias funcion_gradiente = DBrain.funciones_activacion.sigmoid_gradient;
alias herramientas = DBrain.herramientas.herramientas_network;

// Clase para combinar GRU con Q-learning
class GRUQNetwork {
    // Red GRU para procesar secuencias
    TimeSeriesNetwork tsNetwork;
    
    // Parámetros de Q-learning
    float learningRate = 0.09;
    float discountFactor = 0.95;
    float explorationRate = 0.1;
    
    // Historial para backward pass
    float[][] inputHistory;
    float[] outputHistory;
    
    this(int inputFeatures, int hiddenSize, int outputSize, int sequenceLength) {
        // Inicializar la red temporal
        tsNetwork = new TimeSeriesNetwork(inputFeatures, hiddenSize, outputSize, sequenceLength);
        
        // Inicializar historiales
        inputHistory = [];
        outputHistory = [];
        
        // Configurar funciones de activación
        DBrain.modelos.times_series_network.funcion_activacion = &DBrain.funciones_activacion.sigmoid;
        DBrain.modelos.times_series_network.funcion_gradiente = &DBrain.funciones_activacion.sigmoid_gradient;
    }
    
    // Procesar una secuencia completa y obtener Q-values
    float[] process(float[][] inputSequence) {
        return tsNetwork.predict(inputSequence);
    }
    
    // Predecir usando la red con exploración epsilon-greedy
    float[] predictWithExploration(float[][] inputSequence) {
        float[] qValues = process(inputSequence);
        
        // Almacenar para el aprendizaje
        inputHistory ~= inputSequence[$-1];  // Guardamos último estado
        outputHistory ~= qValues;
        
        // Exploración epsilon-greedy (para casos de acciones discretas)
        // Para predicción numérica, podemos añadir un ruido gaussiano
        if (uniform(0.0, 1.0) < explorationRate) {
            foreach (ref q; qValues) {
                q += uniform(-0.1, 0.1);  // Añadir ruido aleatorio
            }
        }
        
        return qValues;
    }
    
    // Actualizar Q-values usando recompensa y siguiente estado
    void updateQValues(float[][] state, float reward, float[][] nextState) {
        // Obtener Q-values actuales y siguientes
        float[] currentQValues = process(state);
        float[] nextQValues = process(nextState);
        
        // Calcular target Q-value usando la ecuación de Bellman
        float maxNextQ = nextQValues.reduce!max;
        float targetQ = reward + discountFactor * maxNextQ;
        
        // Calcular error TD (Temporal Difference)
        float tdError = targetQ - currentQValues[0];
        
        // Crear un objetivo para el entrenamiento
        float[] target = currentQValues.dup;
        target[0] += learningRate * tdError;
        
        // Entrenar la red con este estado y target
        trainStep(state, target);
    }
    
    // Actualizar Q-values para un conjunto de datos
    void applyReward(float reward) {
        if (outputHistory.length == 0) return;
        
        // Obtener el último output
        float[] lastOutput = outputHistory[$-1].dup;
        float[] lastInput = inputHistory[$-1].dup;
        
        // Aplicar recompensa al último output
        lastOutput[0] += learningRate * reward;
        
        // Usar el input y output actualizados para entrenar
        float[][] sequence = [lastInput];
        trainStep(sequence, lastOutput);
        
        // Limpiar historia después de aplicar recompensa
        inputHistory = [];
        outputHistory = [];
    }
    
    // Realizar un paso de entrenamiento
    void trainStep(float[][] inputSequence, float[] target) {
        // Forward pass
        float[] prediction = tsNetwork.predict(inputSequence);
        
        // Calcular error
        float[] error = new float[target.length];
        foreach (i; 0..target.length) {
            error[i] = target[i] - prediction[i];
        }
        
        // Entrenar la red temporal con estos datos
        tsNetwork.train(inputSequence, target, learningRate);
    }
}
class TimeSeriesNetwork {
    // Usamos GRU para procesamiento de secuencias
    GRUCell[] gruLayer;
    
    // Capa de salida
    Neuron[] outputLayer;
    Synapse[][] outputSynapses;
    
    int inputFeatures;
    int hiddenSize;
    int outputSize;
    int sequenceLength;
    
    this(int inputFeatures, int hiddenSize, int outputSize, int sequenceLength) {
        this.inputFeatures = inputFeatures;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.sequenceLength = sequenceLength;
        
        // Inicializar capa GRU
        gruLayer = new GRUCell[1]; // Usamos una sola celda GRU para toda la capa
        gruLayer[0] = new GRUCell(inputFeatures, hiddenSize);
        
        // Inicializar capa de salida para predicción
        outputLayer = new Neuron[outputSize];
        outputSynapses = new Synapse[][](hiddenSize, outputSize);
        
        // Crear neuronas de salida
        for (int i = 0; i < outputSize; i++) {
            outputLayer[i] = new Neuron("Output " ~ i.to!string);
        }
        
        // Conectar capa GRU con capa de salida
        import std.random : uniform;
        
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                float initialWeight = uniform(-0.1, 0.1);
                // Crear neurona ficticia para representar la salida de GRU
                Neuron gruOutput = new Neuron("GRU_Output_" ~ i.to!string);
                outputSynapses[i][j] = new Synapse(initialWeight, gruOutput, outputLayer[j]);
                outputLayer[j].backwardSynapses ~= outputSynapses[i][j];
            }
        }
    }
    
    // Predicción para serie temporal
    float[] predict(float[][] inputSequence) {
        // Reset estados internos
        resetInternalStates();
        
        // Procesar secuencia
        float[] hiddenState;
        foreach (input; inputSequence) {
            hiddenState = processTimeStep(input);
        }
        
        // Predicción final basada en último estado
        return computeOutput(hiddenState);
    }
    
    float[] processTimeStep(float[] input) {
        // Procesar un paso temporal en la secuencia usando GRU
        return gruLayer[0].forward(input);
    }
    
    float[] computeOutput(float[] hiddenState) {
        // Calcular salida a partir del estado oculto
        foreach (i, ref neuron; outputLayer) {
            neuron.input = 0.0;
            
            // Conectar la salida de GRU a la capa de salida
            for (int h = 0; h < hiddenSize; h++) {
                neuron.input += hiddenState[h] * outputSynapses[h][i].weight;
            }
            
            // Aplicar función de activación
            neuron.output = funcion_activacion(neuron.input);
            neuron.input = 0.0;
        }
        
        // Devolver las salidas
        return outputLayer.map!(n => n.output).array;
    }
    
    void train(float[][] inputSequence, float[] expectedOutput, float learningRate) {
        // Forward pass
        float[] prediction = predict(inputSequence);
        
        // Calcular error en la capa de salida
        foreach (i, ref neuron; outputLayer) {
            neuron.error = expectedOutput[i] - neuron.output;
        }
        
        // Backpropagation para la capa de salida
        foreach (ref neuron; outputLayer) {
            neuron.delta = neuron.error * funcion_gradiente(neuron.output);
            
            // Actualizar pesos de la capa de salida
            foreach (ref synapse; neuron.backwardSynapses) {
                // En este caso, backwardNeuron es una neurona ficticia que representa la salida GRU
                float gruOutput = synapse.backwardNeuron.name.startsWith("GRU_Output_") ? 
                    gruLayer[0].output[synapse.backwardNeuron.name["GRU_Output_".length].to!int] : 0.0;
                
                synapse.weight += gruOutput * neuron.delta * learningRate;
                synapse.weight = clamp(synapse.weight, -20.0, 20.0);
            }
        }
        
        // Backpropagation para la capa GRU (simplificado)
        // Nota: Una implementación completa requeriría backpropagation a través del tiempo (BPTT)
        // que es más compleja y requiere mantener un historial de estados y gradientes
        
        // Podemos simplificar asumiendo que el gradiente se propaga hacia atrás en el tiempo
        // y afecta solo al último estado GRU
        float[] outputGradient = new float[hiddenSize];
        foreach (i, ref neuron; outputLayer) {
            for (int h = 0; h < hiddenSize; h++) {
                outputGradient[h] += neuron.delta * outputSynapses[h][i].weight;
            }
        }
        
        // Llamar al método backward de GRU (que habría que implementar completamente)
        // gruLayer[0].backward(outputGradient, learningRate);
    }
    
    void trainBPTT(float[][][] batchInputSequences, float[][] batchExpectedOutputs, int epochs, float learningRate) {
        // Implementación de Backpropagation Through Time (BPTT)
        // Esta es una versión simplificada
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalError = 0.0;
            
            for (int b = 0; b < batchInputSequences.length; b++) {
                float[][] inputSequence = batchInputSequences[b];
                float[] expectedOutput = batchExpectedOutputs[b];
                
                // Forward pass
                float[] prediction = predict(inputSequence);
                
                // Calcular error
                float error = 0.0;
                foreach (i; 0 .. prediction.length) {
                    float diff = expectedOutput[i] - prediction[i];
                    error += diff * diff;
                }
                totalError += error;
                
                // Backward pass
                train(inputSequence, expectedOutput, learningRate);
            }
            
            import std.stdio : writeln;
            if (epoch % 10 == 0) {
                writeln("Epoch ", epoch, " - Error: ", totalError / batchInputSequences.length);
            }
        }
    }
    
}