//#!/usr/bin/env dmd 
//module DBrain.modelos.network_gru;

//import std;
//import std.math : exp;
//import std.math.exponential: log;
//import std.algorithm : map, clamp, reduce, max;
//import std.array : array;
//import std.range : iota, chunks;
//import std.stdio;
//import DBrain.DBrain;
///* ------------------ */
//import DBrain.funciones_activacion;
//import DBrain.herramientas.herramientas_network;
///* ------------------ */

//alias print = writeln;

///**
// * GRU Cell implementation that properly manages hidden state vectors
// */
//class GRUCell {
//    float[] resetGateWeights;
//    float[] resetGateBias;
//    float[] updateGateWeights;
//    float[] updateGateBias;
//    float[] candidateWeights;
//    float[] candidateBias;
//    float[] hiddenState;
//    int hiddenSize;
    
//    // Function pointers for activation functions
//    float function(float) sigmoid;
//    float function(float) tanh;
    
//    this(int inputSize, int hiddenSize) {
//        this.hiddenSize = hiddenSize;
        
//        // Initialize weights with proper scaling
//        float scale = sqrt(1.0f / (inputSize + hiddenSize));
        
//        // For reset gate
//        resetGateWeights = limpiar_nan(new float[inputSize + hiddenSize]);
        
//        foreach (ref w; resetGateWeights) w = uniform(-scale, scale);
//        resetGateBias = limpiar_nan(new float[hiddenSize]);
        
//        // For update gate
//        updateGateWeights = limpiar_nan(new float[inputSize + hiddenSize]);
//        foreach (ref w; updateGateWeights) w = uniform(-scale, scale);
//        updateGateBias = limpiar_nan(new float[hiddenSize]);
    	
        
//        // For candidate values
//        candidateWeights = limpiar_nan(new float[inputSize + hiddenSize]);
//        foreach (ref w; candidateWeights) w = uniform(-scale, scale);
//        candidateBias = limpiar_nan(new float[hiddenSize]);
    	
        
//        // Initialize hidden state
//        hiddenState = limpiar_nan(new float[hiddenSize]);
        
        
//        // Set activation functions
//        sigmoid = (float x) => 1.0f / (1.0f + exp(-x));
//        tanh = (float x) => (exp(x) - exp(-x)) / (exp(x) + exp(-x));
//    }
    
//    float[] forward(float[] input) {
//        // Ensure hidden state is initialized
//        if (hiddenState.length != hiddenSize) {
//            hiddenState = limpiar_nan(new float[hiddenSize]);
//        }
        
//        // Concatenate input with previous hidden state for weight calculations
//        float[] combinedInput = input ~ hiddenState;
        
//        // Reset gate computation
//        float[] resetGate = limpiar_nan(new float[hiddenSize]);
//        for (int h = 0; h < hiddenSize; h++) {
//            resetGate[h] = sigmoid(dotProduct(combinedInput, resetGateWeights, h * to!int(combinedInput.length), to!int(combinedInput.length)) + resetGateBias[h]);
//        }
        
//        // Update gate computation
//        float[] updateGate = limpiar_nan(new float[hiddenSize]);
//        for (int h = 0; h < hiddenSize; h++) {
//            updateGate[h] = sigmoid(dotProduct(combinedInput, updateGateWeights, h * to!int(combinedInput.length), to!int(combinedInput.length)) + updateGateBias[h]);
//        }
        
//        // Compute reset hidden state
//        float[] resetHiddenState = limpiar_nan(new float[hiddenSize]);
//        for (int h = 0; h < hiddenSize; h++) {
//            resetHiddenState[h] = resetGate[h] * hiddenState[h];
//        }
        
//        // Compute candidate values using reset gate
//        float[] candidate = limpiar_nan(new float[hiddenSize]);
//        combinedInput = input ~ resetHiddenState; // Recombine with reset hidden state
//        for (int h = 0; h < hiddenSize; h++) {
//            candidate[h] = tanh(dotProduct(combinedInput, candidateWeights, h * to!int(combinedInput.length), to!int(combinedInput.length)) + candidateBias[h]);
//        }
        
//        // Update hidden state
//        float[] newHiddenState = limpiar_nan(new float[hiddenSize]);
//        for (int h = 0; h < hiddenSize; h++) {
//            newHiddenState[h] = updateGate[h] * hiddenState[h] + (1 - updateGate[h]) * candidate[h];
//        }
        
//        // Store the new hidden state
//        hiddenState = newHiddenState;
        
//        return hiddenState;
//    }

//    private float dotProduct(float[] a, float[] b, int offset, int length) {
//        float result = 0.0f;
//        int end = min(offset + length, b.length);
//        for (int i = offset; i < end; i++) {
//            result += a[i - offset] * b[i];
//        }
//        return result;
//    }
    
//    float[] getState() {
//        return hiddenState.dup;
//    }
    
//    void resetState() {
//        foreach (ref h; hiddenState) h = 0.0f;
//    }
//}

///**
// * GRU Network with proper sequence handling and backpropagation
// */
//class GRUNetwork {
//    GRUCell[] layers;
//    float[][] outputWeights;
//    float[] outputBias;
//    int[] layerSizes;
//    int inputSize;
//    int outputSize;
//    float learningRate;
//    float discountFactor;
    
//    this(int inputSize, int[] layerSizes, int outputSize, float learningRate = 0.01, float discountFactor = 0.95) {
//        this.inputSize = inputSize;
//        this.layerSizes = layerSizes;
//        this.outputSize = outputSize;
//        this.learningRate = learningRate;
//        this.discountFactor = discountFactor;
        
//        // Create GRU layers
//        int prevSize = inputSize;
//        foreach (size; layerSizes) {
//            layers ~= new GRUCell(prevSize, size);
//            prevSize = size;
//        }
        
//        // Initialize output layer weights
//        int lastLayerSize = layerSizes[$ - 1];
//        outputWeights = new float[][](lastLayerSize, outputSize);
//        foreach (i; 0 .. lastLayerSize) {
//            foreach (j; 0 .. outputSize) {
//                outputWeights[i][j] = uniform(-0.1f, 0.1f);
//            }
//        }
        
//        // Initialize output bias
//        outputBias = new float[outputSize];
//    }
    
//    float[] forward(float[] input) {
//        float[] state = input;
        
//        // Process through each GRU layer
//        foreach (layer; layers) {
//            state = layer.forward(state);
//        }
        
//        // Calculate final output
//        float[] output = limpiar_nan(new float[outputSize]);
//        for (int j = 0; j < outputSize; j++) {
//            output[j] = outputBias[j];
//            for (int i = 0; i < state.length; i++) {
//                output[j] += state[i] * outputWeights[i][j];
//            }
//        }
        
//        return output;
//    }
    
//    // Reset all hidden states in the network
//    void resetStates() {
//        foreach (layer; layers) {
//            layer.resetState();
//        }
//    }
    
//    // Process a sequence of inputs
//    float[][] processSequence(float[][] inputSequence) {
//        resetStates();
//        float[][] outputs;
        
//        foreach (input; inputSequence) {
//            outputs ~= forward(input);
//        }
        
//        return outputs;
//    }
    
//    // Supervised learning for sequence data
//    void trainSupervised(float[][] inputSequence, float[][] targetSequence, int epochs = 1) {
//        for (int epoch = 0; epoch < epochs; epoch++) {
//            float totalLoss = 0.0f;
            
//            resetStates();
            
//            // Forward pass through sequence
//            float[][] predictions;
//            foreach (input; inputSequence) {
//                predictions ~= forward(input);
//            }
            
//            // Calculate loss
//            for (int t = 0; t < predictions.length; t++) {
//                for (int j = 0; j < outputSize; j++) {
//                    float error = targetSequence[t][j] - predictions[t][j];
//                    totalLoss += error * error;
//                }
//            }
            
//            totalLoss /= (predictions.length * outputSize);
//            writeln("Epoch ", epoch, " Loss: ", totalLoss);
            
//            // Simple backpropagation (simplified for clarity)
//            // In a full implementation, you would include backpropagation through time (BPTT)
//            for (int t = 0; t < inputSequence.length; t++) {
//                float[] target = targetSequence[t];
//                float[] prediction = predictions[t];
                
//                // Calculate output layer gradients
//                float[] outputGradients = limpiar_nan(new float[outputSize]);
//                for (int j = 0; j < outputSize; j++) {
//                    outputGradients[j] = target[j] - prediction[j];
//                }
                
//                // Update output weights
//                float[] lastLayerOutput = layers[$ - 1].getState();
//                for (int i = 0; i < lastLayerOutput.length; i++) {
//                    for (int j = 0; j < outputSize; j++) {
//                        outputWeights[i][j] += learningRate * outputGradients[j] * lastLayerOutput[i];
//                    }
//                }
                
//                // Update output bias
//                for (int j = 0; j < outputSize; j++) {
//                    outputBias[j] += learningRate * outputGradients[j];
//                }
                
//                // For a full implementation, you would now backpropagate through the GRU layers
//                // This is simplified for clarity
//            }
//        }
//    }
    
//    //// Reinforcement learning using Q-learning
//    //void updateQValues(float[] state, int action, float reward, float[] nextState) {
//    //    // Get current Q-values
//    //    float[] currentQValues = forward(state);
        
//    //    // Get next state's maximum Q-value
//    //    float[] nextQValues = forward(nextState);
//    //    float maxNextQ = nextQValues.reduce!max;
        
//    //    // Calculate target Q-value for the action taken
//    //    float targetQ = reward + discountFactor * maxNextQ;
        
//    //    // Update only the Q-value for the action taken
//    //    float[] targetQValues = currentQValues.dup;
//    //    targetQValues[action] = currentQValues[action] + learningRate * (targetQ - currentQValues[action]);
        
//    //    // Calculate error gradients
//    //    float[] gradients = limpiar_nan(new float[outputSize]);
//    //    for (int i = 0; i < outputSize; i++) {
//    //        gradients[i] = targetQValues[i] - currentQValues[i];
//    //    }
        
//    //    // Update output layer weights
//    //    float[] lastLayerOutput = layers[$ - 1].getState();
//    //    for (int i = 0; i < lastLayerOutput.length; i++) {
//    //        for (int j = 0; j < outputSize; j++) {
//    //            outputWeights[i][j] += learningRate * gradients[j] * lastLayerOutput[i];
//    //        }
//    //    }
        
//    //    // Update output bias
//    //    for (int j = 0; j < outputSize; j++) {
//    //        outputBias[j] += learningRate * gradients[j];
//    //    }
        
//    //    // For a complete implementation, you would include backpropagation through the GRU layers
//    //}
    
//    // Save model parameters to file
//    void saveModel(string filename) {
//        File file = File(filename, "w");
        
//        // Save network architecture
//        file.writeln(inputSize);
//        file.writeln(layerSizes.length);
//        foreach (size; layerSizes) {
//            file.writeln(size);
//        }
//        file.writeln(outputSize);
//        file.writeln(learningRate);
//        file.writeln(discountFactor);
        
//        // Save weights and biases for each layer
//        // (Implementation details omitted for brevity)
        
//        file.close();
//    }
    
//    // Load model parameters from file
//    static GRUNetwork loadModel(string filename) {
//        File file = File(filename, "r");
        
//        // Read network architecture
//        int inputSize = to!int(file.readln().strip());
//        int numLayers = to!int(file.readln().strip());
//        int[] layerSizes = new int[numLayers];
        
//        for (int i = 0; i < numLayers; i++) {
//            layerSizes[i] = to!int(file.readln().strip());
//        }
        
//        int outputSize = to!int(file.readln().strip());
//        float learningRate = to!float(file.readln().strip());
//        float discountFactor = to!float(file.readln().strip());
        
//        // Create network
//        auto network = new GRUNetwork(inputSize, layerSizes, outputSize, learningRate, discountFactor);
        
//        // Load weights and biases
//        // (Implementation details omitted for brevity)
        
//        file.close();
//        return network;
//    }
//}

//// Simple usage example
//void main() {
//    // Create a GRU network with 3 inputs, one hidden layer of 5 units, and 2 outputs
//    auto network = new GRUNetwork(3, [5], 2, 0.01, 0.95);
    
//    // Example input sequence (2 time steps, 3 features each)
//    float[][] inputSequence = [
//        [0.1, 0.2, 0.3],
//        [0.4, 0.5, 0.6]
//    ];
    
//    // Example target sequence
//    float[][] targetSequence = [
//        [0.7, 0.8],
//        [0.9, 1.0]
//    ];
    
//    // Train the network
//    network.trainSupervised(inputSequence, targetSequence, 100);
    
//    // Use the trained network
//    float[] newInput = [0.4, 0.5, 0.6];
//    float[] prediction = network.forward(newInput);
//    writeln("Prediction: ", prediction);
//}