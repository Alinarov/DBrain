#!/usr/bin/env dmd 
module DBrain.modelos.network_gru;

import std;
import std.math : exp;
import std.math.algebraic : sqrt;
import std.math.exponential: log;
import std.algorithm : map, clamp, reduce, max;
import std.array : array;
import std.range : iota, chunks;
import std.stdio;

/* ------------------ */
import DBrain.DBrain;
import DBrain.funciones_activacion;
import DBrain.herramientas.herramientas_network;
/* ------------------ */

alias print = writeln;

/* DATOS GLOBALES DE CONFIGURACION */

float learningRate = 0.09;
float discountFactor = 0.95;


// Esto va a jalar de la biblioteca de funciones 
// funciones.d


//float function (float x, float alpha = -1.0) funcion_activacion;

//float function (float x, float alpha = -1.0) funcion_gradiente;

float function (float x) funcion_activacion;

float function (float x) funcion_gradiente;


/* ------------------------------------------------------- */


class Synapse {
    float weight;
    Neuron backwardNeuron, forwardNeuron;

    this(float weight, ref Neuron backwardNeuron, ref Neuron forwardNeuron) {
        this.weight = weight;
        this.forwardNeuron = forwardNeuron;
        this.backwardNeuron = backwardNeuron;
    }
}
class GRUCell {
    float[] resetGateWeights, updateGateWeights, candidateWeights;
    float hiddenState;
    
    this(int inputSize) {
        float factor = 1.0 / sqrt(to!float(inputSize));
        resetGateWeights = new float[inputSize].map!(x => uniform(-factor, factor)).array;
        updateGateWeights = new float[inputSize].map!(x => uniform(-factor, factor)).array;
        candidateWeights = new float[inputSize].map!(x => uniform(-factor, factor)).array;
        hiddenState = 0.0;
    }

    float forward(float[] input) {
        float resetGate = funcion_activacion(dotProduct(resetGateWeights, input));
        float updateGate = funcion_activacion(dotProduct(updateGateWeights, input));
        float candidate = funcion_gradiente(dotProduct(candidateWeights, input) + resetGate * hiddenState);
        hiddenState = updateGate * hiddenState + (1 - updateGate) * candidate;
        return hiddenState;
    }

    float dotProduct(float[] a, float[] b) {

        int minLengt = min(to!int(a.length), to!int(b.length));
        //assert(a.length == b.length, "Error: Diferente longitud en dotProduct");
        float result = 0.0;
        foreach (i; 0 .. minLengt) {
            result += a[i] * b[i];
        }
        return result;
    }
}

class GRUNetwork {
    GRUCell[] layers;

    this(int inputSize, int[] layerSizes) {
        foreach (size; layerSizes) {
            layers ~= new GRUCell(inputSize);
            inputSize = size;
        }
    }

    float[] forward(float[] input) {
        float[] state = input;
        foreach (layer; layers) {
            state = [layer.forward(state)];
        }
        return state;
    }

    void updateQValues(float[] currentState, int action, float reward, float[] nextState) {
        float[] currentQValues = this.forward(currentState);
        float currentQ = currentQValues[action];

        float[] nextQValues = this.forward(nextState);
        float maxNextQ = nextQValues.reduce!max;

        float updatedQ = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);

        currentQValues[action] = updatedQ;
        this.learn(currentState, currentQValues);
    }

    void updateWeights(ref float[] weights, float[] input, float[] error) {
        int minLength = min(to!int(weights.length), to!int(input.length)); // Evitar índices fuera de rango
        foreach (i; 0 .. minLength) {
            weights[i] -= learningRate * error[0] * input[i]; // Ajuste de peso
        }
    }

    float[] backpropagate(float[] error, float[] output, float[] weights) {
        return error.map!(e => e * funcion_gradiente(output[0]) * weights[0]).array;
    }



    void learn(float[] input, float[] target) {
        float[] output = this.forward(input);
        float[] error = output.map!(o => target[0] - o).array;

        foreach_reverse (layer; layers) {
            // Calcula el gradiente de error para esta capa
            float[] layerError = error.map!(e => e * funcion_gradiente(layer.hiddenState)).array;

            // Actualiza los pesos de la capa
            updateWeights(layer.resetGateWeights, input, layerError);
            updateWeights(layer.updateGateWeights, input, layerError);
            updateWeights(layer.candidateWeights, input, layerError);

            // Calcula y pasa el error hacia la capa anterior
            error = backpropagate(layerError, input, layer.resetGateWeights);
        }
    }
}
