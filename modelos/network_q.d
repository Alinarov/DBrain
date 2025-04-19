#!/usr/bin/env dmd 
module DBrain.modelos.network_q;

import std;
import std.math : exp;
import std.math.exponential: log;
import std.algorithm : map, clamp, reduce, max;
import std.array : array;
import std.range : iota, chunks;
import std.stdio;

/* ------------------ */
import DBrain.funciones_activacion;
import DBrain.herramientas.herramientas_network;
/* ------------------ */

alias print = writeln;

/* DATOS GLOBALES DE CONFIGURACION */

float learningRate = 0.09;
float discountFactor = 0.95;


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

class Neuron {

    // Esto va a jalar de la biblioteca de funciones 
    // funciones.d
    float function (float x) funcion_activacion;
    float function (float x) funcion_gradiente;

    float momentum = 0.9;
    float [] prevWeightUpdates;

    string name;
    float delta, error = 0.0, input = 0.0, output;
    Synapse[] backwardSynapses, forwardSynapses;

	this(string name) {
	    this.name = name;
	    this.output = 0.0;
	    this.input = 0.0;
	    this.error = 0.0;
	    this.delta = 0.0;
	    this.prevWeightUpdates = [];
	}

    void backpropagateAndAdjust() {
        if (this.forwardSynapses.length > 0)
            this.error /= this.forwardSynapses.length;

        this.delta = this.error * funcion_gradiente(this.output);

        this.error = 0.0;

        foreach (ref synapse; this.backwardSynapses) {
            synapse.backwardNeuron.error += this.delta * synapse.weight;
            synapse.weight += synapse.backwardNeuron.output * this.delta * learningRate;

            synapse.weight = clamp(synapse.weight, -20.0, 20.0);
        }
    }
}

class Network_Q {
    Neuron[][] layers;


    this(ref Neuron[][] layers) {
        this.layers = layers;
    }

    // Modifica la función think
    float[] think(float[] input) {

		assert(input.length == this.layers[0].length, "Input length must match the number of input neurons. Expected: " ~ this.layers[0].length.to!string ~ ", Got: " ~ input.length.to!string);
        // Propagación hacia adelante
        foreach (i, ref neuron; this.layers[0]) {
            neuron.output = input[i];

            foreach (ref synapse; neuron.forwardSynapses) 
                synapse.forwardNeuron.input += neuron.output * synapse.weight;
        }

        // Capa oculta
        foreach (ref layer; this.layers[1 .. $]) {
            foreach (ref neuron; layer) {
                neuron.output = neuron.funcion_activacion(neuron.input);
                neuron.input = 0.0;

                foreach (ref synapse; neuron.forwardSynapses)
                    synapse.forwardNeuron.input += neuron.output * synapse.weight;
            }
        }

        // Obtener las salidas
        float[] outputs = this.layers[$ - 1].map!(n => n.output).array;

        return outputs;
    }

    // Función de aprendizaje (backpropagation)
    void learn(float[] expected) {
        // Calcula el error y ajusta los pesos
		foreach (i, ref neuron; this.layers[$ - 1]) {
		    neuron.error = expected[i] - neuron.output;
		}

        // Retropropagación del error
        foreach_reverse (ref layer; this.layers[1 .. $])
            foreach (ref neuron; layer)
                neuron.backpropagateAndAdjust();
    }

    // Función para actualizar los valores Q usando la fórmula de Q-Learning
    void updateQValues(float[] currentState, int action, float reward, float[] nextState) {
        // Obtén el valor Q actual (antes de tomar la acción)
        float[] qValues = this.think(currentState);  // Q(s, a) para todas las acciones
        float currentQ = qValues[action];  // Q(s, a) para la acción específica

        // Obtén los valores Q para el siguiente estado
        float[] nextQValues = this.think(nextState);
        //float maxNextQ = nextQValues.maxElement;  // max_a' Q(s', a')
        float maxNextQ = nextQValues.reduce!max;  // max_a' Q(s', a')

        // Fórmula de Q-Learning
        float updatedQ = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);
        
        // Actualiza el valor Q para la acción seleccionada
        qValues[action] = updatedQ;

        // Usa la red neuronal para ajustar los pesos basados en el valor Q actualizado
        this.learn(qValues);
    }

    // Función para imprimir los pesos
    void printWeights() {
        foreach (layerIndex, ref layer; this.layers) {
            writeln("Capa ", layerIndex);
            foreach (neuronIndex, ref neuron; layer) {
                writeln("  Neurona ", neuronIndex, " (", neuron.name, "):");
                foreach (synapseIndex, ref synapse; neuron.backwardSynapses) {
                    writeln("    Sinapsis ", synapseIndex, ": Peso = ", synapse.weight);
                }
            }
        }
    }



    void guardarPesos(Network_Q network, string nombreArchivo) {
        print("(+) Guardando pesos");

        string contenido;

        foreach (layerIndex, ref layer; network.layers) {
            foreach (neuronIndex, ref neuron; layer) {
                foreach (synapseIndex, ref synapse; neuron.backwardSynapses) {
                    contenido ~= "Capa " ~ to!string(layerIndex) ~ 
                                 ", Neurona " ~ to!string(neuronIndex) ~ 
                                 ", Sinapsis " ~ to!string(synapseIndex) ~ 
                                 ", Peso: " ~ to!string(synapse.weight) ~ "\n";
                }
            }
        }

        // Escribir el contenido en el archivo
        std.file.write(nombreArchivo, contenido);
        print("(-) Pesos guardados");

    }

    void cargarPesos(ref Network_Q network, string nombreArchivo) {
        print("(+) cargando pesos");


        string contenido = readText(nombreArchivo);
        string[] lineas = splitLines(contenido);

        foreach (linea; lineas) {
            string[] partes = split(linea, ", ");
            if (partes.length == 4) {
                int layerIndex = to!int(partes[0].split(" ")[1]);
                int neuronIndex = to!int(partes[1].split(" ")[1]);
                int synapseIndex = to!int(partes[2].split(" ")[1]);
                float peso = to!float(partes[3].split(" ")[1]);

                // Asignar el peso a la sinapsis correspondiente
                network.layers[layerIndex][neuronIndex].backwardSynapses[synapseIndex].weight = peso;
            }
        }
    }



}


