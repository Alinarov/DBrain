#!/usr/bin/env dmd
module DBrain.herramientas.app_herramientas;

import std;

/* --------------------------------------- */
import DBrain.modelos.network_q;
import DBrain.herramientas.calculaciones.estadistica;
/* ----------------------------------------- */

alias print = writeln;

// Codificación de estaciones
float[] codificarEstacion(string estacion) {
    switch(estacion) {
        case "Primavera": return [1,0,0,0];
        case "Verano":    return [0,1,0,0];
        case "Otoño":     return [0,0,1,0];
        case "Invierno":  return [0,0,0,1];
        default:          return [0,0,0,0];
    }
}

// Lista de productos filtrados
protected string[] productos;

string[] get_productos () {
	return productos;
}

void set_productos (string[] product) {
	productos ~= product;
}

// One-hot encoding para productos
float[] codificarProducto(string producto) {
    string[] productos = get_productos();
    float[] resultado = new float[productos.length];
    auto index = productos.countUntil(producto);
    if (index != -1) resultado[index] = 1;
    // limpio los datos nan **not a number** 
    foreach(i, ref dato; resultado) {
        if(isNaN(dato)) resultado[i] = 0.0;
    }
    
    return resultado;
}
/* -------------------------------------------------------------------------*/

// Estructura para almacenar los productos binarios normalizados
struct list_datos_Z {
    float[] productos_binarios;
}

// Función para obtener los productos binarios
float[] get_productos_binarios(list_datos_Z lista) {
    return lista.productos_binarios;
}

// Función para agregar productos binarios
void set_productos_binarios(float[] produ, ref list_datos_Z lista) {
    lista.productos_binarios ~= produ;
}

// Función para convertir un código binario en un número entero
int binarioAEntero(int[] binario) {
    string cadenaBinaria = binario.map!(b => to!string(b)).join("");
    return to!int(cadenaBinaria); // Convertir de binario a entero
}

// Función para codificar un índice en binario
int[] codificarBinario(int indice, int numBits) {
    int[] binario = new int[numBits];
    binario[] = 0; // Inicializar con 0

    for (int i = 0; i < numBits; i++) {
        binario[numBits - 1 - i] = indice & 1; // Obtener el bit menos significativo
        indice >>= 1; // Desplazar a la derecha
    }

    return binario;
}

// Función para calcular el número de bits necesarios
int calcularNumBits(int numProductos) {
    return cast(int)(log2(to!float(numProductos))) + 1;
}




// Función para codificar y normalizar una columna categórica
list_datos_Z codificarYNormalizar(string[] categorias) {
    list_datos_Z datos_bin;

    // Calcular el número de bits necesarios
    int numBits = calcularNumBits(to!int(categorias.length));

    // Codificar y convertir las categorías a enteros
    foreach (i, ref dato; categorias) {
        int[] binario = codificarBinario(to!int(i), numBits);
        int entero = binarioAEntero(binario);
        set_productos_binarios([to!float(entero)], datos_bin);
    }

    // Normalizar los enteros con Z-score
    float[] valoresNormalizados = normalizarColumnaZScore(get_productos_binarios(datos_bin));
    datos_bin.productos_binarios = valoresNormalizados;

    return datos_bin;
}


// Función para dividir los datos en entrenamiento y prueba
float[][][] dividirDatos(float[][] datos, float porcentajeEntrenamiento) {
    Random rnd;
    rnd.seed(42); // Semilla para reproducibilidad

    // Mezclar los datos
    auto datosMezclados = datos.dup;
    for (ulong i = datosMezclados.length - 1; i > 0; i--) {
        
        int x = to!int(i);
        int j = uniform(0, x + 1, rnd);
        auto temp = datosMezclados[x];
        datosMezclados[x] = datosMezclados[j];
        datosMezclados[j] = temp;
    }

    // Dividir los datos
    int indiceDivision = cast(int)(datosMezclados.length * porcentajeEntrenamiento);
    float[][] entrenamiento = datosMezclados[0 .. indiceDivision];
    float[][] prueba = datosMezclados[indiceDivision .. $];

    return [entrenamiento, prueba];
}




Neuron[][] createNeuralNetwork(int inputNeurons, int[] hiddenLayers, int outputNeurons) {
    Neuron[][] layers;

    // Crear capa de entrada
    Neuron[] inputLayer;
    for (int i = 0; i < inputNeurons; i++) {
        inputLayer ~= new Neuron("Input " ~ i.to!string);
    }
    layers ~= inputLayer;

    // Crear capas ocultas
    Neuron[] prevLayer = inputLayer;
    foreach (layerSize; hiddenLayers) {
        Neuron[] hiddenLayer;
        for (int i = 0; i < layerSize; i++) {
            Neuron neuron = new Neuron("Hidden " ~ i.to!string);
            hiddenLayer ~= neuron;

            // Crear sinapsis entre la capa previa y la actual
            foreach (prevNeuron; prevLayer) {
                Synapse synapse = new Synapse(uniform(-1, 1), prevNeuron, neuron);
                prevNeuron.forwardSynapses ~= synapse;
                neuron.backwardSynapses ~= synapse;
            }
        }
        layers ~= hiddenLayer;
        prevLayer = hiddenLayer;
    }

    // Crear capa de salida
    Neuron[] outputLayer;
    for (int i = 0; i < outputNeurons; i++) {
        Neuron neuron = new Neuron("Output " ~ i.to!string);
        outputLayer ~= neuron;

        // Crear sinapsis entre la última capa oculta y la capa de salida
        foreach (prevNeuron; prevLayer) {
            Synapse synapse = new Synapse(uniform(-1, 1), prevNeuron, neuron);
            prevNeuron.forwardSynapses ~= synapse;
            neuron.backwardSynapses ~= synapse;
        }
    }
    layers ~= outputLayer;

    return layers;
}

float xavierInit(float tamaño_entrenamiento, float tamaño_prueba) {
    float bound = sqrt(6.0f / (tamaño_entrenamiento + tamaño_prueba));  // Ajusta inputSize/hiddenSize
    return uniform(-bound, bound);
}


