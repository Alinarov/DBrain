#!/usr/bin/env dmd
module DBrain.herramientas.herramientas_network;
//import std.file : readText;
//import std.string : splitLines, split;
//import std.conv : to;
import std;


float calcularRecompensa(float prediccion, float valorReal) {
    float error = abs(prediccion - valorReal);
    // Recompensa basada en cercanía del error
    return 1.0 / (1.0 + error);
}

//float xavierInit(float tamano_salida, float tamano_entrada) {
//    return uniform(
//        -sqrt(6.0f / (tamano_entrada + tamano_salida)), 
//        sqrt(6.0f / (tamano_entrada + tamano_salida))
//    );
//}

