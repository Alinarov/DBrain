#!/usr/bin/env dmd
module DBrain.herramientas.calculaciones.estadistica;
import std;
/* ---------------------------- */
import DBrain.herramientas.calculaciones.math;
/* --------------------------- */

alias print = writeln;

// Función para calcular la media de una columna
float calcularMedia(float[] columna) {
    return columna.sum / columna.length;
}

// Función para calcular la desviación estándar de una columna
float calcularDesviacionEstandar(float[] columna, float media) {
    float sumaCuadrados = columna.map!(valor => (valor - media) * (valor - media)).sum;
    return sqrt(sumaCuadrados / columna.length);
}

// Función para normalizar una columna usando Z-score
float[] normalizarColumnaZScore(float[] columna) {
    float media = calcularMedia(columna);
    float desviacion = calcularDesviacionEstandar(columna, media);
    return columna.map!(valor => (valor - media) / desviacion).array;
}

// Función para desnormalizar un valor Z-score
float desnormalizarZScore(float valorNormalizado, float media, float desviacion) {
    return (valorNormalizado * desviacion) + media;
}

float normalize(float x, float min, float max) {
    return (x - min) / (max - min);
}


float desnormalize(float value, float min, float max) {
    return value * (max - min) + min;
}

float[] desnormalizeArray(float[] valores, float min, float max) {
    return valores.map!(x => x * (max - min) + min).array;
}

