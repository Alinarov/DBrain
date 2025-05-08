#!/usr/bin/env dmd
// Esta es una biblioteca de funciones de activacion donde pondre todos las funciones de activacion
module DBrain.funciones_activacion;
import std;
import std.math;
import std.math : exp, log, tanh, sqrt, PI, pow;
import std.algorithm : map, reduce;
import std.array : array;
import std.stdio : writeln;
/* ------------------------------------ */


/* -------------------------------------- */

alias print = writeln;


// 
float learningRate = 0.09;
float discountFactor = 0.95;


/* -------------- LSTM ---------------- */

float sigmoid_estable(float x) {
    x = max(-50.0f, min(50.0f, x)); // Evita overflow
    return 1.0f / (1.0f + exp(-x));
}

float tanh_estable(float x) {
    x = max(-50.0f, min(50.0f, x)); // Evita overflow
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

float tanh_derivada_estable(float x) {
    float t = tanh_estable(x);
    return 1 - t * t;
}


/* --------------------------------------- */





// funcion de activacion: tanh
float tanh_hiperbolica(float x) {
    float e_pos = exp(x);
    float e_neg = exp(-x);
    return (e_pos - e_neg) / (e_pos + e_neg);
}


float tanh_derivada(float x) {
    float tanh_x = tanh_hiperbolica(x);
    return 1.0f - tanh_x * tanh_x;  // tanh'(x) = 1 - tanh²(x)
}


// Función de activación: Sigmoid
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Gradiente de la función de activación: Sigmoid
float sigmoid_gradient(float x) {
    float sig = sigmoid(x);
    return sig * (1.0 - sig);
}
// Función de activación: Leaky ReLU
float leaky_relu(float x) {
    return x > 0 ? x : 0.05 * x;
}

// Gradiente de la función de activación: Leaky ReLU
float leaky_relu_gradient(float x) {
    return x > 0 ? 1 : 0.05;
}

// Softplus
float softplus(float x) {
    return log(1.0 + exp(x));
}

float softplus_gradient(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// ELU (Exponential Linear Unit)
float elu(float x, float alpha = 1.0) {
    return x > 0 ? x : alpha * (exp(x) - 1);
}

float elu_gradient(float x, float alpha = 1.0) {
    return x > 0 ? 1 : elu(x, alpha) + alpha;
}

// Swish
float swish(float x) {
    return x * (1.0 / (1.0 + exp(-x)));
}

float swish_gradient(float x) {
    float sig = 1.0 / (1.0 + exp(-x));
    return sig + x * sig * (1 - sig);
}

// Softmax (para un vector de entradas)
float[] softmax(float[] x) {
    float max_x = x.reduce!max;
    float[] exp_x = x.map!(a => exp(a - max_x)).array;
    float sum_exp_x = exp_x.reduce!((a, b) => a + b);
    return exp_x.map!(a => a / sum_exp_x).array;
}

// Softmax gradient (para un solo elemento)
float softmax_gradient(float x, float[] softmax_output) {
    return softmax_output[0] * (1 - softmax_output[0]); // Para el primer elemento
}

// Linear (Identidad)
float linear(float x) {
    return x;
}

float linear_gradient(float x) {
    return 1;
}

// Step (Escalón)
float step(float x) {
    return x > 0 ? 1 : 0;
}

float step_gradient(float x) {
    return 0; // No diferenciable en x = 0
}

// GELU (Gaussian Error Linear Unit)
float gelu(float x) {
    return 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * pow(x, 3))));
}

float gelu_gradient(float x) {
    float cdf = 0.5 * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * pow(x, 3))));
    return cdf + x * (1 - cdf * cdf) * sqrt(2 / PI) * (1 + 0.134145 * x * x);
}

// SELU (Scaled Exponential Linear Unit)
float selu(float x, float alpha = 1.67326, float scale = 1.0507) {
    return x > 0 ? scale * x : scale * alpha * (exp(x) - 1);
}

float selu_gradient(float x, float alpha = 1.67326, float scale = 1.0507) {
    return x > 0 ? scale : scale * alpha * exp(x);
}

// Mish
float mish(float x) {
    return x * tanh(log(1 + exp(x)));
}

float mish_gradient(float x) {
    float omega = exp(3 * x) + 4 * exp(2 * x) + (6 + 4 * x) * exp(x) + 4 * (1 + x);
    float delta = 2 * exp(x) + exp(2 * x) + 2;
    return exp(x) * omega / (delta * delta);
}