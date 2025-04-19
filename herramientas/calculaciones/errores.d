#!/usr/bin/env dmd
module DBrain.herramientas.calculaciones.errores;
import std;

/**
 * Calcula el error absoluto y el error porcentual absoluto.
 * Parámetros:
 *   x: Valor real.
 *   y: Valor predicho.
 * Retorna:
 *   Una tupla con el error absoluto y el error porcentual absoluto.
 */

alias print = writeln;

auto calcular_mae (float original, float predicho) {

	// Error absoluto
	float error_absoluto = abs(original - predicho);

	float error_absoluto_porcentual = error_absoluto / original;

	return tuple(error_absoluto,error_absoluto_porcentual);
}

float calcular_mape (float[] x, float[] y) {

    float sumatoria = 0; 
    ulong n_reales = x.length;

    // Calcular errores para cada par de valores
    foreach (i; 0 .. x.length) {
        auto errores = calcular_mae(x[i], y[i]);
        writeln("Valor real: ", x[i], " | Valor predicho: ", y[i]);
        writeln("Error absoluto mae: ", errores[0]);
        writeln("Error porcentual absoluto: ", errores[1], "%");
        writeln("-----------------------------");
        sumatoria += errores[1];

    }
    
    float mape = sumatoria / n_reales;

    //print("La mape es: ", mape);
    return mape;


}

//void main () {
//    float[] x = [3.14, 2.34, 5.67, 1.23, 4.56, 7.89, 0.98, 6.54, 3.21, 9.87]; // Valores reales
//    float[] y = [6.13, 5.35, 7.70, 2.20, 9.50, 15.85, 1.99, 12.50, 6.22, 19.90]; // Valores predichos

//    float mape = calcular_mape(x, y);
//    print(mape);

//}
