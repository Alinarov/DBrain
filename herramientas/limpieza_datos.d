#!/usr/bin/env dmd
module DBrain.herramientas.limpieza_datos;

import std;

alias print = writeln;


float[] limpiar_nan (float[] datos_input) {
    foreach(i, ref dato; datos_input) {
        if(isNaN(dato)) datos_input[i] = 0.0;
    }
    
    return datos_input;
}


