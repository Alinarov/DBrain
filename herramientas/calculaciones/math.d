#!/usr/bin/env dmd
module DBrain.herramientas.calculaciones.math;

import std;

alias print = writeln;

// Función para detectar si una cadena es un número decimal
bool esDecimal(string s) {
    return s.canFind('.');
}

// Función para redondear a 2 decimales (versión para string)
string redondearDosDecimales(string s) {
    import std.format, std.conv;
    double num = to!double(s); // Convierte el string a double
    return format("%.4f", num); // Formatea con 2 decimales
}

// Función para redondear a 2 decimales (versión para float)
float redondearDosDecimales(float s) {
    import std.format, std.conv;
    //double num = to!double(s); // Convierte el float a double
    //print(num);
    return to!float(format("%.4f", s)); // Formatea con 2 decimales
}