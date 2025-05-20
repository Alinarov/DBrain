#!/usr/bin/env dmd
import std.stdio;
import std.json;
import std.algorithm;
import std.array;

void main() {
    // Datos de registros
    auto registros = [
        ["1", "1", "2025-05-18", "1"],
        ["1", "1", "2025-05-19", "2"],
        ["1", "1", "2025-05-20", "3"],
        ["2", "1", "2025-05-21", "9"],
        ["3", "1", "2025-05-18", "11"],
        ["3", "1", "2025-05-19", "12"],
        ["3", "1", "2025-05-20", "13"],
        ["4", "1", "2025-05-18", "16"],
        ["4", "1", "2025-05-19", "17"],
        ["4", "1", "2025-05-20", "18"],
        ["5", "1", "2025-05-18", "21"],
        ["5", "1", "2025-05-19", "22"],
        ["5", "1", "2025-05-20", "23"],
        ["10", "1", "2025-05-18", "46"],
        ["10", "1", "2025-05-19", "47"],
        ["10", "1", "2025-05-20", "48"],
        ["16", "1", "2025-05-20", "78"],
        ["20", "1", "2025-05-18", "96"],
        ["20", "1", "2025-05-19", "97"],
        ["20", "1", "2025-05-20", "98"]
    ];

    // Diccionario para organizar registros por fecha
    JSONValue registrosFecha = JSONValue("{}");

    foreach (dato; registros) {
        string fecha = dato[2]; // Convertir fecha a string
        if ((fecha in registrosFecha.object)) {
        //if (!registrosFecha.object.containsKey(fecha)) {
            registrosFecha.object[fecha] = JSONValue(registrosFecha.array);
        }
        registrosFecha.object[fecha].array ~= JSONValue(dato);
    }

    // Ordenar y mostrar registros organizados por fecha
    auto fechasOrdenadas = registrosFecha.object.byKey.array.sort;
    foreach (fecha; fechasOrdenadas) {
        writeln("Fecha: ", fecha);
        writeln(registrosFecha.object[fecha]);
    }
}
