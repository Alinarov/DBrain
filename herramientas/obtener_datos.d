#!/usr/bin/env dmd
module DBrain.herramientas.obtener_datos;
import std;

/* ----------------------- */
import DBrain.herramientas.calculaciones.math;
/* ----------------------- */
alias print = writeln;

string[][] get_datos_csv (string _archivo) {
    string ruta_archivo = "datos_entrada/"~ _archivo;

    // Abrir el archivo en modo lectura
    File archivo = File(ruta_archivo, "r");

    // Declarar el arreglo para almacenar los datos cargados
    string[][] datos_cargados;

    // Leer el contenido del archivo
    auto contenido = archivo.byLineCopy;

    // Procesar cada línea del archivo CSV
    foreach (linea; contenido) {
        // Dividir la línea en campos y almacenarla como un arreglo
        auto rango = linea.splitter(',')            
            .map!(c => c.strip())
            .map!(c => esDecimal(c) ? redondearDosDecimales(c) : c)
            .array;
            
        // Agregar la línea procesada al arreglo
        datos_cargados ~= rango;
    }
    //construir el modelo predictivo, basado en redes neuronales, para el desarroollo del proyecto
    //modelo predictivo de ventas para la empresa compupluss
    
    // Imprimir el contenido cargado
    //print(datos_cargados);	
    return datos_cargados;
}

// Aqui filtramos los datos de la columna 
string[] columna_filtrada (int index_columna, string [][] datos_cargados) {
    string[] datos_filtrados;

    foreach (i, ref datos; datos_cargados) {
        if(i == 0) continue;
        
        if (!datos_filtrados.canFind(datos[index_columna])){
            datos_filtrados~= datos[index_columna];
        }
        
    }
    return datos_filtrados;
}