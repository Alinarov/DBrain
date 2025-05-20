//import DBrain.herramientas.calculaciones.estadistica;
//import DBrain.herramientas.obtener_datos;
//import std.algorithm : map, reduce, sum;
//import std.array : array;
//import std.math : sqrt;
//import std.stdio : writeln;
//import std.conv;
//import std.algorithm : map;
//import std.array : array;
//import std.conv : to;
//import std.math : log2;
//import std.stdio : writeln;
//import std;


//void main() {
//    // Inicializar la lista de productos
//    string productos = ["Smartphone", "Tablet", "Monitor", "Mouse", "Altavoz"];

//    // Obtener la lista de productos
//    string[] listaProductos = get_productos();

//    // Calcular el número de bits necesarios
//    int numBits = calcularNumBits(to!int(listaProductos.length));

//    // Codificar cada producto en binario
//    int index = to!int(listaProductos.countUntil("Monitor"));

//    int[] productosBinarios = codificarBinario(index,numBits);


//    int entero = binarioAEntero(productosBinarios);

//    //foreach (i, producto; listaProductos) {
//    //    productosBinarios ~= codificarBinario(to!int(i), numBits);
//    //}

//    // Mostrar resultados
//    writeln("Productos: ", listaProductos);
//    writeln("Codificación binaria: ", productosBinarios);
//    writeln("Codificación binaria entera: ", entero);
//    writeln("Codificación binaria z-score: ", entero);

//}