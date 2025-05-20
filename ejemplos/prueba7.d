//import std;
//import DBrain.DBrain;

//alias print = writeln;

//void main () {

//	Network_LSTM n = new Network_LSTM([3,4,6]);


//}

//class Network_LSTM {

//    // Configuración arquitectura_redl
    
//	size_t[] arquitectura_red;
//    LSTM_celda[][] celdas;

//    // Configuración de entrenamiento
    
//    float tasaDecaimiento;
//    float mejorError = float.max;
//    int sinMejoria = 0;
//    int maxEpochs = 10_000;
//    int paciencia = 500;


//    this(size_t[] arquitectura, float tasaDecaimiento = 0.95f) {
    
        
//        this.arquitectura_red = arquitectura;
        
//        this.tasaDecaimiento = tasaDecaimiento;

//        this.crear_red();
//    }


    
//    void crear_red() {

//	    // Inicializar todas las celdas LSTM
//	    celdas = new LSTM_celda[][](arquitectura_red.length);

//		foreach (i, capaSize; arquitectura_red) {
//		    celdas[i] = new LSTM_celda[capaSize];		    
//		    foreach (j; 0 .. capaSize) {
//				// 2. Inicialización LSTM con Xavier
//		        celdas[i][j] = new LSTM_celda(
//		            pesos: [1,1,1,1],
//		            valores_ponderados: [1,1,1,1],
//		            0.0f  // valor inicial de entrada
//		        );
//				// Definimos las funciones de activacion y de ipervolica o gradiente
//				celdas[i][j].funcion_activacion = &DBrain.funciones_activacion.sigmoid;
//				celdas[i][j].funcion_hiperbolica = &DBrain.funciones_activacion.tanh_hiperbolica;
//				celdas[i][j].funcion_derivada = &DBrain.funciones_activacion.tanh_derivada;
//		    }
//		}


//    }

//}


