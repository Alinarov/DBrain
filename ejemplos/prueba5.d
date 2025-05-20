//// Importar módulos necesarios
//import std;
//import DBrain.DBrain;
//import std.random: uniform;
//import core.thread;

//alias print = writeln;

//// Renombramos para que no sea tan largo
//alias funcion_activacion = DBrain.modelos.network_gru.funcion_activacion;
//alias funcion_gradiente = DBrain.modelos.network_gru.funcion_gradiente;

//void main(string[] args) {
//    print("
//  +*****************************+
//+* Predicción GRU AI LEARNING :D *+
//  +*****************************+        ");

//    // Inicializar la red GRU
//    GRUNetwork network = new GRUNetwork(4, [8, 10], 1);

//    funcion_activacion = &DBrain.funciones_activacion.sigmoid;
//    funcion_gradiente = &DBrain.funciones_activacion.tanh_activacion;

//    // Simulación de datos (input temporal y targets)
//    float[][] inputs = [
//        [0.1, 0.2, 0.3, 0.4],
//        [0.2, 0.3, 0.4, 0.5],
//        [0.3, 0.4, 0.5, 0.6],
//    ];

//    float[][] targets = [
//        [0.9],
//        [0.8],
//        [0.7],
//    ];

//    float[][] x_prueba = [
//        [0.4, 0.5, 0.6, 0.7],
//        [0.5, 0.6, 0.7, 0.8],
//        [0.6, 0.7, 0.8, 0.9],
//    ];

//    // Normalizar inputs y x_prueba
//    inputs = inputs.map!(row => row.map!(value => normalize(value, 0.0, 1.0)).array).array;
//    x_prueba = x_prueba.map!(row => row.map!(value => normalize(value, 0.0, 1.0)).array).array;

//    int action = 0;  // Acción simulada
//    float reward = 1.0;  // Recompensa inicial
//    float[] nextState = [0.5, 0.6, 0.7, 0.8];  // Estado futuro simulado

//    network.loadModel("DBrain/experiencias/gru_model.memoria");
//    // Función de entrenamiento
//    void reentreno() {
//        foreach (epoch; 0 .. 20000) {
//            foreach (i, input; inputs) {
//                float[] prediction = network.forward(input);
//                network.learn(input, targets[i]);  // Aprendizaje supervisado
//            }
//        }
//    }

//    //reentreno();  // Entrenar antes de las pruebas

//    // Función para actualizar valores Q
//    void actualizar_q_values() {
//        foreach (i, input; x_prueba) {
//            network.updateQValues(input, action, reward, nextState);
//        }
//    }

//    //actualizar_q_values(); // Primera actualización de valores Q

//    float epsilon = 0.001;

//    float[][] expected_results = [
//        [0.6],  // Resultado esperado para x_prueba[0]
//        [0.7],  // Resultado esperado para x_prueba[1]
//        [0.8],  // Resultado esperado para x_prueba[2]
//    ];

//    // Mostrar resultados de predicciones
//    print("Resultados de las predicciones en datos de prueba:");
//    foreach (i, input; x_prueba) {
//        float[] prediccion_normalizada = network.forward(input);
//        float[] prediccion_real = desnormalizeArray(prediccion_normalizada, 0.0, 1.0); // Desnormalizar la predicción

//        print("Entrada ", i, ": ", input);
//        print("Predicción normalizada ", i, ": ", prediccion_normalizada);
//        print("Predicción desnormalizada ", i, ": ", prediccion_real);

//        float error = abs(prediccion_real[0] - expected_results[i][0]); // Comparar con resultado esperado desnormalizado

//        if (error < epsilon) {
//            print("Predicción aceptable para la entrada ", i);
//            reward = 1.0;  // Recompensa alta si la predicción es buena
//        } else {
//            print("Predicción fuera de rango para la entrada ", i);
//            reward = 1.0 - error;  // Ajustar recompensa según el error
//        }

//        actualizar_q_values();
//    }

//    network.saveModel("DBrain/experiencias/gru_model.memoria");

//    print("Proceso completado.");
//}
