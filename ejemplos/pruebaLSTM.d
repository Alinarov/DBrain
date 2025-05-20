//import std.stdio;
//import std.math;
//import std.array;
//import std.algorithm;
//import mir.ndslice;
//import mir.random;
//import mir.random.engine;
//import mir.random.variable;

//class LSTM {
//    private:
//        // Parámetros
//        Slice!(double*, 2) Wf, Wi, Wo, Wc, Uf, Ui, Uo, Uc;
//        Slice!(double*, 1) bf, bi, bo, bc;
//        size_t inputSize, hiddenSize;
//        double learningRate = 0.01;

//        // Funciones de activación
//        static auto sigmoid(Slice!(double*, 1) x) {
//            return 1.0 / (1.0 + exp(-x));
//        }

//        static auto tanh(Slice!(double*, 1) x) {
//            return map!(a => std.math.tanh(a))(x);
//        }

//    public:
//        this(size_t inputDim, size_t hiddenDim) {
//            inputSize = inputDim;
//            hiddenSize = hiddenDim;
//            initializeWeights();
//        }

//        void initializeWeights() {
//            auto rng = Random(unpredictableSeed);
//            auto bound = sqrt(6.0 / (inputSize + hiddenSize));
            
//            auto init = (size_t rows, size_t cols) => uniform(rng, -bound, bound).slice!double(rows, cols);
            
//            Wf = init(hiddenSize, inputSize);
//            Wi = init(hiddenSize, inputSize);
//            Wo = init(hiddenSize, inputSize);
//            Wc = init(hiddenSize, inputSize);
            
//            Uf = init(hiddenSize, hiddenSize);
//            Ui = init(hiddenSize, hiddenSize);
//            Uo = init(hiddenSize, hiddenSize);
//            Uc = init(hiddenSize, hiddenSize);
            
//            bf = init(hiddenSize, 1)[0..$, 0];
//            bi = init(hiddenSize, 1)[0..$, 0];
//            bo = init(hiddenSize, 1)[0..$, 0];
//            bc = init(hiddenSize, 1)[0..$, 0];
//        }

//        auto forward(Slice!(double*, 2) input, ref Slice!(double*, 2) h, ref Slice!(double*, 2) c) {
//            auto seqLength = input.shape[0];
//            h[] = 0; c[] = 0; // Reset estados
            
//            foreach (t; 0..seqLength) {
//                auto ft = sigmoid(matvec(Wf, input[t]) + matvec(Uf, h[t]) + bf);
//                auto it = sigmoid(matvec(Wi, input[t]) + matvec(Ui, h[t]) + bi);
//                auto ot = sigmoid(matvec(Wo, input[t]) + matvec(Uo, h[t]) + bo);
//                auto ct_tilde = tanh(matvec(Wc, input[t]) + matvec(Uc, h[t]) + bc);
                
//                c[t + 1] = ft * c[t] + it * ct_tilde;
//                h[t + 1] = ot * tanh(c[t + 1]);
//            }
//            return h[1..seqLength + 1];
//        }

//        void train(Slice!(double*, 2) X_train, Slice!(double*, 1) y_train, size_t epochs) {
//            auto h = new double[X_train.shape[0] + 1][ hiddenSize];
//            auto c = new double[X_train.shape[0] + 1][ hiddenSize];
            
//            foreach (epoch; 0..epochs) {
//                auto output = forward(X_train, h, c);
//                auto loss = mseLoss(output[$ - 1], y_train);
                
//                if (epoch % 100 == 0)
//                    writeln("Época ", epoch, ", Pérdida: ", loss);
                
//                // Backpropagation simplificado
//                backward(X_train, y_train, h, c);
//            }
//        }

//        private:
//            auto matvec(Slice!(double*, 2) m, Slice!(double*, 1) v) {
//                auto result = new double[m.shape[0]];
//                foreach (i; 0..m.shape[0]) {
//                    result[i] = sum(m[i] * v);
//                }
//                return result;
//            }

//            double mseLoss(Slice!(double*, 1) pred, Slice!(double*, 1) target) {
                
//                return mean(map!((a, b) => pow(a - b, 2.0)(pred, target)));

//            }

//            void backward(Slice!(double*, 2) x, Slice!(double*, 1) y, 
//                              Slice!(double*, 2) h, Slice!(double*, 2) c) {
//                // Implementación simplificada de BPTT
//                auto seqLen = x.shape[0];
//                auto dh_next = new double[hiddenSize];
//                auto dc_next = new double[hiddenSize];
                
//                foreach_reverse (t; 0..seqLen) {
//                    // Cálculo de gradientes (versión simplificada)
//                    auto error = h[t + 1] - y;
                    
//                    // Actualización de pesos (SGD)
//                    Wf[] -= learningRate * error[] * x[t];
//                    Uf[] -= learningRate * error[] * h[t];
//                    // ... similar para otros pesos
//                }
//            }
//}

//// Datos de entrenamiento (1234) y prueba (5678)
//void main() {
//    // Datos sintéticos: secuencia 1-8 con ruido
//    auto datos = iota(1, 9).map!(x => x + 0.1 * uniform(-1.0, 1.0)).array;
    
//    // Crear secuencias (ventana=3)
//    auto X = new double[4][3];
//    auto y = new double[4];
    
//    foreach (i; 0..4) {
//        X[i] = datos[i..i+3];
//        y[i] = datos[i+3];
//    }
    
//    // Dividir 1234 (train) y 5678 (test)
//    auto X_train = X[0..2];
//    auto y_train = y[0..2];
//    auto X_test = X[2..$];
//    auto y_test = y[2..$];
    
//    // Crear y entrenar modelo
//    auto lstm = new LSTM(3, 10); // input_size=3, hidden_size=10
//    lstm.train(X_train, y_train, 1000);
    
//    // Prueba
//    writeln("\nResultados Prueba (5678):");
//    auto h_test = new double[X_test.shape[0] + 1][10];
//    auto c_test = new double[X_test.shape[0] + 1][10];
    
//    foreach (i; 0..X_test.shape[0]) {
//        auto pred = lstm.forward(X_test[i..i+1], h_test, c_test);
//        writefln("Entrada: %s, Real: %.2f, Predicción: %.2f", 
//                 X_test[i], y_test[i], pred[0][0]);
//    }
//}