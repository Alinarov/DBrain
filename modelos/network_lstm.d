#!/usr/bin/env dmd 
// Linea 300: network
module DBrain.modelos.network_lstm;
import std;

import DBrain.DBrain;

/**
 * LSTM
 */

alias print = writeln;

class LSTM_celda {
	float function (float x) funcion_activacion;
	float function (float x) funcion_hiperbolica;
	float function (float x) funcion_derivada;

protected:

	float peso_olvido;
	float peso_entrada;
	float peso_candidato;
	float peso_salida;

	float ponderados_olvido;
	float ponderados_entrada;
	float ponderados_candidato;
	float ponderados_salida;

	float sesgo_olvido;
	float sesgo_entrada;
	float sesgo_salida;
	float sesgo_celda;
	
	float memoria_largo_plazo;
	float memoria_corto_plazo;
	
	float entrada;

	// Tasas de aprendizajes 
	float tasa_aprendizaje = 0.01f;
	float min_tasa = 0.0001f;
	float factor_decaimiento = 0.98f;
public: 
	this(
	    float[4] pesos = [uniform(-0.2f, 0.2f), uniform(-0.2f, 0.2f), uniform(-0.2f, 0.2f), uniform(-0.2f, 0.2f)],
	    float[4] valores_ponderados = [uniform(-0.2f, 0.2f), uniform(-0.2f, 0.2f), uniform(-0.2f, 0.2f), uniform(-0.2f, 0.2f)],
	    float entrada = 0.0f  // Opcional: valor inicial para el primer paso
	){

		// configuracion de pesos (red rojo, rosado)
		this.peso_olvido = pesos[0];
		this.peso_entrada = pesos[1];
		this.peso_candidato = pesos[2];
		this.peso_salida = pesos[3];

		// configuracion valores ponderados (gris plomo)
		this.ponderados_olvido = valores_ponderados[0];
		this.ponderados_entrada = valores_ponderados[1];
		this.ponderados_candidato = valores_ponderados[2];
		this.ponderados_salida = valores_ponderados[3];

		// sesgos 
		//float bound = sqrt(6.0f / (input_size + hidden_size)); 
		this.sesgo_olvido = 1.0f; // uniform(-bound, bound)
		this.sesgo_entrada = 0.0f;
		this.sesgo_salida = 0.0f;
		this.sesgo_celda = 0.0f;

		// estado de celda (green verde)
		this.memoria_largo_plazo = 0.0f;

		// estado oculto
		this.memoria_corto_plazo = 0.0f;
		this.entrada = entrada;
		
		
	}	

	// entrada_ : numero del patron que queremos probar
	// celda_anterior : es el resultado de la celda lstm anterior
	@trusted public float fordward  (float entrada_, float celda_anterior) {

		// 1 primera etapa
		float puerta_olvido = etapa_olvido(entrada_, celda_anterior);
		// 2 segunda etapa
		float puerta_entrada = etapa_entrada(entrada_, celda_anterior);
		// 3 tercera etapa
		float puerta_candidato = etapa_celda_candidato(entrada_, celda_anterior);
		// 3.5 actualizacion de la memoria largo plazo
		memoria_largo_plazo = puerta_olvido * memoria_largo_plazo + puerta_entrada * puerta_candidato;
		// 4 cuarta etapa
		float puerta_salida = etapa_salida(entrada_, celda_anterior);
		// 5 actualizar la memoria corto plazo
		memoria_corto_plazo = puerta_salida * funcion_hiperbolica(memoria_largo_plazo);

		return memoria_corto_plazo;
	}


	// 1. Primera etapa de la celda 
	@trusted float etapa_olvido (float entrada_, float celda_anterior) {
		memoria_corto_plazo = celda_anterior;
		float sumatoria = (peso_olvido * entrada_) + ( ponderados_olvido * memoria_corto_plazo) + sesgo_olvido;
		float porcentaje_memoria_largo_plazo = funcion_activacion(sumatoria);			
		return porcentaje_memoria_largo_plazo;

	}

	@trusted float etapa_entrada  (float entrada_, float celda_anterior) {
		memoria_corto_plazo = celda_anterior;
		float sumatoria = (peso_entrada * entrada_) + (ponderados_entrada * memoria_corto_plazo) + sesgo_entrada;
		float porcentaje_memoria_largo_plazo = funcion_activacion(sumatoria);
		return porcentaje_memoria_largo_plazo;

	}

	@trusted float etapa_celda_candidato  (float entrada_, float celda_anterior) {
		memoria_corto_plazo = celda_anterior;
		float sumatoria = (peso_candidato * entrada_) + (ponderados_candidato * memoria_corto_plazo) + sesgo_celda;
		float porcentaje_memoria_largo_plazo = funcion_hiperbolica(sumatoria);
		return porcentaje_memoria_largo_plazo;

	}

	@trusted float etapa_salida  (float entrada_, float celda_anterior) {
		memoria_corto_plazo = celda_anterior;
		float sumatoria = (peso_salida * entrada_) + (ponderados_salida * memoria_corto_plazo) + sesgo_salida;
		float porcentaje_memoria_largo_plazo = funcion_activacion(sumatoria);
		return porcentaje_memoria_largo_plazo;
	}

	@trusted float entrenar(LSTM_celda lstm, float[] secuencia, float[] objetivos) {
	    float[] salidas;          // Almacena las salidas de cada paso forward
	    float estado_anterior = 0.0f;  // Estado inicial (h_{t-1})

	    // --- Paso Forward (Propagación hacia adelante) ---
	    foreach (x; secuencia) {
	        salidas ~= lstm.fordward(x, estado_anterior);  // Calcula h_t
	        estado_anterior = salidas.back;                // Actualiza h_{t-1} para el siguiente paso
	    }

	    // --- Paso Backward (Retropropagación) ---
	    float gradiente = 0.0f;  // Gradiente acumulado desde el futuro (inicialmente 0)

	    foreach_reverse (t, ref dato; secuencia) {
	        // 1. Calcula el error en el paso t
	        float error = salidas[t] - objetivos[t];

	        // 2. Retropropaga el error (y el gradiente del futuro)
	        gradiente = lstm.retropropagar(
	            error,                     // Error actual
	            secuencia[t],              // Entrada x_t
	            (t > 0) ? salidas[t-1] : 0.0f,  // h_{t-1} (o 0 si es el primer paso)
	            gradiente                  // Gradiente desde t+1
	        );

	        // Nota: `dato` no se usa aquí, pero está disponible si necesitas modificar la secuencia

	    }
		return gradiente;
	}

    @trusted float retropropagar(float error, float entrada_, float celda_anterior, float gradiente_siguiente = 0.0f) {
		float puerta_olvido = etapa_olvido(entrada_, celda_anterior);
		float puerta_entrada = etapa_entrada(entrada_, celda_anterior);
		float puerta_candidato = etapa_celda_candidato(entrada_, celda_anterior);
		float puerta_salida = etapa_salida(entrada_, celda_anterior);

        // 1. Gradientes locales
        float gradiente_salida = error + gradiente_siguiente;
        float gradiente_estado_celda = gradiente_salida * puerta_salida * funcion_derivada(memoria_largo_plazo);


        // 2. Gradientes de las puertas

        float gradiente_puerta_olvido = gradiente_estado_celda * celda_anterior * funcion_activacion(puerta_olvido);
        float gradiente_puerta_entrada = gradiente_estado_celda * puerta_candidato * funcion_activacion(puerta_entrada);
        float gradiente_candidato = gradiente_estado_celda * puerta_entrada * funcion_derivada(puerta_candidato);
        float gradiente_puerta_salida = gradiente_salida * funcion_hiperbolica(memoria_largo_plazo) * funcion_derivada(puerta_salida);
		
		// Gradient clipping para evitar exploding:
	    gradiente_puerta_salida = max(min(gradiente_puerta_salida, 1.0f), -1.0f);
	    gradiente_puerta_olvido = max(min(gradiente_puerta_olvido, 1.0f), -1.0f);
	    gradiente_puerta_entrada = max(min(gradiente_puerta_entrada, 1.0f), -1.0f);
	    gradiente_candidato = max(min(gradiente_candidato, 1.0f), -1.0f);


        // 3. Actualizar pesos y sesgos
        tasa_aprendizaje *= 0.95;
        tasa_aprendizaje = max(tasa_aprendizaje, 0.0001); // Evita que llegue a 0

	    ////if (pérdida_actual > pérdida_anterior) {
	    ////    tasa_aprendizaje *= factor_decaimiento;
	    ////}
	    //tasa_aprendizaje = max(tasa_aprendizaje, min_tasa);


        peso_salida -= tasa_aprendizaje * gradiente_puerta_salida * entrada_;
        ponderados_salida -= tasa_aprendizaje * gradiente_puerta_salida * memoria_corto_plazo;
        sesgo_salida -= tasa_aprendizaje * gradiente_puerta_salida;

        peso_olvido -= tasa_aprendizaje * gradiente_puerta_olvido * entrada_;
        ponderados_olvido -= tasa_aprendizaje * gradiente_puerta_olvido * memoria_corto_plazo;
        sesgo_olvido -= tasa_aprendizaje * gradiente_puerta_olvido;

        // Añade esto después de actualizar peso_olvido:
		peso_entrada -= tasa_aprendizaje * gradiente_puerta_entrada * entrada_;
		ponderados_entrada -= tasa_aprendizaje * gradiente_puerta_entrada * memoria_corto_plazo;
		sesgo_entrada -= tasa_aprendizaje * gradiente_puerta_entrada;

		peso_candidato -= tasa_aprendizaje * gradiente_candidato * entrada_;
		ponderados_candidato -= tasa_aprendizaje * gradiente_candidato * memoria_corto_plazo;
		sesgo_celda -= tasa_aprendizaje * gradiente_candidato;

        // 4. Propagar gradiente al paso anterior
        float gradiente_anterior = 
		    (gradiente_salida * ponderados_salida * funcion_derivada(puerta_salida)) +
		    (gradiente_estado_celda * ponderados_olvido * funcion_derivada(puerta_olvido)) +
		    (gradiente_estado_celda * ponderados_entrada * funcion_derivada(puerta_entrada)) +
		    (gradiente_estado_celda * ponderados_candidato * funcion_derivada(puerta_candidato));

        return gradiente_anterior;
    }

	void cargarParametros(ref LSTM_celda lstm, string nombreArchivo = "lstm_params.json") {
	    string contenido = readText(nombreArchivo);
	    auto parametros = parseJSON(contenido);

	    // Extraer valores numéricos del JSONValue
	    lstm.peso_olvido = parametros["pesos"]["peso_olvido"].floating.to!float;
	    lstm.peso_entrada = parametros["pesos"]["peso_entrada"].floating.to!float;
	    lstm.peso_candidato = parametros["pesos"]["peso_candidato"].floating.to!float;
	    lstm.peso_salida = parametros["pesos"]["peso_salida"].floating.to!float;

	    lstm.ponderados_olvido = parametros["ponderados"]["ponderados_olvido"].floating.to!float;
	    lstm.ponderados_entrada = parametros["ponderados"]["ponderados_entrada"].floating.to!float;
	    lstm.ponderados_candidato = parametros["ponderados"]["ponderados_candidato"].floating.to!float;
	    lstm.ponderados_salida = parametros["ponderados"]["ponderados_salida"].floating.to!float;

	    lstm.sesgo_olvido = parametros["sesgos"]["sesgo_olvido"].floating.to!float;
	    lstm.sesgo_entrada = parametros["sesgos"]["sesgo_entrada"].floating.to!float;
	    lstm.sesgo_salida = parametros["sesgos"]["sesgo_salida"].floating.to!float;
	    lstm.sesgo_celda = parametros["sesgos"]["sesgo_celda"].floating.to!float;

	    lstm.memoria_largo_plazo = parametros["estados"]["memoria_largo_plazo"].floating.to!float;
	    lstm.memoria_corto_plazo = parametros["estados"]["memoria_corto_plazo"].floating.to!float;

	    writeln("Parámetros cargados desde: ", nombreArchivo);
	}

	void guardarParametros(LSTM_celda lstm, string nombreArchivo = "lstm_params.json") {
	    // Crear un struct JSON con todos los parámetros
	    auto parametros = JSONValue(
	        [
	            "pesos": JSONValue(
	                [
	                    "peso_olvido": JSONValue(lstm.peso_olvido),
	                    "peso_entrada": JSONValue(lstm.peso_entrada),
	                    "peso_candidato": JSONValue(lstm.peso_candidato),
	                    "peso_salida": JSONValue(lstm.peso_salida)
	                ]
	            ),
	            "ponderados": JSONValue(
	                [
	                    "ponderados_olvido": JSONValue(lstm.ponderados_olvido),
	                    "ponderados_entrada": JSONValue(lstm.ponderados_entrada),
	                    "ponderados_candidato": JSONValue(lstm.ponderados_candidato),
	                    "ponderados_salida": JSONValue(lstm.ponderados_salida)
	                ]
	            ),
	            "sesgos": JSONValue(
	                [
	                    "sesgo_olvido": JSONValue(lstm.sesgo_olvido),
	                    "sesgo_entrada": JSONValue(lstm.sesgo_entrada),
	                    "sesgo_salida": JSONValue(lstm.sesgo_salida),
	                    "sesgo_celda": JSONValue(lstm.sesgo_celda)
	                ]
	            ),
	            "estados": JSONValue(
	                [
	                    "memoria_largo_plazo": JSONValue(lstm.memoria_largo_plazo),
	                    "memoria_corto_plazo": JSONValue(lstm.memoria_corto_plazo)
	                ]
	            )
	        ]
	    );

	    // Guardar en archivo
	    File archivo = File(nombreArchivo, "w");
	    archivo.writeln(parametros.toPrettyString());
	    archivo.close();
	    writeln("Parámetros guardados en: ", nombreArchivo);
	}
}


/**
 * ************************************************************************************************************
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * ************************************************************************************************************
 * */

class Network_LSTM {

    // Configuración arquitectura_redl
    size_t tamanoEntrada;
	size_t[] arquitectura_red;
    LSTM_celda[][] celdas;

    // Configuración de entrenamiento
    float tasaAprendizaje;
    float tasaDecaimiento;
    float mejorError = float.max;
    int sinMejoria = 0;
    int maxEpochs = 10_000;
    int paciencia = 500;


    this(size_t tamanoEntrada, size_t[] arquitectura_red, float tasaAprendizaje = 0.01f, float tasaDecaimiento = 0.95f) {
    
        this.tamanoEntrada = tamanoEntrada;
        arquitectura_red = arquitectura_red;
        this.tasaAprendizaje = tasaAprendizaje;
        this.tasaDecaimiento = tasaDecaimiento;

        this.crear_red();
    }


    
    void crear_red() {

	    // Inicializar todas las celdas LSTM
	    celdas = new LSTM_celda[][](arquitectura_red.length);

		foreach (i, capaSize; arquitectura_red) {
		    celdas[i] = new LSTM_celda[capaSize];		    
		    foreach (j; 0 .. capaSize) {
				// 2. Inicialización LSTM con Xavier
		        celdas[i][j] = new LSTM_celda(
		            pesos: [xavierInit(), xavierInit(), xavierInit(), xavierInit()],
		            valores_ponderados: [xavierInit(), xavierInit(), xavierInit(), xavierInit()],
		            0.0f  // valor inicial de entrada
		        );
				// Definimos las funciones de activacion y de ipervolica o gradiente
				celdas[i][j].funcion_activacion = &DBrain.funciones_activacion.sigmoid;
				celdas[i][j].funcion_hiperbolica = &DBrain.funciones_activacion.tanh_hiperbolica;
				celdas[i][j].funcion_derivada = &DBrain.funciones_activacion.tanh_derivada;
		    }
		}


    }

    void entrenar(float[][] secuencias, float[][] objetivos, int maxEpochs = 10000, int paciencia = 500) {
        writeln("Iniciando entrenamiento...");
        
        foreach (epoch; 0 .. maxEpochs) {
            float errorTotal = 0.0f;
            
            foreach (idx, secuencia; secuencias) {
                // Forward pass
                float[] estadosOcultos = forwardPass(secuencia);
				
				error = 0.0f; // Reiniciar el error para cada secuencia
				size_t inicio = estadosOcultos.length - arquitectura_red[$-1]; // Calcula el inicio una vez

				foreach (i; 0 .. arquitectura_red[$-1]) { // Itera solo sobre las salidas relevantes
				    float diferencia = estadosOcultos[inicio + i] - objetivos[idx][i];
				    error += diferencia * diferencia; // Más eficiente que pow() para cuadrados
				}
				error /= arquitectura_red[$-1]; // Normalizar por cantidad de salidas               
                
                // Backward pass
                backwardPass(secuencia, objetivos[idx]);
            }
            
            errorTotal /= secuencias.length;
            
            // Early stopping
            if (errorTotal < mejorError) {
                mejorError = errorTotal;
                sinMejoria = 0;
            } else {
                sinMejoria++;
                if (sinMejoria > paciencia) {
                    writeln("Early stopping en época ", epoch);
                    break;
                }
            }
            
            if (epoch % 1000 == 0) {
                writeln("Época ", epoch, " - Error: ", errorTotal);
            }
        }
    }

    float[] forwardPass(float[] entrada) {
        float[] salida;
        float[] estadoOculto = new float[arquitectura_red[$-1]];
        float[] estadoCelda = new float[arquitectura_red[$-1]];
        
        foreach (t, x; entrada) {
            for (int i = 0; i < arquitectura_red.length; i++) {
                for (int j = 0; j < arquitectura_red[i]; j++) {
                    if (i == 0) {
                        estadoOculto[j] = celdas[i][j].forward(x, estadoCelda[j]);
                    } else {
                        estadoOculto[j] = celdas[i][j].forward(estadoOculto[j], estadoCelda[j]);
                    }
                    estadoCelda[j] = celdas[i][j].memoria_largo_plazo;
                }
            }
            salida ~= estadoOculto[$-1];
        }
        return salida;
    }

    void backwardPass(float[] entrada, float[] objetivo) {
        float[] gradienteSiguiente = new float[arquitectura_red[$-1]];
        
        for (int t = entrada.length - 1; t >= 0; t--) {
            for (int i = arquitectura_red.length - 1; i >= 0; i--) {
                for (int j = 0; j < arquitectura_red[i]; j++) {
                    gradienteSiguiente[j] = celdas[i][j].retropropagar(
                        objetivo[t] - salida[t], // Error
                        (i == 0) ? entrada[t] : estadoOcultoAnterior[i-1][j],
                        estadoCeldaAnterior[i][j],
                        gradienteSiguiente[j]
                    );
                }
            }
        }
    }

    float[] predecir(float[] semilla, int pasos) {
        float[] predicciones;
        float[] estadoActual = semilla.dup;
        
        foreach (_; 0 .. pasos) {
            estadoActual = forwardPass(estadoActual);
            predicciones ~= estadoActual[$-1];
            estadoActual = estadoActual[$-arquitectura_red[$-1] .. $];
        }
        
        return predicciones;
    }

    float[] processTimeStep(float[] input) {
        // Procesar un paso temporal en la secuencia
        float[] output;
        
        // Con LSTM
        foreach (cell; lstmLayer) {
            output = cell.forward(input);
        }
        // O con 
        // foreach (cell; Layer) {
        //    output = cell.forward(input);
        // }
        
        return output;
    }
    


	void resetStates() {
	    foreach (ref capa; celdas) {
	        foreach (ref celda; capa) {
	            celda.memoria_largo_plazo = 0.0f;
	            celda.memoria_corto_plazo = 0.0f;
	        }
	    }
	}

	// Llamar esto antes de cada nueva secuencia en `entrenar`
	float xavierInit() {
	    float bound = sqrt(6.0f / (this.tamanoEntrada + arquitectura_red[0]));  // Ajusta inputSize/hiddenSize
	    return uniform(-bound, bound);
	}

}


