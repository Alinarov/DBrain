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


	// Esto no es parte de la celda mas bien lo necesito aqui para que pueda funcionar el 
	// network
	float gradiente_puerta_olvido;
	float gradiente_puerta_entrada;
	float gradiente_candidato;
	float gradiente_puerta_salida;

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
		//print("lstm ");
		//print("gradiente_salida ", gradiente_salida);
		//print("puerta_salida ", puerta_salida);
		//print("memoria_largo_plazo ", memoria_largo_plazo);


		///* Aqui tiene error el violacion de core en funcion_devirada */
		//print("funcion_derivada(memoria_largo_plazo) ", funcion_derivada(memoria_largo_plazo));
		float gradiente_estado_celda = gradiente_salida * puerta_salida * funcion_derivada(memoria_largo_plazo);


		// 2. Gradientes de las puertas

		gradiente_puerta_olvido = gradiente_estado_celda * celda_anterior * funcion_activacion(puerta_olvido);
		gradiente_puerta_entrada = gradiente_estado_celda * puerta_candidato * funcion_activacion(puerta_entrada);
		gradiente_candidato = gradiente_estado_celda * puerta_entrada * funcion_derivada(puerta_candidato);
		gradiente_puerta_salida = gradiente_salida * funcion_hiperbolica(memoria_largo_plazo) * funcion_derivada(puerta_salida);
		
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



	@trusted void actualizarPesos(float tasa) {
		// Aplicar decaimiento a la tasa de aprendizaje
		tasa_aprendizaje = tasa * factor_decaimiento;
		tasa_aprendizaje = max(tasa_aprendizaje, min_tasa);
		
		// Actualización vectorizada de pesos (más eficiente)
		float[4] gradientes = [
			gradiente_puerta_olvido,
			gradiente_puerta_entrada,
			gradiente_candidato,
			gradiente_puerta_salida
		];
		
		float[4] pesos = [peso_olvido, peso_entrada, peso_candidato, peso_salida];
		float[4] ponderados = [ponderados_olvido, ponderados_entrada, ponderados_candidato, ponderados_salida];
		float[4] sesgos = [sesgo_olvido, sesgo_entrada, sesgo_celda, sesgo_salida];
		
		foreach (i; 0..4) {
			pesos[i] -= tasa_aprendizaje * gradientes[i] * entrada;
			ponderados[i] -= tasa_aprendizaje * gradientes[i] * memoria_corto_plazo;
			sesgos[i] -= tasa_aprendizaje * gradientes[i];
		}
		
		// Asignar de vuelta los valores (D no soporta tuplas completamente)
		peso_olvido = pesos[0];
		peso_entrada = pesos[1];
		peso_candidato = pesos[2];
		peso_salida = pesos[3];

		ponderados_olvido = ponderados[0];
		ponderados_entrada = ponderados[1];
		ponderados_candidato = ponderados[2];
		ponderados_salida = ponderados[3];
		
		sesgo_olvido = sesgos[0];
		sesgo_entrada = sesgos[1];
		sesgo_salida = sesgos[2];
		sesgo_celda = sesgos[3];
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

	float[][] estadosOcultos;
	float[][] estadosCelda;

	

	this(size_t tamanoEntrada, size_t[] arquitectura_red, float tasaAprendizaje = 0.01f, float tasaDecaimiento = 0.95f) {
	
		this.tamanoEntrada = tamanoEntrada;
		this.arquitectura_red = arquitectura_red;
		
		this.tasaAprendizaje = tasaAprendizaje;
		this.tasaDecaimiento = tasaDecaimiento;
		this.estadosOcultos = new float[][](arquitectura_red.length, arquitectura_red[$-1]);
		
		this.estadosCelda = new float[][](arquitectura_red.length, arquitectura_red[$-1]);
		

		this.crear_red();
	}


	
	void crear_red() {

		// Inicializar todas las celdas LSTM
		celdas = new LSTM_celda[][](arquitectura_red.length);
		
		foreach (i, capaSize; arquitectura_red) {
			celdas[i] = new LSTM_celda[capaSize];		    

			foreach (j; 0 .. capaSize) {
		        ulong inputSize_ulong = (i == 0) ? tamanoEntrada : arquitectura_red[i-1];
		        ulong outputSize_ulong = arquitectura_red[i];
		        	
		        int inputSize = to!int(inputSize_ulong);
		        int outputSize = to!int(outputSize_ulong);


				// 2. Inicialización LSTM con Xavier
				celdas[i][j] = new LSTM_celda(
					pesos: [xavierInit(inputSize, outputSize), xavierInit(inputSize, outputSize), xavierInit(inputSize, outputSize), xavierInit(inputSize, outputSize)],
					valores_ponderados: [xavierInit(inputSize, outputSize), xavierInit(inputSize, outputSize), xavierInit(inputSize, outputSize), xavierInit(inputSize, outputSize)],
					0.0f  // valor inicial de entrada
				);
				// Definimos las funciones de activacion y de ipervolica o gradiente
				celdas[i][j].funcion_activacion = &DBrain.funciones_activacion.sigmoid_estable;
				celdas[i][j].funcion_hiperbolica = &DBrain.funciones_activacion.tanh_estable;
				celdas[i][j].funcion_derivada = &DBrain.funciones_activacion.tanh_derivada_estable;
			}
		}


	}

	void entrenar(float[][] secuencias, float[][] objetivos, int maxEpochs = 10000, int paciencia = 500) {
		writeln("Iniciando entrenamiento...");
		
		float error = 0.0f;

		foreach (epoch; 0 .. maxEpochs) {
			float errorTotal = 0.0f;
			
			foreach (idx, secuencia; secuencias) {
			  
				// Forward pass / salida
				//estadosOcultos = forwardPass(secuencia);
				float[] estadosOcultos_buffer = forwardPass(secuencia);
				
				error = 0.0f; // Reiniciar el error para cada secuencia
				size_t inicio = estadosOcultos_buffer.length - arquitectura_red[$-1]; // Calcula el inicio una vez

				foreach (i; 0 .. arquitectura_red[$-1]) { // Itera solo sobre las salidas relevantes
					float diferencia = estadosOcultos_buffer[inicio + i] - objetivos[idx][i];
					error += diferencia * diferencia; // Más eficiente que pow() para cuadrados
				}
				error /= arquitectura_red[$-1]; // Normalizar por cantidad de salidas               
				
				// Backward pass
				print("entrenar " , error);
				backwardPass(secuencia, objetivos[idx], estadosOcultos_buffer);
			}
			print("dsojnwioeun");
			errorTotal /= secuencias.length;
			print("errorTotal ", errorTotal);
			print("epoch ", epoch);
			
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
		foreach (t, x; entrada) {
			for (int i = 0; i < arquitectura_red.length; i++) {
				for (int j = 0; j < arquitectura_red[i]; j++) {
					float entradaActual = (i == 0) ? x : this.estadosOcultos[i-1][j];
					//print("forwardPass ", this.estadosOcultos);
					//print("length estadosOcultos ", this.estadosOcultos.length);
					//print("i ", i);
					//print("j ", j);
					try {
						this.estadosOcultos[i][j] = celdas[i][j].fordward(
							entradaActual,
							estadosCelda[i][j]
						);
						estadosCelda[i][j] = celdas[i][j].memoria_largo_plazo;

					} catch (core.exception.ArrayIndexError e) {	
						print(e.msg);
					}
				}
			}
			salida ~= this.estadosOcultos[$-1].dup;
		}
		print("forwardPass ", salida);
		print("forwardPass ", salida.length);
		return salida;
	}

	void backwardPass(float[] entrada, float[] objetivo, float[] salida) {
	    writeln("Iniciando backwardPass...");
	    
	    // 1. Inicialización más segura de gradientes
	    // aqui convertimos los NAN a un 0 para que se pueda calcular
	    float[][] gradientes = new float[][](arquitectura_red.length);
	    foreach(i, ref capa; gradientes) {
	        capa = new float[arquitectura_red[i]];
	        capa[] = 0.0f; // Inicialización explícita
	    } 

	    


	    // 2. Retropropagación con foreach_reverse para mantener el orden
	    foreach_reverse (t; 0 .. min(entrada.length, objetivo.length, salida.length)) {
	        foreach_reverse (i; 0 .. arquitectura_red.length) {

	            foreach (j; 0 .. arquitectura_red[i]) {
	            	print(j);
	            	print(arquitectura_red[i]);
	                // 3. Cálculo seguro del error
	                float error = calcularError(t, i, j, objetivo, salida, gradientes);


	                print( "backwardPass ", error);
	                // 4. Verificación de NaN y límites
	                if (isNaN(error)) {
	                    error = 0.0f;
	                    print("Advertencia: NaN detectado en capa ", i, " neurona ", j);
	                } 
	                print("backwardPass i ", i);
	                print("backwardPass j ", j);
	                // 5. Retropropagación segura
	                if (i < celdas.length && j < celdas[i].length) {
	                	print(" if ", i < celdas.length && j < celdas[i].length);
	                	print("if gradientes[i][j] ", gradientes[i][j]);
	                	print("celdas[i][j] ", celdas[i][j]);
	                    gradientes[i][j] = celdas[i][j].retropropagar(
	                        error,
	                        (i == 0) ? entrada[t] : 
	                        (i > 0 && j < estadosOcultos[i-1].length) ? estadosOcultos[i-1][j] : 0.0f,
	                        (i < estadosCelda.length && j < estadosCelda[i].length) ? estadosCelda[i][j] : 0.0f,
	                        gradientes[i][j]
	                    );

	                    print("if gradientes[i][j] ", gradientes[i][j]);

	                }

	                print(" _----------- foreasch ");
	            }


	        }
	    }

	    // 6. Actualización de pesos con foreach (versión simplificada)
	    void actualizarPesosCapa(LSTM_celda[] capa, float tasa, float decaimiento) {
	        foreach (ref celda; capa) {
	            celda.actualizarPesos(tasa);
	            
	            // Regularización L2 con verificación
	            if (decaimiento > 0) {
	                celda.peso_olvido = isNaN(celda.peso_olvido) ? 0.0f : celda.peso_olvido * (1 - decaimiento);
	                celda.peso_entrada = isNaN(celda.peso_entrada) ? 0.0f : celda.peso_entrada * (1 - decaimiento);
	                celda.peso_candidato = isNaN(celda.peso_candidato) ? 0.0f : celda.peso_candidato * (1 - decaimiento);
	                celda.peso_salida = isNaN(celda.peso_salida) ? 0.0f : celda.peso_salida * (1 - decaimiento);
	            }
	        }
	    }

	    // Actualizar todas las capas
	    foreach (ref capa; celdas) {
	        actualizarPesosCapa(capa, tasaAprendizaje, tasaDecaimiento);
	    }

	    debug writeln("BackwardPass completado");
	}


	//void mostrarPesosIniciales() {
	//    writeln("=== Pesos Iniciales ===");
	//    foreach(i, capa; celdas) {
	//        writeln("Capa ", i, ":");
	//        foreach(j, celda; capa) {
	//            writeln("  Neurona ", j, ":");
	//            writeln("    Pesos: ", celda.pesos);
	//            writeln("    Ponderados: ", celda.valores_ponderados);
	//        }
	//    }
	//}


	float[] predecir(float[] semilla, int pasos) {
		float[] predicciones;
		float[] estadoActual = semilla.dup;
		
		// Procesar semilla inicial
		forwardPass(estadoActual);
		
		// Predecir pasos futuros
		foreach (_; 0 .. pasos) {
			float ultimoValor = estadosOcultos[$-1][$-1];
			predicciones ~= ultimoValor;
			
			// Usar predicción como nueva entrada
			forwardPass([ultimoValor]);
		}
		
		return predicciones;



	    // "Calentar" la LSTM con la secuencia completa
	    //float estado = 0.0f;
	    //foreach (x; secuencia_norm) {
	    //    estado = lstm.fordward(x, estado);
	    //}
	    
	    //// Predicciones
	    //foreach (i; 0..pasos) {
	    //    estado = lstm.fordward(ultimo_valor, estado);
	    //    float prediccion = round(estado) * 10.0f; // Desnormalizar
	    //    writeln("Predicción ", i+1, ": ", prediccion);
	    //    ultimo_valor = estado; // Usar la predicción como entrada
	    //}


	}

	void resetStates() {
		foreach (ref capa; celdas) {
			foreach (ref celda; capa) {
				celda.memoria_largo_plazo = 0.0f;
				celda.memoria_corto_plazo = 0.0f;
			}
		}
	}

	float xavierInit(int inputSize, int outputSize) {
	    // Versión mejorada para LSTM
	    float scale = sqrt(4.0f / (inputSize + outputSize));  // Factor 4 es mejor para LSTM
	    
	    auto rnd = Random(unpredictableSeed);
	    
	    return uniform(-scale, scale, rnd);
	}
	
	void guardarPesos(string archivo) {
		auto f = File(archivo, "w");
		foreach (capa; celdas) {
			foreach (celda; capa) {
				f.writefln("%f %f %f %f", 
					celda.peso_olvido, celda.peso_entrada,
					celda.peso_candidato, celda.peso_salida);
			}
		}
	}

	float calcularError(size_t t, size_t i, size_t j, float[] objetivo, float[] salida, float[][] gradientes) {
	    // Si es la última capa
	    if (i == arquitectura_red.length - 1) {
	        // Verificar límites temporales
	        if (t >= objetivo.length || t >= salida.length) {
	            debug writeln("Advertencia: Índice temporal ", t, " fuera de rango");
	            return 0.0f;
	        }
	        return objetivo[t] - salida[t];
	    }
	    // Para capas ocultas
	    else  {
	        // Verificar existencia de capa siguiente y neurona
	        if (i+1 >= gradientes.length || j >= gradientes[i+1].length) {
	            debug writeln("Advertencia: Intento de acceso inválido a gradientes[", i+1, "][", j, "]");
	            return 0.0f;
	        }
	        return gradientes[i+1][j];
	    }
	}


	/* --------------------------- Aqui son las funciones de desnormalizacion ----------------------------- */

	float desnormalizar(float valor_normalizado, float min_val, float max_val) {
	    return valor_normalizado * (max_val - min_val) + min_val;
	}

	// Versión para arrays
	float[] desnormalizarArray(float[] valores_normalizados, float min_val, float max_val) {
	    return valores_normalizados.map!(x => x * (max_val - min_val) + min_val).array;
	}


}


