import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
import itertools
import gc
import json

# ===========================
# 0) REPRODUCIBILIDAD
# ===========================

def fijar_semilla(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def evaluar_modelo(creador_modelo, f, bits_red, n_muestras, epochs, repeticiones=5, seed_base=1000):
    """
    Entrena 'repeticiones' veces y devuelve (media, desviación típica) de accuracy.
    Cada repetición usa una semilla distinta para que la media±std sea significativa.
    """
    accs = []
    for k in range(repeticiones):
        fijar_semilla(seed_base + k)
        modelo = creador_modelo()
        acc = entrenar_red(modelo, f, bits_red, n_muestras, epochs, seed=seed_base + k)
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


# ===========================
# 1. ENTRENAMIENTO DE LA RED NEURONAL
# ===========================

def entrenar_red(modelo, funcion_objetivo, n_bits=24, n_muestras=500_000, epochs=20, seed=42):
    """
    La semilla del muestreo se pasa como parámetro, de modo que cada
    repetición usa datos distintos y la std es real.

    OPTIMIZACIÓN DE MEMORIA: solo se materializan las cadenas muestreadas.
    (A 24 bits, generar las 2^24 cadenas completas ocuparía ~1,6 GB de RAM;
    así solo se generan las n_muestras elegidas, ~50 MB.)
    """
    N = 2**n_bits
    bits = np.arange(n_bits, dtype=np.uint32)

    # Muestreo aleatorio de índices y generación SOLO de esas cadenas
    n_muestras = min(n_muestras, N)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n_muestras, replace=False)
    X = ((idx[:, None].astype(np.uint32) >> bits) & 1).astype(np.float32)

    y = np.array([funcion_objetivo(row) for row in X], dtype=np.float32)

    # División entrenamiento/test
    X_entreno, X_test, y_entreno, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # batch_size alto para acelerar; si da problemas de memoria, bajar a 2048
    modelo.fit(X_entreno, y_entreno, epochs=epochs, batch_size=4096, verbose=0)

    y_pred = (modelo.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    resultado = accuracy_score(y_test, y_pred)

    # Liberar memoria explícitamente tras cada entrenamiento
    del X, y, X_entreno, X_test, y_entreno, y_test, y_pred
    gc.collect()
    tf.keras.backend.clear_session()

    return resultado


# ===========================
# 2. MEDIDAS DE FRONTERA (SENSIBILIDAD)
# ===========================

def calcular_sensibilidad(funcion, n):
    """
    Sensibilidad media normalizada en [0,1] y varianza normalizada.
    Solo es exacta para n pequeño (≤ 20 aprox).
    """
    total_cambios = 0
    sensibilidades = []

    for x in itertools.product([0, 1], repeat=n):
        fx = funcion(x)
        sensibilidad_x = 0

        for i in range(n):
            x_mod = list(x)
            x_mod[i] ^= 1
            f_mod = funcion(tuple(x_mod))
            cambio = int(fx != f_mod)
            sensibilidad_x += cambio
            total_cambios += cambio

        sensibilidades.append(sensibilidad_x)

    sens_media = total_cambios / (2**n * n)
    media = np.mean(sensibilidades)
    varianza = np.mean((np.array(sensibilidades) - media) ** 2)
    var_normalizada = varianza / ((n**2) / 4)

    return sens_media, var_normalizada


def tasa_positivos(funcion, n):
    """
    Fracción de cadenas de longitud n que pertenecen al lenguaje.
    Útil para detectar lenguajes muy desbalanceados (p. ej. la copia w=uu),
    donde una accuracy alta puede deberse a predecir siempre la clase mayoritaria.
    """
    pos = 0
    for x in itertools.product([0, 1], repeat=n):
        pos += funcion(x)
    return pos / (2**n)


# ===========================
# 3. DEFINICIÓN DE MODELOS NEURONALES
# ===========================

# IMPORTANTE: N_BITS = 24 (par y múltiplo de 3).
# LT1 (copia w = u·u) exige longitud PAR: con longitud impar el lenguaje
# es vacío y el experimento pierde el sentido. 24 es par y mantiene la
# comparabilidad con el experimento de lenguajes incontextuales.
N_BITS = 24   # tamaño de entrada global (debe coincidir con bits_red del main)

def crear_modelo_lineal():
    """Sin capas ocultas: modelo logístico."""
    return Sequential([
        Input(shape=(N_BITS,)),
        Dense(1, activation="sigmoid"),
    ])


def crear_modelo_simple(activacion="relu"):
    """1 capa oculta pequeña."""
    return Sequential([
        Input(shape=(N_BITS,)),
        Dense(8, activation=activacion),
        Dense(1, activation="sigmoid"),
    ])


def crear_modelo_profundo(activacion="relu"):
    """2 capas ocultas medianas."""
    return Sequential([
        Input(shape=(N_BITS,)),
        Dense(32, activation=activacion),
        Dense(16, activation=activacion),
        Dense(1, activation="sigmoid"),
    ])


def crear_modelo_grande(activacion="relu"):
    """3 capas ocultas grandes."""
    return Sequential([
        Input(shape=(N_BITS,)),
        Dense(64, activation=activacion),
        Dense(64, activation=activacion),
        Dense(32, activation=activacion),
        Dense(1, activation="sigmoid"),
    ])


def crear_modelo_muy_grande(activacion="relu"):
    """4 capas ocultas muy grandes."""
    return Sequential([
        Input(shape=(N_BITS,)),
        Dense(128, activation=activacion),
        Dense(128, activation=activacion),
        Dense(64,  activation=activacion),
        Dense(32,  activation=activacion),
        Dense(1, activation="sigmoid"),
    ])


def crear_modelo_enorme(activacion="relu"):
    """5 capas ocultas — nueva arquitectura para máquinas potentes.
    Si un lenguaje sigue en el azar incluso aquí, la evidencia de que la
    dificultad no es cuestión de capacidad se refuerza."""
    return Sequential([
        Input(shape=(N_BITS,)),
        Dense(256, activation=activacion),
        Dense(256, activation=activacion),
        Dense(128, activation=activacion),
        Dense(64,  activation=activacion),
        Dense(32,  activation=activacion),
        Dense(1, activation="sigmoid"),
    ])


# Catálogo de arquitecturas: (nombre_corto, función_creadora, necesita_activacion)
ARQUITECTURAS = [
    ("lineal",      crear_modelo_lineal,      False),
    ("simple",      crear_modelo_simple,      True),
    ("profundo",    crear_modelo_profundo,     True),
    ("grande",      crear_modelo_grande,       True),
    ("muy_grande",  crear_modelo_muy_grande,   True),
    ("enorme",      crear_modelo_enorme,       True),
]

ACTIVACIONES = ["relu", "tanh", "sigmoid"]


# ===========================
# 4. LENGUAJES DE NIVEL MÁQUINA DE TURING
#    (decidibles pero NO incontextuales)
# ===========================
# Ninguno de estos lenguajes es libre de contexto:
#  - LT1 (copia w = u·u) es el ejemplo clásico de lenguaje no incontextual.
#  - LT2, LT3 y LT5 no lo son por el teorema de Parikh: la imagen de Parikh
#    de todo lenguaje incontextual es semilineal, y los conjuntos
#    {cuadrados}, {primos} y {potencias de dos} no son semilineales.
#  - LT4 (representaciones binarias de los números primos) es un ejemplo
#    conocido de lenguaje no incontextual.
# Todos son decidibles (de hecho, sensibles al contexto), por lo que se
# sitúan por encima de los incontextuales en la jerarquía de Chomsky y
# requieren una máquina de Turing (con cinta acotada basta) para
# reconocerlos; no existe autómata de pila que lo haga.

def _es_primo(k):
    """Test de primalidad por división de prueba (suficiente para k < 2^18)."""
    if k < 2:
        return False
    if k < 4:
        return True
    if k % 2 == 0:
        return False
    i = 3
    while i * i <= k:
        if k % i == 0:
            return False
        i += 2
    return True


def LT1_copia(entrada):
    """w = u·u (la primera mitad es idéntica a la segunda).
    El lenguaje no incontextual por excelencia. Requiere longitud PAR.
    OJO: muy desbalanceado (a n=18 solo 2^9/2^18 ≈ 0,2% de positivos).
    Una accuracy ~99,8% puede significar 'predecir siempre 0'; contrastar
    con la tasa de positivos que se imprime junto a los resultados."""
    v = [int(b) for b in entrada]
    n = len(v)
    if n % 2 != 0:
        return 0.0
    h = n // 2
    return 1.0 if v[:h] == v[h:] else 0.0


def LT2_unos_cuadrado(entrada):
    """El número de unos es un cuadrado perfecto (0, 1, 4, 9, 16...).
    No incontextual: {k²} no es un conjunto semilineal (Parikh)."""
    s = int(np.sum(entrada))
    r = int(np.sqrt(s))
    return 1.0 if r * r == s else 0.0


def LT3_unos_primo(entrada):
    """El número de unos es primo (2, 3, 5, 7, 11, 13, 17...).
    No incontextual: los primos no forman un conjunto semilineal.
    Su sensibilidad es muy alta (≈0,87 a n=12): los primos y no primos
    se alternan con frecuencia, así que cambiar un bit cambia la clase
    muy a menudo. Es el candidato natural a 'inaprendible' de este bloque."""
    return 1.0 if _es_primo(int(np.sum(entrada))) else 0.0


def LT4_numero_primo(entrada):
    """La cadena, interpretada como número en binario, es un número primo.
    El lenguaje de las representaciones binarias de los primos es un
    ejemplo conocido de lenguaje no incontextual."""
    v = 0
    for b in entrada:
        v = (v << 1) | int(b)
    return 1.0 if _es_primo(v) else 0.0


def LT5_unos_potencia2(entrada):
    """El número de unos es una potencia de dos (1, 2, 4, 8, 16...).
    No incontextual: {2^k} no es un conjunto semilineal (Parikh)."""
    s = int(np.sum(entrada))
    return 1.0 if s > 0 and (s & (s - 1)) == 0 else 0.0


# ===========================
# 5. EXPERIMENTOS Y TABLA DE RESULTADOS
# ===========================

if __name__ == "__main__":
    bits_red  = 24   # bits de entrada para la red (PAR: imprescindible para LT1)
    bits_sens = 12   # bits para sensibilidad exacta (2^12 = 4096 puntos; también par)

    # Hiperparámetros
    NMUESTRAS = 500_000
    EPOCHS    = 20
    REPS      = 5      # repeticiones por configuración

    # Lenguajes de nivel máquina de Turing: (nombre, función)
    lenguajes = [
        ("LT1_copia",           LT1_copia),
        ("LT2_unos_cuadrado",   LT2_unos_cuadrado),
        ("LT3_unos_primo",      LT3_unos_primo),
        ("LT4_numero_primo",    LT4_numero_primo),
        ("LT5_unos_potencia2",  LT5_unos_potencia2),
    ]

    resultados = []

    for nombre, f in lenguajes:
        print(f"\n=== {nombre} (nivel máquina de Turing) ===")

        fila = {
            "lenguaje": nombre,
            "tipo":     "turing",
            "estados":  None,   # sin DFA ni autómata de pila que lo reconozca
        }

        seed_offset = 0

        for arq_nombre, arq_fn, necesita_act in ARQUITECTURAS:
            if not necesita_act:
                # Modelo lineal: sin activación configurable
                mean, std = evaluar_modelo(
                    creador_modelo=arq_fn,
                    f=f,
                    bits_red=bits_red,
                    n_muestras=NMUESTRAS,
                    epochs=EPOCHS,
                    repeticiones=REPS,
                    seed_base=1000 + seed_offset,
                )
                fila[f"{arq_nombre}_mean"] = mean
                fila[f"{arq_nombre}_std"]  = std
                seed_offset += 100
                print(f"  {arq_nombre:12}: {mean:.4f} ± {std:.4f}")
            else:
                for act in ACTIVACIONES:
                    mean, std = evaluar_modelo(
                        creador_modelo=lambda fn=arq_fn, a=act: fn(activacion=a),
                        f=f,
                        bits_red=bits_red,
                        n_muestras=NMUESTRAS,
                        epochs=EPOCHS,
                        repeticiones=REPS,
                        seed_base=1000 + seed_offset,
                    )
                    fila[f"{arq_nombre}_{act}_mean"] = mean
                    fila[f"{arq_nombre}_{act}_std"]  = std
                    seed_offset += 100
                    print(f"  {arq_nombre:12} [{act:7}]: {mean:.4f} ± {std:.4f}")

        # Sensibilidad, varianza y tasa de positivos (exactas para bits_sens)
        sens, var = calcular_sensibilidad(f, bits_sens)
        base = tasa_positivos(f, bits_sens)
        fila["sens"] = sens
        fila["var"]  = var
        fila["tasa_positivos"] = base
        print(f"  Sensibilidad={sens:.4f}  Var_norm={var:.4f}  Tasa_positivos={base:.4f}")

        resultados.append(fila)

        # Guardar resultados parciales por si se interrumpe
        with open('resultados_parciales_turing.json', 'w') as fp:
            json.dump(resultados, fp, indent=2)
        print(f"  [Guardado en resultados_parciales_turing.json]")

    # ===========================
    # TABLA FINAL
    # ===========================
    print("\n\n" + "=" * 40 + " RESULTADOS FINALES (NIVEL MÁQUINA DE TURING) " + "=" * 40)

    # Cabecera
    cabecera = f"{'Lenguaje':24} | {'Lineal':14}"
    for arq_nombre, _, necesita_act in ARQUITECTURAS:
        if not necesita_act:
            continue
        for act in ACTIVACIONES:
            cabecera += f" | {arq_nombre[:4]}({act[:4]})"
    cabecera += f" | {'Sens':6} | {'Var':6} | {'Pos':6}"
    print(cabecera)
    print("-" * len(cabecera))

    for r in resultados:
        fila_str = f"{r['lenguaje']:24} | {r['lineal_mean']:.4f}±{r['lineal_std']:.3f}"
        for arq_nombre, _, necesita_act in ARQUITECTURAS:
            if not necesita_act:
                continue
            for act in ACTIVACIONES:
                m = r.get(f"{arq_nombre}_{act}_mean", float("nan"))
                s = r.get(f"{arq_nombre}_{act}_std",  float("nan"))
                fila_str += f" | {m:.4f}±{s:.3f}"
        fila_str += f" | {r['sens']:.4f} | {r['var']:.4f} | {r['tasa_positivos']:.4f}"
        print(fila_str)

    print("-" * len(cabecera))

    # Valores de referencia a bits_sens=12 (calculados exactos):
    #   LT1 copia w=uu          sens≈0.0312  positivos≈0.0156
    #   LT2 #unos cuadrado      sens≈0.3545  positivos≈0.1777
    #   LT3 #unos primo         sens≈0.8652  positivos≈0.4595
    #   LT4 nº binario primo    sens≈0.2273  positivos≈0.1377
    #   LT5 #unos potencia de 2 sens≈0.5107  positivos≈0.2607
