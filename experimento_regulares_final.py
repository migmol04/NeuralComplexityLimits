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


# ===========================
# 3. DEFINICIÓN DE MODELOS NEURONALES
# ===========================

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
    Si la paridad sigue en el azar incluso aquí, la evidencia de que la
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
# 4. AUTÓMATAS FINITOS (LENGUAJES REGULARES)
# ===========================

def crear_automata(transiciones, estado_inicial, estados_aceptacion):
    estados_aceptacion = set(estados_aceptacion)

    def f(entrada):
        estado = estado_inicial
        for b in entrada:
            estado = transiciones[(estado, int(b))]
        return 1.0 if estado in estados_aceptacion else 0.0

    return f


def contar_estados(transiciones):
    estados = set()
    for (q, _b), q2 in transiciones.items():
        estados.add(q)
        estados.add(q2)
    return len(estados)


def inflar_automata(transiciones, estado_inicial, estados_aceptacion, M=10):
    """
    Producto con un contador cíclico de tamaño M.
    Mismo lenguaje, ~|Q|·M estados → misma sensibilidad, más complejidad estructural.
    """
    estados_orig = sorted(
        set(q for (q, _) in transiciones.keys()) | set(transiciones.values()),
        key=str
    )
    acc = set(estados_aceptacion)

    trans2 = {}
    for q in estados_orig:
        for i in range(M):
            for b in [0, 1]:
                q2 = transiciones[(q, b)]
                trans2[((q, i), b)] = (q2, (i + 1) % M)

    acc2 = {(q, i) for q in acc for i in range(M)}
    return trans2, (estado_inicial, 0), acc2


# ===========================
# Lenguajes / autómatas
# ===========================

# L1: solo acepta la cadena de todos ceros
trans_todos_ceros = {
    ("q0", 0): "q0",
    ("q0", 1): "q1",
    ("q1", 0): "q1",
    ("q1", 1): "q1",
}
L1_todos_ceros = crear_automata(trans_todos_ceros, "q0", ["q0"])

# L2: al menos un 1
trans_al_menos_un_1 = {
    ("q0", 0): "q0",
    ("q0", 1): "q1",
    ("q1", 0): "q1",
    ("q1", 1): "q1",
}
L2_al_menos_un_1 = crear_automata(trans_al_menos_un_1, "q0", ["q1"])

# L3/L10: paridad impar / par
trans_paridad = {
    ("par",   0): "par",
    ("par",   1): "impar",
    ("impar", 0): "impar",
    ("impar", 1): "par",
}
L3_paridad_impar = crear_automata(trans_paridad, "par", ["impar"])
L10_paridad_par  = crear_automata(trans_paridad, "par", ["par"])

# L4: empieza en 1 y termina en 0  (función directa, no autómata)
def L4_empieza1_termina0(entrada):
    return 1.0 if int(entrada[0]) == 1 and int(entrada[-1]) == 0 else 0.0

# L5: número de unos ≡ 0 (mod 3)
trans_mod3 = {
    ("r0", 0): "r0",
    ("r0", 1): "r1",
    ("r1", 0): "r1",
    ("r1", 1): "r2",
    ("r2", 0): "r2",
    ("r2", 1): "r0",
}
L5_mod3 = crear_automata(trans_mod3, "r0", ["r0"])

# L6: termina en 1
def L6_acaba_en_1(entrada):
    return 1.0 if int(entrada[-1]) == 1 else 0.0

# L7: empieza en 0
def L7_empieza_en_0(entrada):
    return 1.0 if int(entrada[0]) == 0 else 0.0

# L8: exactamente 2 unos
def L8_exactamente_2(entrada):
    return 1.0 if int(np.sum(entrada)) == 2 else 0.0

# L9: hasta 3 unos
def L9_hasta_3_unos(entrada):
    return 1.0 if int(np.sum(entrada)) <= 3 else 0.0

# L11: termina en 01
def L11_termina_01(entrada):
    return 1.0 if (int(entrada[-2]) == 0 and int(entrada[-1]) == 1) else 0.0

# L12: contiene la subcadena 101
trans_contiene_101 = {
    ("q0", 0): "q0",
    ("q0", 1): "q1",
    ("q1", 0): "q2",
    ("q1", 1): "q1",
    ("q2", 0): "q0",
    ("q2", 1): "q3",
    ("q3", 0): "q3",
    ("q3", 1): "q3",
}
L12_contiene_101 = crear_automata(trans_contiene_101, "q0", ["q3"])


# ===========================
# 5. EXPERIMENTOS Y TABLA DE RESULTADOS
# ===========================

if __name__ == "__main__":
    bits_red  = 24   # bits de entrada para la red (2^24 = 16,7M cadenas)
    bits_sens = 12   # bits para el cálculo exacto de sensibilidad (2^12 = 4096 puntos)

    # Hiperparámetros (configuración para máquina potente)
    NMUESTRAS = 500_000
    EPOCHS    = 20
    REPS      = 5      # repeticiones por configuración

    # Lenguajes base (nombre, función, transiciones, estado_ini, estados_acc)
    lenguajes_base = [
        ("L1_todos_ceros",    L1_todos_ceros,    trans_todos_ceros,    "q0",  ["q0"]),
        ("L2_al_menos_un_1",  L2_al_menos_un_1,  trans_al_menos_un_1,  "q0",  ["q1"]),
        ("L3_paridad_impar",  L3_paridad_impar,  trans_paridad,        "par", ["impar"]),
        ("L5_mod3",           L5_mod3,           trans_mod3,           "r0",  ["r0"]),
        ("L12_contiene_101",  L12_contiene_101,  trans_contiene_101,   "q0",  ["q3"]),
    ]

    # Crear lista final con versiones originales e infladas
    SOLO_ORIGINALES = True
    M_inflado = 50

    lenguajes = []
    for nombre, f, trans, ini, acc in lenguajes_base:
        lenguajes.append((nombre, f, trans, ini, acc, False))
        if not SOLO_ORIGINALES:
            trans2, ini2, acc2 = inflar_automata(trans, ini, acc, M=M_inflado)
            f2 = crear_automata(trans2, ini2, acc2)
            lenguajes.append((nombre + f"_x{M_inflado}", f2, trans2, ini2, acc2, True))

    resultados = []

    for nombre, f, trans, ini, acc, inflado in lenguajes:
        print(f"\n=== {nombre} ({'inflado' if inflado else 'original'}) ===")
        n_estados = contar_estados(trans)

        fila = {
            "lenguaje": nombre,
            "inflado":  inflado,
            "estados":  n_estados,
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

        # Sensibilidad y varianza (exactas para bits_sens)
        sens, var = calcular_sensibilidad(f, bits_sens)
        fila["sens"] = sens
        fila["var"]  = var
        print(f"  Sensibilidad={sens:.4f}  Var_norm={var:.4f}")

        resultados.append(fila)

        # Guardar resultados parciales por si se interrumpe
        with open('resultados_parciales.json', 'w') as fp:
            json.dump(resultados, fp, indent=2)
        print(f"  [Guardado en resultados_parciales.json]")

    # ===========================
    # TABLA FINAL
    # ===========================
    print("\n\n" + "=" * 40 + " RESULTADOS FINALES " + "=" * 40)

    # Cabecera
    cabecera = f"{'Lenguaje':40} | {'Est':5} | {'Lineal':14}"
    for arq_nombre, _, necesita_act in ARQUITECTURAS:
        if not necesita_act:
            continue
        for act in ACTIVACIONES:
            cabecera += f" | {arq_nombre[:4]}({act[:4]})"
    cabecera += f" | {'Sens':6} | {'Var':6}"
    print(cabecera)
    print("-" * len(cabecera))

    for r in resultados:
        fila_str = f"{r['lenguaje']:40} | {r['estados']:5d} | {r['lineal_mean']:.4f}±{r['lineal_std']:.3f}"
        for arq_nombre, _, necesita_act in ARQUITECTURAS:
            if not necesita_act:
                continue
            for act in ACTIVACIONES:
                m = r.get(f"{arq_nombre}_{act}_mean", float("nan"))
                s = r.get(f"{arq_nombre}_{act}_std",  float("nan"))
                fila_str += f" | {m:.4f}±{s:.3f}"
        fila_str += f" | {r['sens']:.4f} | {r['var']:.4f}"
        print(fila_str)

    print("-" * len(cabecera))

    # Verificación: solo si hay inflados
    if not SOLO_ORIGINALES:
        print("\nVERIFICACIÓN sensibilidad original vs inflado:")
        print("-" * 80)
        for nombre_base, *_ in lenguajes_base:
            orig = next(x for x in resultados if x["lenguaje"] == nombre_base)
            infl = next(x for x in resultados if x["lenguaje"].startswith(nombre_base + "_x"))
            print(
                f"{nombre_base:25} | estados {orig['estados']:3d} → {infl['estados']:5d}"
                f" | sens {orig['sens']:.4f} → {infl['sens']:.4f}"
                f" | var {orig['var']:.4f} → {infl['var']:.4f}"
            )
        print("-" * 80)
