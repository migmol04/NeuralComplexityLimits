import os
import gc
import random
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense


# =========================================================
# 0) PARÁMETROS (configuración para máquina potente)
# =========================================================

N_BITS = 12
N = 2 ** N_BITS

# Algoritmo genético
N_POBLACION = 40          # antes 15
N_GENERACIONES = 30       # antes 5
P_MUTACION = 0.01
SEED_BASE = 42

# Red neuronal base del AG
N_CAPAS = 4               # antes 3
ANCHO = 64                # antes 16
ACTIVACION = "relu"

# Entrenamiento
EPOCHS = 6                # antes 3
BATCH_SIZE = 512          # antes 256

# Evaluación
REPS_FITNESS = 2          # antes 1: fitness menos ruidoso
REPS_CONTROL = 3

# Controles finales
CAPAS_CONTROL = [1, 2, 4]
ACTIVACIONES_CONTROL = ["relu", "tanh"]

# Ejecutar o no los bloques pesados (ahora activados)
RUN_ANALISIS_FINAL = True
RUN_EXPERIMENTO_CONTROL = True

# Pesos del fitness
# OJO — COHERENCIA CON LA MEMORIA: la tabla 8.1 del TFG (Ruido 1.1539,
# Paridad 1.6756, Imparidad 1.6951, Paridad+10% 1.4889) se generó con el
# fitness COMBINADO (1.0 / 1.0 / 0.2). Si ejecutas con otros pesos, los
# valores de referencia y del mejor individuo cambian y hay que actualizar
# la tabla y su interpretación. No mezclar números de configuraciones
# distintas en la misma tabla.
PESO_DUREZA = 1.0
PESO_SENSIBILIDAD = 1.0
PESO_BALANCE = 0.2


# =========================================================
# 1) REPRODUCIBILIDAD
# =========================================================

def fijar_semilla(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================================================
# 2) DATOS FIJOS {0,1}^N_BITS Y SPLIT FIJO
# =========================================================

fijar_semilla(SEED_BASE)

BITS = np.arange(N_BITS, dtype=np.uint32)
X_TODO = ((np.arange(N, dtype=np.uint32)[:, None] >> BITS) & 1).astype(np.float32)

rng_split = np.random.default_rng(12345)
perm = rng_split.permutation(N)
n_train = int(0.8 * N)

IDX_TRAIN = perm[:n_train]
IDX_TEST = perm[n_train:]

X_TRAIN = X_TODO[IDX_TRAIN]
X_TEST = X_TODO[IDX_TEST]


# =========================================================
# 3) MODELO NEURONAL
# =========================================================

def crear_modelo(n_capas=2, activacion="relu", ancho=16):
    capas = [Input(shape=(N_BITS,))]
    for _ in range(n_capas):
        capas.append(Dense(ancho, activation=activacion))
    capas.append(Dense(1, activation="sigmoid"))
    return Sequential(capas)


def entrenar_y_evaluar(tabla_verdad, seed=42, n_capas=2, activacion="relu", ancho=16):
    """
    Devuelve:
      - acc: accuracy normal
      - acc_efectiva: max(acc, 1-acc), para no distinguir una función de su complemento
    """
    fijar_semilla(seed)

    y_todo = tabla_verdad.astype(np.float32)
    y_train = y_todo[IDX_TRAIN]
    y_test = y_todo[IDX_TEST]

    modelo = crear_modelo(n_capas=n_capas, activacion=activacion, ancho=ancho)
    modelo.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    modelo.fit(
        X_TRAIN,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    y_pred = (modelo.predict(X_TEST, verbose=0) > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred)
    acc_efectiva = max(acc, 1.0 - acc)

    del y_todo, y_train, y_test, y_pred, modelo
    gc.collect()
    tf.keras.backend.clear_session()

    return float(acc), float(acc_efectiva)


# =========================================================
# 4) MÉTRICAS BOOLEANAS
# =========================================================

def sensibilidad_tabla(tabla):
    """
    Sensibilidad media exacta de la tabla de verdad.
    Paridad -> 1.0
    Ruido puro -> ~0.5
    """
    tabla = np.asarray(tabla, dtype=np.int8)
    indices = np.arange(N, dtype=np.int32)

    total = 0.0
    for i in range(N_BITS):
        mask = 1 << i
        total += np.mean(tabla != tabla[indices ^ mask])

    return float(total / N_BITS)


def balance_clases(tabla):
    """
    1 si la tabla está perfectamente balanceada (50% unos),
    0 si es constante.
    """
    p = float(np.mean(tabla))
    return float(1.0 - 2.0 * abs(p - 0.5))


def tabla_paridad():
    return (X_TODO.sum(axis=1) % 2).astype(np.int8)


def tabla_imparidad():
    return 1 - tabla_paridad()


def tabla_ruido(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=N).astype(np.int8)


def tabla_paridad_con_ruido(prob_ruido=0.1, seed=0):
    base = tabla_paridad().copy()
    rng = np.random.default_rng(seed)
    mascara = rng.random(N) < prob_ruido
    base[mascara] ^= 1
    return base.astype(np.int8)


def distancia_hamming_min_paridad(tabla):
    paridad = tabla_paridad()
    imparidad = tabla_imparidad()

    hamming_paridad = int(np.sum(tabla != paridad))
    hamming_imparidad = int(np.sum(tabla != imparidad))
    hamming_min = min(hamming_paridad, hamming_imparidad)

    return hamming_paridad, hamming_imparidad, hamming_min


# =========================================================
# 4b) TABLAS DE LOS LENGUAJES DE LOS OTROS EXPERIMENTOS
# =========================================================
# N_BITS = 12 es par y múltiplo de 3, así que los 10 lenguajes de los
# experimentos incontextuales y de nivel máquina de Turing están bien
# definidos en este espacio y pueden usarse como referencias del AG.

def _es_primo(k):
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


def tabla_de_funcion(f):
    """Convierte una función sobre cadenas en su tabla de verdad (orden de X_TODO)."""
    return np.array([f(fila) for fila in X_TODO], dtype=np.int8)


def _copia(x):
    v = [int(b) for b in x]
    h = len(v) // 2
    return 1.0 if v[:h] == v[h:] else 0.0


def _num_binario(x):
    v = 0
    for b in x:
        v = (v << 1) | int(b)
    return v


LENGUAJES_REFERENCIA = {
    # Incontextuales (no regulares)
    "LC1_igual_0s_1s":   lambda x: 1.0 if 2 * int(np.sum(x)) == len(x) else 0.0,
    "LC2_mayoria_unos":  lambda x: 1.0 if 2 * int(np.sum(x)) > len(x) else 0.0,
    "LC3_palindromo":    lambda x: 1.0 if [int(b) for b in x] == [int(b) for b in x][::-1] else 0.0,
    "LC4_doble_mayoria": lambda x: 1.0 if 2 * int(np.sum(x)) > (len(x) - int(np.sum(x))) else 0.0,
    "LC5_doble_ceros":   lambda x: 1.0 if (len(x) - int(np.sum(x))) == 2 * int(np.sum(x)) else 0.0,
    # Nivel máquina de Turing (no incontextuales)
    "LT1_copia":         _copia,
    "LT2_unos_cuadrado": lambda x: 1.0 if int(np.sqrt(int(np.sum(x)))) ** 2 == int(np.sum(x)) else 0.0,
    "LT3_unos_primo":    lambda x: 1.0 if _es_primo(int(np.sum(x))) else 0.0,
    "LT4_numero_primo":  lambda x: 1.0 if _es_primo(_num_binario(x)) else 0.0,
    "LT5_unos_pot2":     lambda x: 1.0 if int(np.sum(x)) > 0 and (int(np.sum(x)) & (int(np.sum(x)) - 1)) == 0 else 0.0,
}


def distancias_hamming_referencias(tabla):
    """
    Distancia de Hamming de 'tabla' a paridad, imparidad, y a todos los
    lenguajes de referencia. Devuelve lista (nombre, distancia) ordenada.
    """
    dist = []
    dist.append(("Paridad", int(np.sum(tabla != tabla_paridad()))))
    dist.append(("Imparidad", int(np.sum(tabla != tabla_imparidad()))))
    for nombre, f in LENGUAJES_REFERENCIA.items():
        t = tabla_de_funcion(f)
        dist.append((nombre, int(np.sum(tabla != t))))
    dist.sort(key=lambda p: p[1])
    return dist


# =========================================================
# 5) FITNESS DEL AG
# =========================================================

def evaluar_fitness(tabla_verdad, seed=42, n_capas=2, activacion="relu", ancho=16):
    """
    Fitness:
    - quiere funciones difíciles para la red
    - con sensibilidad alta
    - y balanceadas
    (los pesos relativos se controlan con PESO_DUREZA / PESO_SENSIBILIDAD / PESO_BALANCE)
    """
    acc, acc_efectiva = entrenar_y_evaluar(
        tabla_verdad,
        seed=seed,
        n_capas=n_capas,
        activacion=activacion,
        ancho=ancho
    )

    dureza = 1.0 - acc_efectiva
    sens = sensibilidad_tabla(tabla_verdad)
    balance = balance_clases(tabla_verdad)

    fitness = (
        PESO_DUREZA * dureza
        + PESO_SENSIBILIDAD * sens
        + PESO_BALANCE * balance
    )

    return float(fitness), float(acc), float(acc_efectiva), float(sens), float(balance)


def evaluar_fitness_medio(tabla_verdad, n_capas=2, activacion="relu", ancho=16,
                          repeticiones=1, seed_base=1000):
    fitness_vals = []
    acc_vals = []
    acc_eff_vals = []

    for i in range(repeticiones):
        fit, acc, acc_eff, sens, balance = evaluar_fitness(
            tabla_verdad,
            seed=seed_base + i,
            n_capas=n_capas,
            activacion=activacion,
            ancho=ancho
        )
        fitness_vals.append(fit)
        acc_vals.append(acc)
        acc_eff_vals.append(acc_eff)

    return {
        "fitness_mean": float(np.mean(fitness_vals)),
        "fitness_std": float(np.std(fitness_vals)),
        "acc_mean": float(np.mean(acc_vals)),
        "acc_std": float(np.std(acc_vals)),
        "acc_eff_mean": float(np.mean(acc_eff_vals)),
        "acc_eff_std": float(np.std(acc_eff_vals)),
        "sens": float(sens),
        "balance": float(balance),
    }


# =========================================================
# 6) ALGORITMO GENÉTICO
# =========================================================

def individuo_aleatorio(seed=None):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=N).astype(np.int8)


def cruce(padre1, padre2, seed=None):
    rng = np.random.default_rng(seed)
    punto = rng.integers(1, len(padre1) - 1)
    hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
    hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
    return hijo1, hijo2


def mutar(individuo, p_mutacion=P_MUTACION, seed=None):
    rng = np.random.default_rng(seed)
    mascara = rng.random(size=len(individuo)) < p_mutacion
    mutado = individuo.copy()
    mutado[mascara] ^= 1
    return mutado


def seleccion_torneo(poblacion, fitness, k=3, seed=None):
    rng = np.random.default_rng(seed)
    candidatos = rng.choice(len(poblacion), size=k, replace=False)
    mejor = max(candidatos, key=lambda i: fitness[i])
    return poblacion[mejor]


# =========================================================
# 7) REFERENCIAS
# =========================================================

def imprimir_referencia(nombre, tabla, n_capas=N_CAPAS, activacion=ACTIVACION):
    res = evaluar_fitness_medio(
        tabla,
        n_capas=n_capas,
        activacion=activacion,
        ancho=ANCHO,
        repeticiones=REPS_FITNESS,
        seed_base=7000
    )
    hp, hi, hm = distancia_hamming_min_paridad(tabla)

    print(
        f"{nombre:20} | "
        f"fitness={res['fitness_mean']:.4f}±{res['fitness_std']:.4f} | "
        f"acc={res['acc_mean']:.4f} | "
        f"acc_eff={res['acc_eff_mean']:.4f} | "
        f"sens={res['sens']:.4f} | "
        f"balance={res['balance']:.4f} | "
        f"Hmin={hm} ({100*hm/N:.1f}%)"
    )


def experimento_control():
    print("\n" + "=" * 90)
    print("EXPERIMENTO DE CONTROL: PARIDAD / RUIDO / PARIDAD+RUIDO")
    print("=" * 90)

    referencias = [
        ("Ruido", tabla_ruido(seed=1)),
        ("Paridad", tabla_paridad()),
        ("Imparidad", tabla_imparidad()),
        ("Paridad+5%ruido", tabla_paridad_con_ruido(0.05, seed=2)),
        ("Paridad+10%ruido", tabla_paridad_con_ruido(0.10, seed=3)),
        ("Paridad+20%ruido", tabla_paridad_con_ruido(0.20, seed=4)),
    ]

    print("\n--- Mismo modelo que usa el AG ---")
    for nombre, tabla in referencias:
        imprimir_referencia(nombre, tabla, n_capas=N_CAPAS, activacion=ACTIVACION)

    print("\n--- Efecto de capas y activación sobre RUIDO / PARIDAD / PARIDAD+10% ---")
    subconjunto = [
        ("Ruido", tabla_ruido(seed=10)),
        ("Paridad", tabla_paridad()),
        ("Paridad+10%ruido", tabla_paridad_con_ruido(0.10, seed=11)),
    ]

    for n_capas in CAPAS_CONTROL:
        for activacion in ACTIVACIONES_CONTROL:
            print(f"\nCapas={n_capas}, activación={activacion}")
            for nombre, tabla in subconjunto:
                res = evaluar_fitness_medio(
                    tabla,
                    n_capas=n_capas,
                    activacion=activacion,
                    ancho=ANCHO,
                    repeticiones=REPS_CONTROL,
                    seed_base=9000
                )
                print(
                    f"  {nombre:18} | "
                    f"fitness={res['fitness_mean']:.4f} | "
                    f"acc={res['acc_mean']:.4f} | "
                    f"acc_eff={res['acc_eff_mean']:.4f} | "
                    f"sens={res['sens']:.4f}"
                )


# =========================================================
# 8) BUCLE PRINCIPAL
# =========================================================

if __name__ == "__main__":
    fijar_semilla(SEED_BASE)

    print(f"Iniciando AG: {N_POBLACION} individuos, {N_GENERACIONES} generaciones, {N_BITS} bits")
    print(f"Espacio de búsqueda: 2^(2^{N_BITS}) posibles funciones booleanas")
    print(f"Modelo del AG: {N_CAPAS} capas ocultas, activación {ACTIVACION}, ancho {ANCHO}")
    print(f"Fitness = {PESO_DUREZA}·dureza + {PESO_SENSIBILIDAD}·sensibilidad + {PESO_BALANCE}·balance")
    print()

    print("=" * 90)
    print("REFERENCIAS INICIALES")
    print("=" * 90)
    imprimir_referencia("Ruido", tabla_ruido(seed=1))
    imprimir_referencia("Paridad", tabla_paridad())
    imprimir_referencia("Imparidad", tabla_imparidad())
    imprimir_referencia("Paridad+10%ruido", tabla_paridad_con_ruido(0.10, seed=2))
    print()

    print("=" * 90)
    print("REFERENCIAS: LENGUAJES DE LOS OTROS EXPERIMENTOS")
    print("=" * 90)
    for nombre_ref, f_ref in LENGUAJES_REFERENCIA.items():
        imprimir_referencia(nombre_ref, tabla_de_funcion(f_ref))
    print()

    poblacion = [individuo_aleatorio(seed=SEED_BASE + i) for i in range(N_POBLACION)]

    mejor_historico = None
    mejor_res_historico = None

    for gen in range(N_GENERACIONES):
        print(f"=== Generación {gen + 1}/{N_GENERACIONES} ===")

        fitness = []
        resultados_gen = []

        for i, ind in enumerate(poblacion):
            res = evaluar_fitness_medio(
                ind,
                n_capas=N_CAPAS,
                activacion=ACTIVACION,
                ancho=ANCHO,
                repeticiones=REPS_FITNESS,
                seed_base=SEED_BASE + gen * 1000 + i * 10
            )
            resultados_gen.append(res)
            fitness.append(res["fitness_mean"])

            print(
                f"  Individuo {i+1:2d}: "
                f"fitness={res['fitness_mean']:.4f}±{res['fitness_std']:.4f} | "
                f"acc={res['acc_mean']:.4f} | "
                f"acc_eff={res['acc_eff_mean']:.4f} | "
                f"sens={res['sens']:.4f} | "
                f"balance={res['balance']:.4f}"
            )

        mejor_idx = int(np.argmax(fitness))
        mejor_gen = resultados_gen[mejor_idx]

        print(
            f"  Mejor de la generación: "
            f"fitness={mejor_gen['fitness_mean']:.4f}±{mejor_gen['fitness_std']:.4f} | "
            f"acc_eff={mejor_gen['acc_eff_mean']:.4f} | "
            f"sens={mejor_gen['sens']:.4f}"
        )

        if mejor_res_historico is None or mejor_gen["fitness_mean"] > mejor_res_historico["fitness_mean"]:
            mejor_historico = poblacion[mejor_idx].copy()
            mejor_res_historico = mejor_gen.copy()
            print("  *** Nuevo mejor histórico ***")

        # Guardar el mejor histórico en cada generación por si se interrumpe
        np.save("mejor_individuo_parcial.npy", mejor_historico)

        nueva_poblacion = []
        nueva_poblacion.append(poblacion[mejor_idx].copy())  # elitismo

        while len(nueva_poblacion) < N_POBLACION:
            padre1 = seleccion_torneo(
                poblacion, fitness,
                seed=SEED_BASE + gen + len(nueva_poblacion)
            )
            padre2 = seleccion_torneo(
                poblacion, fitness,
                seed=SEED_BASE + gen + len(nueva_poblacion) + 1
            )

            hijo1, hijo2 = cruce(
                padre1, padre2,
                seed=SEED_BASE + gen + len(nueva_poblacion)
            )

            hijo1 = mutar(
                hijo1,
                seed=SEED_BASE + gen + len(nueva_poblacion)
            )
            hijo2 = mutar(
                hijo2,
                seed=SEED_BASE + gen + len(nueva_poblacion) + 1
            )

            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < N_POBLACION:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion
        print()

    print("=" * 90)
    print("MEJOR INDIVIDUO ENCONTRADO")
    print("=" * 90)

    print(f"Mejor fitness encontrado: {mejor_res_historico['fitness_mean']:.4f} ± {mejor_res_historico['fitness_std']:.4f}")
    print(f"Accuracy media:           {mejor_res_historico['acc_mean']:.4f} ± {mejor_res_historico['acc_std']:.4f}")
    print(f"Accuracy efectiva media:  {mejor_res_historico['acc_eff_mean']:.4f} ± {mejor_res_historico['acc_eff_std']:.4f}")
    print(f"Sensibilidad:             {mejor_res_historico['sens']:.4f}")
    print(f"Balance de clases:        {mejor_res_historico['balance']:.4f}")
    print(f"Proporción de unos:       {mejor_historico.mean():.4f}")

    np.save("mejor_individuo.npy", mejor_historico)
    print("Mejor individuo guardado en mejor_individuo.npy")

    hp, hi, hm = distancia_hamming_min_paridad(mejor_historico)
    print("\nDistancia de Hamming al mejor individuo:")
    print(f"  vs Paridad:   {hp} bits de {N} ({100*hp/N:.1f}%)")
    print(f"  vs Imparidad: {hi} bits de {N} ({100*hi/N:.1f}%)")
    print(f"  Mínimo:       {hm} bits ({100*hm/N:.1f}%)")

    if hm < N * 0.10:
        print("  → El mejor individuo es PARIDAD + poco ruido")
    elif hm < N * 0.30:
        print("  → El mejor individuo se parece bastante a la paridad")
    else:
        print("  → El mejor individuo es ruido puro o una función alejada de paridad")

    # Comparación extendida: ¿se parece a ALGÚN lenguaje estructurado?
    print("\nDistancia de Hamming a TODAS las referencias (ordenada, menor = más parecido):")
    print("(50% ≈ función sin relación; <10% ≈ prácticamente la misma función)")
    for nombre_ref, d in distancias_hamming_referencias(mejor_historico):
        print(f"  {nombre_ref:20} {d:5d} bits de {N} ({100*d/N:5.1f}%)")

    # ==========================================
    # BLOQUES PESADOS
    # ==========================================

    if RUN_ANALISIS_FINAL:
        print("\n" + "=" * 90)
        print("MEJOR INDIVIDUO: EFECTO DE CAPAS Y ACTIVACIÓN")
        print("=" * 90)

        for n_capas in CAPAS_CONTROL:
            for activacion in ACTIVACIONES_CONTROL:
                res = evaluar_fitness_medio(
                    mejor_historico,
                    n_capas=n_capas,
                    activacion=activacion,
                    ancho=ANCHO,
                    repeticiones=REPS_CONTROL,
                    seed_base=11000
                )
                print(
                    f"Capas={n_capas}, activación={activacion:7s} | "
                    f"fitness={res['fitness_mean']:.4f} | "
                    f"acc={res['acc_mean']:.4f} | "
                    f"acc_eff={res['acc_eff_mean']:.4f} | "
                    f"sens={res['sens']:.4f}"
                )

    if RUN_EXPERIMENTO_CONTROL:
        experimento_control()
