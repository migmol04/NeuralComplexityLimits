import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
import itertools


# --------------------------------------------------
# 1. ENTRENAMIENTO DE LA RED
# --------------------------------------------------

def red_neuronal(model, f, n_bits=21, n_muestras=500_000, epochs=10):
    N = 2**n_bits
    bits = np.arange(n_bits, dtype=np.uint32)
    X_all = ((np.arange(N, dtype=np.uint32)[:, None] >> bits) & 1).astype(np.float32)
    
    # Muestreo aleatorio de n_muestras ejemplos
    n_muestras = min(n_muestras, N)
    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=n_muestras, replace=False)
    X = X_all[idx]
    
    y = np.array([f(row) for row in X], dtype=np.float32)
    
    # Dividir 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=2048, verbose=0)
    
    # Evaluación
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred)
    return acc


# --------------------------------------------------
# 2. SENSIBILIDAD / FRONTERA
# --------------------------------------------------

def sensitivity_stats(f, n):
    """
    Calcula sensibilidad media y varianza normalizada.
    OJO: recorre todas las cadenas de {0,1}^n → usar n pequeño (por ejemplo 10-12).
    """
    total_changes = 0
    sensitivities = []

    for x in itertools.product([0, 1], repeat=n):
        fx = f(x)
        sensitivity = 0
        for i in range(n):
            x_flip = list(x)
            x_flip[i] ^= 1
            f_flip = f(tuple(x_flip))
            change = int(fx != f_flip)
            sensitivity += change
            total_changes += change

        sensitivities.append(sensitivity)

    avg_sens = total_changes / (2**n * n)
    mean_sens = sum(sensitivities) / len(sensitivities)
    var_sens = sum((s - mean_sens) ** 2 for s in sensitivities) / len(sensitivities)
    norm_var = var_sens / ((n**2)/4) if n > 0 else 0

    return avg_sens, norm_var


# --------------------------------------------------
# 3. DEFINICIÓN DE MODELOS
# --------------------------------------------------

def crear_modelo():

    model = Sequential([
        Input(shape=(21,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
import itertools



# 1. ENTRENAMIENTO DE LA RED NEURONAL


def entrenar_red(modelo, funcion_objetivo, n_bits=21, n_muestras=500_000, epochs=10):
    """
    Entrena una red neuronal 'modelo' para aproximar una función
    f: {0,1}^n_bits -> {0,1}.
    Devuelve la accuracy sobre el conjunto de test.
    """
    N = 2**n_bits
    bits = np.arange(n_bits, dtype=np.uint32)

    # Generar todas las cadenas binarias posibles
    X_todo = ((np.arange(N, dtype=np.uint32)[:, None] >> bits) & 1).astype(np.float32)

    # Muestreo aleatorio de ejemplos
    n_muestras = min(n_muestras, N)
    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=n_muestras, replace=False)
    X = X_todo[idx]

    y = np.array([funcion_objetivo(row) for row in X], dtype=np.float32)

    # División entrenamiento/test
    X_entreno, X_test, y_entreno, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    modelo.fit(X_entreno, y_entreno, epochs=epochs, batch_size=2048, verbose=0)

    # Evaluación
    y_pred = (modelo.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    return accuracy_score(y_test, y_pred)



# 2. MEDIDAS DE FRONTERA (SENSIBILIDAD)


def calcular_sensibilidad(funcion, n):
    """
    Calcula:
    - sensibilidad media
    - varianza normalizada de la sensibilidad

    Recorre TODAS las cadenas de {0,1}^n, así que usar n pequeño (<=12).
    """
    total_cambios = 0
    sensibilidades = []

    for x in itertools.product([0, 1], repeat=n):
        fx = funcion(x)
        sensibilidad_x = 0

        for i in range(n):
            x_modificado = list(x)
            x_modificado[i] ^= 1   # cambia un bit
            f_mod = funcion(tuple(x_modificado))
            cambio = int(fx != f_mod)
            sensibilidad_x += cambio
            total_cambios += cambio

        sensibilidades.append(sensibilidad_x)

    sens_media = total_cambios / (2**n * n)
    media = np.mean(sensibilidades)
    varianza = np.mean((np.array(sensibilidades) - media)**2)
    var_normalizada = varianza / ((n**2)/4)

    return sens_media, var_normalizada



# 3. DEFINICIÓN DE MODELOS NEURONALES


def crear_modelo_simple():
    """Red neuronal MLP pequeña para los experimentos."""
    modelo = Sequential([
        Input(shape=(21,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return modelo


# 4. AUTÓMATAS FINITOS (LENGUAJES REGULARES)

def crear_automata(transiciones, estado_inicial, estados_aceptacion):
    estados_aceptacion = set(estados_aceptacion)

    def f(entrada):
        estado = estado_inicial
        for b in entrada:
            b = int(b)
            estado = transiciones[(estado, b)]
        return 1.0 if estado in estados_aceptacion else 0.0

    return f

# L1: todas las cadenas son 0
trans_todos_ceros = {
    ('q0', 0): 'q0',
    ('q0', 1): 'q1',
    ('q1', 0): 'q1',
    ('q1', 1): 'q1',
}
L1_todos_ceros = crear_automata(trans_todos_ceros, 'q0', ['q0'])

# L2: al menos un 1
trans_al_menos_un_1 = {
    ('q0', 0): 'q0',
    ('q0', 1): 'q1',
    ('q1', 0): 'q1',
    ('q1', 1): 'q1',
}
L2_al_menos_un_1 = crear_automata(trans_al_menos_un_1, 'q0', ['q1'])

# L3: paridad impar
trans_paridad_impar = {
    ('par', 0): 'par',
    ('par', 1): 'impar',
    ('impar', 0): 'impar',
    ('impar', 1): 'par',
}
L3_paridad_impar = crear_automata(trans_paridad_impar, 'par', ['impar'])

# L4: empieza en 1 y termina en 0
def L4_empieza1_termina0(entrada):
    return 1.0 if entrada[0] == 1 and entrada[-1] == 0 else 0.0

# L5: número de 1s múltiplo de 3
trans_mod3 = {
    ('r0', 0): 'r0',
    ('r0', 1): 'r1',
    ('r1', 0): 'r1',
    ('r1', 1): 'r2',
    ('r2', 0): 'r2',
    ('r2', 1): 'r0',
}
L5_mod3 = crear_automata(trans_mod3, 'r0', ['r0'])

# L6: acaba en 1
def L6_acaba_en_1(entrada):
    return 1.0 if entrada[-1] == 1 else 0.0

# L7: empieza en 0
def L7_empieza_en_0(entrada):
    return 1.0 if entrada[0] == 0 else 0.0

# L8: exactamente 2 unos
def L8_exactamente_2(entrada):
    return 1.0 if np.sum(entrada) == 2 else 0.0

# L9: número de unos ≤ 3
def L9_hasta_3_unos(entrada):
    return 1.0 if np.sum(entrada) <= 3 else 0.0

# L10: paridad par
L10_paridad_par = crear_automata(trans_paridad_impar, 'par', ['par'])

# L11: termina en "01"
def L11_termina_01(entrada):
    return 1.0 if entrada[-2:] == [0, 1] else 0.0

# L12: contiene la subcadena "101"
trans_contiene_101 = {
    ('q0', 0): 'q0',
    ('q0', 1): 'q1',
    ('q1', 0): 'q2',
    ('q1', 1): 'q1',
    ('q2', 0): 'q0',
    ('q2', 1): 'q3',
    ('q3', 0): 'q3',
    ('q3', 1): 'q3'
}
L12_contiene_101 = crear_automata(trans_contiene_101, 'q0', ['q3'])



# 5. EXPERIMENTOS Y TABLA DE RESULTADOS


if __name__ == "__main__":
    bits_red = 21       # longitud para entrenar la red
    bits_sens = 12      # longitud para sensibilidad

    lenguajes = [
        ("L1_todos_ceros",        L1_todos_ceros),
        ("L2_al_menos_un_1",      L2_al_menos_un_1),
        ("L3_paridad_impar",      L3_paridad_impar),
        ("L4_empieza1_termina0",  L4_empieza1_termina0),
        ("L5_mod3",               L5_mod3),
        ("L6_acaba_en_1",         L6_acaba_en_1),
        ("L7_empieza_en_0",       L7_empieza_en_0),
        ("L8_exactamente_2",      L8_exactamente_2),
        ("L9_hasta_3_unos",       L9_hasta_3_unos),
        ("L10_paridad_par",       L10_paridad_par),
        ("L11_termina_01",        L11_termina_01),
        ("L12_contiene_101",      L12_contiene_101),
    ]

    resultados = []

    for nombre, f in lenguajes:
        print(f"=== Entrenando: {nombre} ===")
        modelo = crear_modelo_simple()
        accuracy = entrenar_red(modelo, f, n_bits=bits_red, n_muestras=200_000, epochs=8)
        sens, var = calcular_sensibilidad(f, bits_sens)

        resultados.append({
            "lenguaje": nombre,
            "accuracy": accuracy,
            "sensibilidad": sens,
            "varianza": var,
        })

    # Mostrar tabla de resultados
    print("\nRESULTADOS:")
    print("-" * 90)
    print(f"{'Lenguaje':28} | {'Accuracy':8} | {'Sensibilidad':12} | {'Varianza':10}")
    print("-" * 90)
    for r in resultados:
        print(f"{r['lenguaje']:28} | {r['accuracy']:.4f}   | {r['sensibilidad']:.4f}       | {r['varianza']:.4f}")
    print("-" * 90)
