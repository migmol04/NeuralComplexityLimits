import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
import math
import itertools


def red_neuronal(model, f):
    # Generar todas las combinaciones de 20 bits
    N = 2**21
    bits = np.arange(21, dtype=np.uint32)
    X_all = ((np.arange(N, dtype=np.uint32)[:, None] >> bits) & 1).astype(np.float32)
    
    # Muestreo aleatorio de 500_000 ejemplos
    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=500_000, replace=False)
    X = X_all[idx]
    
    y = np.array([f(row) for row in X], dtype=np.float32)
    
    # Dividir 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Entrenamiento rápido
    model.fit(X_train, y_train, epochs=10, batch_size=2048, verbose=2)
    
    # Evaluación
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    print("Test accuracy:", accuracy_score(y_test, y_pred))

def sensitivity_stats(f, n):
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


    print(f"Sensibilidad media: {avg_sens}")
    print(f"Varianza sensibilidad (normalizada): {norm_var}")


model_1 = Sequential([
    Input(shape=(21,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_2 = Sequential([
    Input(shape=(21,)),
    Dense(9, activation='relu'),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_3 = Sequential([
    Input(shape=(21,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_4 = Sequential([
    Input(shape=(21,)),
    Dense(40, activation='relu'),
    Dense(20, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_5 = Sequential([
    Input(shape=(21,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(3, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_6 = Sequential([
    Input(shape=(21,)),
    Dense(30, activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])

def automata_finito(transition, start_state, accept_states):
    accept_states = set(accept_states)

    def f(entrada):
        state = start_state
        for b in entrada:
            b = int(b)
            state = transition[(state, b)]
        return 1.0 if state in accept_states else 0.0

    return f

# Lenguaje L1: todos los bits son 0
transition_all_zeros = {
    ('q0', 0): 'q0',
    ('q0', 1): 'q1',
    ('q1', 0): 'q1',
    ('q1', 1): 'q1',
}
dfa_all_zeros = automata_finito(transition_all_zeros, 'q0', ['q0'])

# Lenguaje L2: al menos un 1
transition_at_least_one_1 = {
    ('q0', 0): 'q0',
    ('q0', 1): 'q1',
    ('q1', 0): 'q1',
    ('q1', 1): 'q1',
}
dfa_at_least_one_1 = automata_finito(transition_at_least_one_1, 'q0', ['q1'])

# Lenguaje L3: paridad impar de 1,s
transition_parity_odd = {
    ('even', 0): 'even',
    ('even', 1): 'odd',
    ('odd', 0): 'odd',
    ('odd', 1): 'even',
}
dfa_parity_odd = automata_finito(transition_parity_odd, 'even', ['odd'])

# Lenguaje L4: empieza por 1 y termina en 0 (directo, sin AFD)
def lang_start1_end0(entrada):
    return 1.0 if (entrada[0] == 1 and entrada[-1] == 0) else 0.0

print("LENGUAJE L1: todos ceros")
red_neuronal(model_1, dfa_all_zeros)
sensitivity_stats(dfa_all_zeros, 21)

print("LENGUAJE L2: al menos un 1")
red_neuronal(model_1, dfa_at_least_one_1)
sensitivity_stats(dfa_at_least_one_1, 21)

print("LENGUAJE L3: paridad impar")
red_neuronal(model_1, dfa_parity_odd)
sensitivity_stats(dfa_parity_odd, 21)

print("LENGUAJE L4: empieza en 1 y acaba en 0")
red_neuronal(model_1, lang_start1_end0)
sensitivity_stats(lang_start1_end0, 21)
