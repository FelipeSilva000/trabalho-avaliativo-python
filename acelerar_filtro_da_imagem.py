import numpy as np
import time
import scipy.linalg

# ==========================================
# 1. CONSTRUÇÃO DA MATRIZ A (100x100)
# ==========================================
# A equação de Laplace discretizada -u[i-1] + 2u[i] - u[i+1] gera uma matriz tridiagonal.
n = 100
A = np.zeros((n, n))

for i in range(n):
    A[i, i] = 2.0          # Diagonal principal
    if i > 0:
        A[i, i-1] = -1.0   # Diagonal inferior
    if i < n - 1:
        A[i, i+1] = -1.0   # Diagonal superior

# ==========================================
# 2. DEFINIÇÃO DOS 3 VETORES b
# ==========================================
np.random.seed(42)  # Garante reprodutibilidade do vetor aleatório

b_constante = np.ones(n)
b_senoidal = np.sin(np.linspace(0, np.pi, n))
b_aleatorio = np.random.rand(n)

vetores_b = [b_constante, b_senoidal, b_aleatorio]

# ==========================================
# 3. MEDIÇÃO DE TEMPO
# ==========================================

# --- Abordagem (a): Eliminação de Gauss separada 3x ---
# Nota: np.linalg.solve resolve o sistema do zero usando fatoração para cada chamada.
inicio_gauss = time.perf_counter()
for b in vetores_b:
    x_gauss = np.linalg.solve(A, b)
fim_gauss = time.perf_counter()
tempo_gauss = fim_gauss - inicio_gauss

# --- Abordagem (b): LU única + 3 substituições ---
inicio_lu = time.perf_counter()
# Fatoração feita apenas UMA vez
lu, piv = scipy.linalg.lu_factor(A)
for b in vetores_b:
    # Apenas substituição (fase rápida) repetida 3 vezes
    x_lu = scipy.linalg.lu_solve((lu, piv), b)
fim_lu = time.perf_counter()
tempo_lu = fim_lu - inicio_lu

# ==========================================
# EXIBIÇÃO DOS RESULTADOS
# ==========================================
print(f"{'Abordagem':<35} | {'Tempo Total (segundos)':<25}")
print("-" * 65)
print(f"{'(a) Gauss Separado (3x)':<35} | {tempo_gauss:<25.6f}")
print(f"{'(b) LU Único + 3 Substituições':<35} | {tempo_lu:<25.6f}")
print(f"\nGanho de desempenho (Speedup): {tempo_gauss / tempo_lu:.2f}x mais rápido.")