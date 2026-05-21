import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# CONSTANTES E CONFIGURAÇÃO DA PLACA
# ==========================================
N = 50  # Tamanho total da malha (50x50)
tolerancia = 1e-5
max_iter = 10000

def inicializar_placa():
    # Cria a placa preenchida com zeros
    T = np.zeros((N, N))
    
    # Condições de contorno (Bordas)
    T[:, 0] = 100.0   # Borda Esquerda = 100°C
    T[0, :] = 100.0   # Borda Superior = 100°C
    T[:, -1] = 0.0    # Borda Direita = 0°C
    T[-1, :] = 0.0    # Borda Inferior = 0°C
    
    # Opcional: Chute inicial para o interior (média das bordas ajuda a convergir mais rápido)
    T[1:-1, 1:-1] = 50.0
    return T

# ==========================================
# MÉTODO 1: GAUSS-SEIDEL
# ==========================================
def resolver_gauss_seidel():
    T = inicializar_placa()
    iteracoes = 0
    
    inicio = time.perf_counter()
    for k in range(max_iter):
        T_antigo = T.copy()
        max_dif = 0.0
        
        # No Gauss-Seidel, atualizamos os valores sequencialmente na mesma matriz.
        # A discretização de Laplace de 5 pontos diz: T[i,j] = (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1]) / 4
        for i in range(1, N-1):
            for j in range(1, N-1):
                valor_novo = 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1])
                dif = abs(valor_novo - T[i, j])
                if dif > max_dif:
                    max_dif = dif
                T[i, j] = valor_novo
        
        iteracoes += 1
        # Critério de parada baseado na tolerância (norma do infinito da diferença)
        if max_dif < tolerancia:
            break
            
    fim = time.perf_counter()
    return T, iteracoes, fim - inicio

# ==========================================
# MÉTODO 2: JACOBI (OPCIONAL DE COMPARAÇÃO)
# ==========================================
def resolver_jacobi():
    T = inicializar_placa()
    T_novo = T.copy()
    iteracoes = 0
    
    inicio = time.perf_counter()
    for k in range(max_iter):
        # No Jacobi, todas as atualizações usam estritamente os valores da iteração ANTERIOR
        # Podemos vetorizar com fatiamento do NumPy para ficar incrivelmente rápido
        T_novo[1:-1, 1:-1] = 0.25 * (T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2])
        
        max_dif = np.max(np.abs(T_novo - T))
        T = T_novo.copy()
        
        iteracoes += 1
        if max_dif < tolerancia:
            break
            
    fim = time.perf_counter()
    return T, iteracoes, fim - inicio

# ==========================================
# EXECUÇÃO E PLOTAGEM
# ==========================================
print("Executando as simulações...")
T_gs, iter_gs, tempo_gs = resolver_gauss_seidel()
T_jac, iter_jac, tempo_jac = resolver_jacobi()

# Plotando o Mapa de Calor (Gauss-Seidel)
plt.figure(figsize=(6, 5))
plt.imshow(T_gs, cmap='jet', origin='upper')
plt.colorbar(label='Temperatura (°C)')
plt.title(f'Distribuição de Temperatura (Gauss-Seidel)\nMalha {N}x{N} | Iterações: {iter_gs}')
plt.xlabel('X (colunas)')
plt.ylabel('Y (linhas)')
plt.show()