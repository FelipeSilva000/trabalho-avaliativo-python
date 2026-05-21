import numpy as np

# 1. PARÂMETROS DO DIODO (Dados do enunciado)
I_s = 1e-12    # Corrente de saturação (A)
n = 1.5        # Fator de idealidade
V_t = 0.025    # Tensão térmica (V)
I_des = 0.001  # Corrente desejada (1 mA)

def f(V):
    """Função alvo baseada na Equação de Shockley."""
    return I_s * (np.exp(V / (n * V_t)) - 1) - I_des


# 2. MÉTODO DA BISSECÇÃO
def metodo_bisseccao(a, b, tol):
    if f(a) * f(b) >= 0:
        raise ValueError("O intervalo não atende ao Teorema de Bolzano.")
    
    it = 0
    while (b - a) / 2.0 > tol:
        it += 1
        c = (a + b) / 2.0
        if abs(f(c)) < tol:
            return c, it
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2.0, it


# 3. MÉTODO DA POSIÇÃO FALSA
def metodo_posicao_falsa(a, b, tol, max_iter=2000):
    if f(a) * f(b) >= 0:
        raise ValueError("O intervalo não atende ao Teorema de Bolzano.")
    
    it = 0
    c = a
    while it < max_iter:
        it += 1
        fa, fb = f(a), f(b)
        c_ant = c
        
        # Fórmula da reta secante (interseção com o eixo x)
        c = (a * fb - b * fa) / (fb - fa)
        
        # Critério de parada pelo resíduo ou variação do ponto
        if abs(f(c)) < tol or abs(c - c_ant) < tol:
            return c, it
            
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c, it


# 4. EXECUÇÃO DO TESTE COMPATIVEL COM O DESAFIO
limite_a = 0.0
limite_b = 1.0
tolerancia = 1e-6

raiz_bis, it_bis = metodo_bisseccao(limite_a, limite_b, tolerancia)
raiz_pf, it_pf = metodo_posicao_falsa(limite_a, limite_b, tolerancia)

print(f"Bissecção:     Raiz = {raiz_bis:.6f} V | Iterações = {it_bis}")
print(f"Posição Falsa: Raiz = {raiz_pf:.6f} V | Iterações = {it_pf}")