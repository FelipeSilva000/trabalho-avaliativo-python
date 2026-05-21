import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. IMPLEMENTAÇÃO DAS FUNÇÕES E DERIVADAS
# ==========================================

def f(x):
    """Função original de quarto grau (função de custo)."""
    return x**4 - 8*x**3 + 18*x**2 - 11*x + 2

def df(x):
    """Primeira derivada analítica f'(x) -> Função cujas raízes buscamos."""
    return 4*x**3 - 24*x**2 + 36*x - 11

def ddf(x):
    """Segunda derivada analítica f''(x) -> Usada no Método de Newton."""
    return 12*x**2 - 48*x + 36

# ==========================================
# 2. MÉTODO DE NEWTON-RAPHSON
# ==========================================

def metodo_newton(x0, tol=1e-6, max_iter=50):
    historico = [x0]
    x = x0
    
    for i in range(1, max_iter + 1):
        derivada = df(x)
        segunda_derivada = ddf(x)
        
        # Guardrail contra divisão por zero (onde o método explode/diverge)
        if abs(segunda_derivada) < 1e-10:
            print(f"  [Newton] x0 = {x0} -> ERRO: Segunda derivada próxima de zero em x = {x:.4f}. O método EXPLODIU!")
            return None
        
        x_novo = x - derivada / segunda_derivada
        historico.append(x_novo)
        
        # Critério de parada por proximidade
        if abs(x_novo - x) < tol:
            print(f"  [Newton] x0 = {x0} -> Convergiu para {x_novo:.4f} em {i} iterações.")
            return historico
            
        x = x_novo
        
    print(f"  [Newton] x0 = {x0} -> Não convergiu no limite de iterações.")
    return historico

# ==========================================
# 3. MÉTODO DA SECANTE
# ==========================================

def metodo_secante(x0, x1, tol=1e-6, max_iter=50):
    historico = [x0, x1]
    
    for i in range(1, max_iter + 1):
        f_x0 = df(x0)
        f_x1 = df(x1)
        
        if abs(f_x1 - f_x0) < 1e-10:
            print(f"  [Secante] Par ({historico[0]}, {historico[1]}) -> ERRO: Divisão por zero (Secante horizontal).")
            return None
            
        # Fórmula do passo da Secante
        x_novo = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        historico.append(x_novo)
        
        if abs(x_novo - x1) < tol:
            print(f"  [Secante] Par ({historico[0]}, {historico[1]}) -> Convergiu para {x_novo:.4f} em {i} iterações.")
            return historico
            
        x0 = x1
        x1 = x_novo
        
    print(f"  [Secante] Par ({historico[0]}, {historico[1]}) -> Não convergiu.")
    return historico

# ==========================================
# 4. EXECUÇÃO DOS TESTES E ANÁLISE
# ==========================================

print("=== Executando Método de Newton ===")
hist_n0 = metodo_newton(x0=0)
hist_n2 = metodo_newton(x0=2) # Ponto crítico / Inflexão que explode
hist_n4 = metodo_newton(x0=4)

print("\n=== Executando Método da Secante ===")
hist_s1 = metodo_secante(x0=0, x1=1)
hist_s2 = metodo_secante(x0=3, x1=5)

# ==========================================
# 5. GERANDO OS GRÁFICOS PARA O RELATÓRIO
# ==========================================

plt.figure(figsize=(12, 5))

# Subplot 1: Gráfico da derivada f'(x) para entender onde estão as raízes reais
plt.subplot(1, 2, 1)
x_val = np.linspace(-0.5, 4.5, 500)
plt.plot(x_val, df(x_val), label="$f'(x)$", color='blue', linewidth=2)
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title("Gráfico da Função Alvo: $f'(x) = 0$")
plt.xlabel("x")
plt.ylabel("$f'(x)$")
plt.grid(True, alpha=0.4)
plt.legend()

# Subplot 2: Evolução da Convergência das iterações bem-sucedidas
plt.subplot(1, 2, 2)
if hist_n0: plt.plot(hist_n0, marker='o', label="Newton ($x_0=0$)")
if hist_n4: plt.plot(hist_n4, marker='s', label="Newton ($x_0=4$)")
if hist_s1: plt.plot(hist_s1, marker='^', label="Secante ($(0,1)$)")
if hist_s2: plt.plot(hist_s2, marker='d', label="Secante ($(3,5)$)")

plt.title("Velocidade de Convergência (Evolução de $x_n$)")
plt.xlabel("Iteração")
plt.ylabel("Valor aproximado de $x$")
plt.grid(True, alpha=0.4)
plt.legend()

plt.tight_layout()
plt.show()