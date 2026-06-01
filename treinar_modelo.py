import pandas as pd                                    # manipulação de tabelas
import numpy as np                                     # operações numéricas
import matplotlib.pyplot as plt                        # criação de gráficos
import matplotlib.gridspec as gridspec                 # layout de múltiplos gráficos
from sklearn.ensemble import RandomForestRegressor     # Random Forest (ML) - Regressão
from sklearn.model_selection import train_test_split   # dividir dados em treino e teste
from sklearn.metrics import r2_score, mean_absolute_error  # métricas de avaliação
import joblib                                          # guardar/carregar o modelo treinado
from pathlib import Path
BASE = Path(__file__).parent

# Lê o CSV gerado pelo gerar_dados.py — tabela com 1000 linhas e 7 colunas
df = pd.read_csv(BASE / "dados_piscicultura.csv")

# 2. SEPARAR FEATURES (X) DO TARGET (y)

# Lista com os nomes das colunas que o modelo vai usar como entrada
FEATURES = [
    "temperatura_agua_C",       # temperatura da água em graus Celsius
    "oxigenio_dissolvido_mgL",  # oxigénio dissolvido em mg/L
    "racao_diaria_g",           # quantidade de ração diária em gramas
    "dias_desde_criacao",       # dias desde o início do lote
    "densidade_tanque_pm3",     # densidade de peixes por metro cúbico
    "ph_agua",                  # pH da água do tanque
]

TARGET = "peso_g"  # variável que queremos prever (peso do peixe em gramas)

# X = tabela 1000×6 com as features — o que o modelo recebe como input
X = df[FEATURES]

# y = coluna 1000×1 com o target — o que o modelo tem de aprender a prever
y = df[TARGET]


# DIVIDIR EM TREINO E TESTE

# Divide os dados em dois grupos:
#   - X_train, y_train: 800 registos (80%) — usados para o modelo aprender
#   - X_test,  y_test:  200 registos (20%) — usados para avaliar o modelo
# test_size=0.2  → 20% dos dados vão para teste
# random_state=42 → garante que a divisão é sempre igual (reproducibilidade)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cria e treina o modelo de Random Forest Regressor

# Random Forest Regressor: constrói N árvores de decisão independentes.
# n_estimators=200 → número de árvores (mais árvores = mais preciso, mais lento)
# random_state=42  → reproducibilidade do treino
# n_jobs=-1        → usa todos os núcleos do CPU para treinar em paralelo
modelo = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

# .fit() é onde acontece o treino — o modelo aprende os padrões dos dados
modelo.fit(X_train, y_train)

# FAZER PREVISÕES E AVALIAR O MODELO
# O modelo prevê o peso para os 200 registos de teste
y_pred = modelo.predict(X_test)

# R² (coeficiente de determinação):
#   - Varia entre 0 e 1. Quanto mais próximo de 1, melhor = 0.88 significa que o modelo explica 88% da variação do peso.
r2 = r2_score(y_test, y_pred)

# - 22.9g significa que o modelo erra em média 23 gramas (MAE = Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)

# Imprimir resultados no terminal
print(f"R²  : {r2:.4f}")
print(f"MAE : {mae:.2f} g")
print(f"Amostras treino : {len(X_train)}")
print(f"Amostras teste  : {len(X_test)}")
print(df["peso_g"].describe())

# GUARDAR O MODELO TREINADO

# joblib.dump() serializa o modelo num ficheiro binário (.pkl).
# app.py vai carregar este ficheiro com joblib.load()
joblib.dump(modelo, BASE / "modelo.pkl")
print("Modelo guardado em modelo.pkl")

# GRÁFICOS DE AVALIAÇÃO

fig = plt.figure(figsize=(13, 5))
fig.patch.set_facecolor('#F8F7F4')

# GridSpec permite definir um layout de subgráficos com controle preciso do espaço
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)


# --- Gráfico 1: Real vs Previsto ---
# Cada ponto = um peixe do conjunto de teste.
# Eixo X = peso real | Eixo Y = peso que o modelo previu.
ax1 = fig.add_subplot(gs[0])
ax1.scatter(y_test, y_pred, alpha=0.35, s=18, color="#534AB7")

# Linha de previsão perfeita (y = x)
lims = [min(y_test.min(), y_pred.min()) - 10,
        max(y_test.max(), y_pred.max()) + 10]
ax1.plot(lims, lims, "r--", linewidth=1.4, label="Previsão perfeita")

ax1.set_xlabel("Peso real (g)", fontsize=10)
ax1.set_ylabel("Peso previsto (g)", fontsize=10)
ax1.set_title(f"Real vs Previsto   R² = {r2:.3f}", fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_facecolor('#FAFAF8')
ax1.grid(True, linestyle='--', alpha=0.35)
ax1.spines[['top','right']].set_visible(False)


# --- Gráfico 2: Importância das features ---
# O Random Forest calcula automaticamente quais as variáveis que mais
# influenciaram as previsões durante o treino.
# Valores mais altos = variável mais importante para o modelo.
importancias = pd.Series(
    modelo.feature_importances_, index=FEATURES
).sort_values()

# A barra mais importante fica destacada a azul escuro
colors = ["#534AB7" if v == importancias.max() else "#AFA9EC" for v in importancias]

ax2 = fig.add_subplot(gs[1])
bars = ax2.barh(importancias.index, importancias.values, color=colors, edgecolor='none')

ax2.set_xlabel("Importância relativa", fontsize=10)
ax2.set_title("Importância das features", fontsize=11, fontweight='bold')
ax2.set_facecolor('#FAFAF8')
ax2.grid(True, axis='x', linestyle='--', alpha=0.35)
ax2.spines[['top','right']].set_visible(False)

# Adicionar o valor numérico ao lado de cada barra
for bar, val in zip(bars, importancias.values):
    ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va='center', fontsize=8.5)

fig.suptitle("Avaliação do modelo — Random Forest Regressor",
             fontsize=12, fontweight='bold', y=1.02)

# Guardar o gráfico como imagem PNG
plt.savefig(BASE / "avaliacao_modelo.png", dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Gráfico de avaliação guardado.")