import numpy as np #  gera números aleatórios e faz os cálculos matemáticos
import pandas as pd # organizar os dados numa tabela e guarda o CSV

np.random.seed(42) # números aleatórios gerados são os mesmos toda vez que rodar o código, para garantir reprodutibilidade
N = 1000

# --- Features ---

# Cada linha gera 1000 valores aleatórios 
# O uniform distribui os valores entre o mínimo e máximo. O randint faz o mesmo para números inteiros (dias).
temperatura    = np.random.uniform(18, 30, N)        # °C  (ótimo ~24°C)
oxigenio       = np.random.uniform(5, 10, N)         # mg/L
racao_diaria   = np.random.uniform(10, 80, N)        # g/dia
dias_criacao   = np.random.randint(30, 365, N)       # dias desde o início do lote
densidade      = np.random.uniform(5, 40, N)         # peixes/m³
ph             = np.random.uniform(6.5, 8.5, N)      # pH

# --- Peso (target) com relações biológicas realistas ---
# Efeito principal: dias de criação + ração
peso = (
    0.8  * dias_criacao              # crescimento diário base
  + 1.2  * racao_diaria              # mais ração → mais peso
  - 4.5  * np.abs(temperatura - 24)  # penalização por afastamento do ótimo
  + 8.0  * oxigenio                  # mais O2 → metabolismo melhor
  - 3.0  * densidade                 # mais peixe/m3 → competição por recursos
  - 20.0 * np.abs(ph - 7.2)         # penalização por pH fora do ótimo
  + np.random.normal(0, 25, N)      # ruído realista
)

# Garantir que o peso mínimo é positivo (peixe pequeno ao início)
peso = np.clip(peso, 50, 1800)

# --- DataFrame ---
df = pd.DataFrame({ # Junta tudo numa tabela com colunas nomeadas e guarda em CSV
    "temperatura_agua_C":      np.round(temperatura, 1),
    "oxigenio_dissolvido_mgL": np.round(oxigenio, 2),
    "racao_diaria_g":          np.round(racao_diaria, 1),
    "dias_desde_criacao":      dias_criacao,
    "densidade_tanque_pm3":    np.round(densidade, 1),
    "ph_agua":                 np.round(ph, 2),
    "peso_g":                  np.round(peso, 1),
})

df.to_csv("/home/paulo-victor/projects/piscicultura/dados_piscicultura.csv", index=False)
print("Dataset gerado com sucesso!")
print(df.describe().round(2).to_string())
