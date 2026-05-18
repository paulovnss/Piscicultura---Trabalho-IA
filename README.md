# 🐟 AquaPredict — Previsão de Crescimento em Piscicultura

> **Projeto académico** — Prova de Conceito (POC) de Machine Learning  
> Startup fictícia de consultoria ML · Cliente: empresa de piscicultura

---

## Índice

1. [O que é este projeto?](#1-o-que-é-este-projeto)
2. [Problema que resolve](#2-problema-que-resolve)
3. [Estrutura de ficheiros](#3-estrutura-de-ficheiros)
4. [Como instalar](#4-como-instalar)
5. [Como correr — passo a passo](#5-como-correr--passo-a-passo)
6. [Como funciona cada ficheiro](#6-como-funciona-cada-ficheiro)
7. [O modelo de Machine Learning](#7-o-modelo-de-machine-learning)
8. [A web app](#8-a-web-app)
9. [Resultados do modelo](#9-resultados-do-modelo)
10. [Perguntas frequentes](#10-perguntas-frequentes)

---

## 1. O que é este projeto?

Este projeto é uma **Prova de Conceito (POC)** desenvolvida para demonstrar como o Machine Learning pode ser aplicado numa empresa de piscicultura.

O objetivo é convencer o cliente de que é possível prever o **peso dos peixes** com base em parâmetros do tanque — sem precisar de pesar os peixes manualmente todos os dias.

O projeto inclui:
- Geração de dados artificiais que simulam uma piscicultura real
- Treino de um modelo de regressão com esses dados
- Uma web app interativa que demonstra o modelo ao cliente

---

## 2. Problema que resolve

Numa piscicultura, saber quando os peixes atingem o peso ideal para venda é crítico:

- **Colheita cedo** → peixe abaixo do peso → perda de receita
- **Colheita tarde** → custos de alimentação desnecessários + risco de sobrelotação
- **Pesar manualmente** → stressante para os peixes + mão de obra cara e muitas vezes não eficiente

**Solução:** um modelo de ML que prevê o peso com base em dados já monitorizados (temperatura, oxigénio, ração, etc.), sem tocar nos peixes.

---

## 3. Estrutura de ficheiros

```
piscicultura_ml/
│
├── gerar_dados.py              # Gera o dataset artificial (CSV)
├── treinar_modelo.py           # Treina o modelo e guarda o .pkl
├── app.py                      # Web app interativa (Streamlit)
│
├── dados_piscicultura.csv      # Gerado por gerar_dados.py
├── modelo.pkl                  # Gerado por treinar_modelo.py
│
└── README.md                   # Este ficheiro
```

> **Nota:** `dados_piscicultura.csv` e `modelo.pkl` são gerados automaticamente ao correr os scripts. Não precisas de os criar manualmente.

---

## 4. Como instalar

### Pré-requisitos
- Python 3.8 ou superior
- pip (gestor de pacotes Python)

### Instalar dependências

Abre o terminal na pasta do projeto e corre:

```bash
pip install streamlit scikit-learn joblib pandas numpy matplotlib seaborn
```

Ou, se tiveres um ficheiro `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Conteúdo do requirements.txt (opcional)

```
streamlit
scikit-learn
joblib
pandas
numpy
matplotlib
seaborn
```

---

## 5. Como correr — passo a passo

### Passo 1 — Gerar os dados

```bash
python gerar_dados.py
```

- Cria o ficheiro `dados_piscicultura.csv` com 1000 registos sintéticos

### Passo 2 — Treinar o modelo

```bash
python treinar_modelo.py
```

- Lê o CSV gerado no passo anterior
- Treina o modelo de Random Forest
- Guarda o modelo em `modelo.pkl`
- Mostra as métricas de avaliação no terminal (R² e MAE)
- Gera o gráfico `avaliacao_modelo.png`

### Passo 3 — Lançar a web app

```bash
streamlit run app.py
```

- O browser abre automaticamente em `http://localhost:8501`
- A app fica disponível enquanto o terminal estiver aberto
- Para parar a app: `Ctrl + C` no terminal

---

## 6. Como funciona cada ficheiro

### `gerar_dados.py`

Cria um dataset artificial com 1000 registos que simulam medições reais de um tanque de piscicultura.

**Features geradas:**

| Coluna                    | Descrição                  | Intervalo    |
|---------------------------|----------------------------|--------------|
| `temperatura_agua_C`      | Temperatura da água        | 18–30 °C     |
| `oxigenio_dissolvido_mgL` | Oxigénio dissolvido        | 5–10 mg/L    |
| `racao_diaria_g`          | Ração dada por dia         | 10–80 g      |
| `dias_desde_criacao`      | Dias desde início do lote  | 30–365 dias  |
| `densidade_tanque_pm3`    | Peixes por metro cúbico    | 5–40 peixe/m³|
| `ph_agua`                 | pH da água                 | 6.5–8.5      |
| `peso_g`                  | Peso do peixe (**target**) | 50–1800 g    |

A fórmula que gera o peso inclui relações biológicas reais:
- Mais dias + mais ração → mais peso
- Temperatura afastada de 24°C → menos peso
- pH afastado de 7.2 → menos peso
- Mais peixes/m³ → menos peso (competição)
- "Ruído aleatório" → simula variação natural entre peixes

---

### `treinar_modelo.py`

Treina um modelo de Machine Learning supervisionado com os dados do CSV.

**Fluxo:**

```
CSV (1000 linhas)
      ↓
X (6 features) + y (peso_g)
      ↓
80% treino / 20% teste
      ↓
RandomForestRegressor.fit(X_train, y_train)
      ↓
Avaliar: R² e MAE nos dados de teste
      ↓
Guardar modelo.pkl
```

**Porquê Random Forest?**
- Lida bem com features em escalas diferentes (não precisa de normalização)
- Robusto a outliers
- Calcula automaticamente a importância de cada feature
- Bom desempenho com datasets pequenos (como este)

---

### `app.py`

Web app construída com **Streamlit** que permite demonstrar o modelo ao cliente em tempo real.

**Como funciona:**

1. A app carrega o `modelo.pkl` ao iniciar (apenas uma vez, graças ao cache)
2. O utilizador move os sliders na sidebar
3. A cada movimento, o Streamlit re-executa o script inteiro
4. O modelo prevê o peso com os valores atuais dos sliders
5. Todos os gráficos e métricas atualizam instantaneamente

**O que a app mostra:**
- 4 métricas no topo: peso (g), peso (kg), crescimento/dia, dias para 500g
- Gráfico de projeção de crescimento do dia 30 ao dia 365
- Estado do tanque com badges coloridos (verde/amarelo/vermelho)
- Caixa de previsão com cor dinâmica conforme o peso
- Importância das features no modelo

---

## 7. O modelo de Machine Learning

### Tipo de problema
**Regressão** — prever um valor numérico contínuo (peso em gramas)

### Algoritmo
**Random Forest Regressor** — conjunto de 200 árvores de decisão independentes. A previsão final é a média das previsões de todas as árvores.

### Features de entrada (X)
As 6 variáveis do tanque listadas na secção do `gerar_dados.py`.

### Target (y)
`peso_g` — o peso do peixe em gramas.

### Divisão treino/teste
- **80%** dos dados (800 registos) usados para treinar
- **20%** dos dados (200 registos) usados para avaliar

### Importância das features (fixos)

| Feature              | Importância |
|----------------------|-------------|
| Dias desde criação   | 61.2%       |
| Ração diária         | 19.8%       |
| Densidade do tanque  | 7.1%        |
| Oxigénio dissolvido  | 5.8%        |
| Temperatura da água  | 3.8%        |
| pH da água           | 2.3%        |

---

## 8. A web app

### Como lançar

```bash
streamlit run app.py
```

### Interface

```
┌─────────────────────────────────────────────────────┐
│  Sidebar         │  Área principal                   │
│  ─────────       │  ──────────────                   │
│  Slider temp.    │  Hero banner                      │
│  Slider O₂       │                                   │
│  Slider ração    │  [Peso] [kg] [g/dia] [dias→500g] │
│  Slider dias     │                                   │
│  Slider dens.    │  Gráfico projeção | Estado tanque │
│  Slider pH       │                                   │
│                  │  Importância das features         │
└─────────────────────────────────────────────────────┘
```

### Tecnologias usadas
- **Streamlit** — framework da web app
- **Matplotlib** — gráficos
- **Joblib** — carregar o modelo
- **NumPy** — cálculos do array de input

---

## 9. Resultados do modelo (fixos)

| Métrica           | Valor      | Interpretação                            |
|-------------------|------------|------------------------------------------|
| **R²**            | **0.88**   | O modelo explica 88% da variação do peso |
| **MAE**           | **22.9 g** | Erra em média 23 gramas                  |
| Amostras treino   | 800        |
| Amostras teste    | 200        |

**R² de 0.88 é excelente para uma POC** — significa que o modelo capturou os padrões principais dos dados. Num projeto real com dados reais, este valor tenderia a ser ainda mais representativo.

---

## 10. Perguntas/Dúvidas

**Posso usar este projeto com dados reais?**  
Sim. Bastaria substituir o `dados_piscicultura.csv` por um ficheiro com dados reais no mesmo formato, e re-correr o `treinar_modelo.py`.


**Posso mudar o peso alvo de colheita (500g)?**  
Sim. No ficheiro `app.py`, procura a linha com `500 - peso_previsto` e substitui `500` pelo valor desejado.

---

*Projeto desenvolvido para a cadeira de Machine Learning — Prova de Conceito*
