# =============================================================================
# app.py
# Web app interativa para previsão de peso de peixes em piscicultura.
# Desenvolvida com Streamlit — permite demo ao cliente em tempo real.
#
# Para correr:
#   streamlit run app.py
# =============================================================================

import streamlit as st      # framework para construir a web app
import joblib               # carregar o modelo treinado (.pkl)
import numpy as np          # operações numéricas
import pandas as pd         # manipulação de dados
import matplotlib.pyplot as plt         # gráficos
import matplotlib.gridspec as gridspec  # layout de gráficos


# Título, ícone e layout
st.set_page_config(
    page_title="AquaPredict | Piscicultura Inteligente",
    page_icon="🐟",
    layout="wide",
)


# CSS (StreamLit custom styling)

# st.markdown() com unsafe_allow_html=True para isso.
st.markdown("""
<style>
/* Importar fontes do Google Fonts:
   - Syne: display/títulos — geométrica e moderna
   - Space Grotesk: corpo de texto — legível e técnica */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* Fundo geral da app */
.stApp { background: #0a1628; color: #e8f4f8; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1f3c;
    border-right: 1px solid #1a3a5c;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span { color: #a8c8e8 !important; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0d2a4a 0%, #0a3d6b 50%, #0d2a4a 100%);
    border: 1px solid #1a4a7a;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
/* Efeito decorativo de brilho no canto */
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,180,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem; font-weight: 800;
    color: #ffffff; margin: 0 0 8px 0; letter-spacing: -1px;
}
.hero p { color: #7ab8d8; font-size: 1rem; margin: 0; }

/* Cards de métricas */
.metric-card {
    background: #0d2a4a;
    border: 1px solid #1a4a7a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    color: #00b4ff; line-height: 1; margin-bottom: 4px;
}
.metric-label {
    font-size: 0.8rem; color: #5a8aaa;
    text-transform: uppercase; letter-spacing: 1px;
}

/* Badges de estado — verde / amarelo / vermelho */
.badge {
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-size: 0.78rem;
    font-weight: 600; letter-spacing: 0.5px;
}
.badge-ok   { background: #0d3a2a; color: #4ade80; border: 1px solid #1a6040; }
.badge-warn { background: #3a2a0d; color: #fbbf24; border: 1px solid #6a4a1a; }
.badge-bad  { background: #3a0d0d; color: #f87171; border: 1px solid #6a1a1a; }

/* Títulos de secção */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 700;
    color: #a8c8e8; text-transform: uppercase; letter-spacing: 2px;
    margin-bottom: 16px; padding-bottom: 8px;
    border-bottom: 1px solid #1a3a5c;
}

/* Caixa da previsão principal */
.prediction-box {
    background: linear-gradient(135deg, #0d3a2a, #0a2a1a);
    border: 2px solid #1a6040;
    border-radius: 16px; padding: 28px; text-align: center;
}
.prediction-main {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem; font-weight: 800;
    color: #4ade80; line-height: 1;
}
.prediction-sub { color: #5a8a6a; font-size: 0.85rem; margin-top: 8px; }

/* Caixas de informação dos parâmetros */
.info-box {
    background: #0d1f3c;
    border-left: 3px solid #00b4ff;  /* barra azul à esquerda */
    border-radius: 0 8px 8px 0;
    padding: 12px 16px; margin-bottom: 12px;
    font-size: 0.88rem; color: #a8c8e8;
}
</style>
""", unsafe_allow_html=True)

# 3. CARREGAR O MODELO TREINADO

# @st.cache_resource: guarda o modelo em memória após o primeiro carregamento.
# Sem isto, o Streamlit recarregaria o modelo a cada interação do utilizador,
# tornando a app muito lenta.
@st.cache_resource
def load_model():
    return joblib.load("modelo.pkl")

modelo = load_model()

# 4. SIDEBAR — SLIDERS DE INPUT

# ajuste dos 6 parâmetros do tanque
# Cada vez que um slider é movido, o Streamlit re-executa o script inteiro
with st.sidebar:
    st.markdown("### 🎛️ Parâmetros do Tanque")
    st.markdown("---")

    # Sintaxe: st.slider(label, min, max, valor_inicial, passo)
    temperatura = st.slider("🌡️ Temperatura da água (°C)", 18.0, 30.0, 24.0, 0.1)
    oxigenio    = st.slider("💧 Oxigénio dissolvido (mg/L)", 5.0, 10.0, 7.5, 0.1)
    racao       = st.slider("🍽️ Ração diária (g)", 10.0, 80.0, 45.0, 0.5)
    dias        = st.slider("📅 Dias desde criação", 30, 365, 180, 1)
    densidade   = st.slider("🐟 Densidade do tanque (peixe/m³)", 5.0, 40.0, 20.0, 0.5)
    ph          = st.slider("⚗️ pH da água", 6.5, 8.5, 7.2, 0.05)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#3a6a8a; text-align:center;'>
    AquaPredict v1.0<br>Powered by Random Forest
    </div>
    """, unsafe_allow_html=True)


# 5. PREVISÃO EM TEMPO REAL

# Monta o array de input com os valores atuais dos sliders.
# A ordem das colunas = a mesma usada no treino.
X_input = np.array([[temperatura, oxigenio, racao, dias, densidade, ph]])

# O modelo retorna um array — pegamos no primeiro (e único) valor
peso_previsto = modelo.predict(X_input)[0]
peso_kg = peso_previsto / 1000


# 6. FUNÇÕES AUXILIARES — ESTADO DO TANQUE

# Cada função avalia se o parâmetro está no intervalo óptimo, aceitável ou crítico.
# Retorna: (classe_CSS_do_badge, texto_descritivo)

def status_temperatura(t):
    if 22 <= t <= 26:                    return "badge-ok",   "Óptima"
    elif 20 <= t < 22 or 26 < t <= 28:  return "badge-warn", "Aceitável"
    else:                                return "badge-bad",  "Crítica"

def status_ph(p):
    if 7.0 <= p <= 7.5:                  return "badge-ok",   "Óptimo"
    elif 6.8 <= p < 7.0 or 7.5 < p <= 7.8: return "badge-warn", "Aceitável"
    else:                                return "badge-bad",  "Fora do intervalo"

def status_oxigenio(o):
    if o >= 7.0:    return "badge-ok",   "Adequado"
    elif o >= 6.0:  return "badge-warn", "Baixo"
    else:           return "badge-bad",  "Crítico"

# Banner
st.markdown("""
<div class="hero">
    <h1>🐟 AquaPredict</h1>
    <p>Sistema inteligente de previsão de crescimento para piscicultura · Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Métricas principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{peso_previsto:.0f}g</div>
        <div class="metric-label">Peso previsto</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{peso_kg:.3f}</div>
        <div class="metric-label">Peso em kg</div>
    </div>""", unsafe_allow_html=True)

with col3:
    # Taxa de crescimento diária = peso atual / dias decorridos
    crescimento_dia = peso_previsto / dias if dias > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{crescimento_dia:.1f}g</div>
        <div class="metric-label">Crescimento/dia</div>
    </div>""", unsafe_allow_html=True)

with col4:
    # Estimar quantos dias faltam para atingir 500g (peso típico de venda)
    # Se o peixe já passou de 500g, mostra "Pronto!"
    dias_colheita = max(0, int((500 - peso_previsto) / max(crescimento_dia, 0.1)))
    label_colheita = f"{dias_colheita}d" if dias_colheita > 0 else "Pronto!"
    cor_colheita = '#4ade80' if dias_colheita == 0 else '#00b4ff'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{cor_colheita}">{label_colheita}</div>
        <div class="metric-label">Para 500g (alvo)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# LAYOUT EM DUAS COLUNAS: GRÁFICO | ESTADO DO TANQUE + PREVISÃO

col_esq, col_dir = st.columns([1.2, 1]) # coluna da esquerda 

with col_esq:
    st.markdown('<div class="section-title">📊 Projeção de Crescimento</div>',
                unsafe_allow_html=True)

    # Projetar o crescimento do dia 30 ao dia 365, fixando todos os outros
    # parâmetros iguais aos sliders — apenas variamos os dias
    dias_proj  = np.arange(30, 366, 5)
    pesos_proj = []
    for d in dias_proj:
        x = np.array([[temperatura, oxigenio, racao, d, densidade, ph]])
        pesos_proj.append(modelo.predict(x)[0])

    # Construir o gráfico com fundo a condizer com o tema da app
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor('#0a1628')
    ax.set_facecolor('#0d2a4a')

    ax.fill_between(dias_proj, pesos_proj, alpha=0.15, color='#00b4ff')  # área preenchida
    ax.plot(dias_proj, pesos_proj, color='#00b4ff', linewidth=2.2)        # curva de crescimento

    # Marcadores da posição atual do utilizador no gráfico
    ax.axvline(dias, color='#4ade80', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.axhline(peso_previsto, color='#4ade80', linewidth=1, linestyle=':', alpha=0.6)

    # Linha de referência para o peso alvo de colheita
    ax.axhline(500, color='#fbbf24', linewidth=1.2, linestyle='--',
               alpha=0.7, label='Alvo 500g')

    # Ponto destacado na posição atual (dia × peso)
    ax.scatter([dias], [peso_previsto], color='#4ade80', s=80, zorder=5)

    # Estilização dos eixos e grelha
    ax.set_xlabel("Dias desde criação", color='#5a8aaa', fontsize=9)
    ax.set_ylabel("Peso previsto (g)", color='#5a8aaa', fontsize=9)
    ax.tick_params(colors='#5a8aaa', labelsize=8)
    ax.spines['bottom'].set_color('#1a3a5c')
    ax.spines['left'].set_color('#1a3a5c')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, facecolor='#0d2a4a', edgecolor='#1a4a7a', labelcolor='#a8c8e8')
    ax.grid(True, linestyle='--', alpha=0.15, color='#ffffff')

    plt.tight_layout()
    st.pyplot(fig)   # renderizar o gráfico no Streamlit
    plt.close()      # libertar memória 


with col_dir:
    st.markdown('<div class="section-title">🔬 Estado do Tanque</div>',
                unsafe_allow_html=True)

    # Obter classificação de cada parâmetro
    cls_t, lbl_t = status_temperatura(temperatura)
    cls_p, lbl_p = status_ph(ph)
    cls_o, lbl_o = status_oxigenio(oxigenio)

    # Renderizar as caixas de estado com badges coloridos
    st.markdown(f"""
    <div class="info-box">
        🌡️ <b>Temperatura</b> &nbsp;
        <span class="badge {cls_t}">{lbl_t}</span> &nbsp; {temperatura}°C
        <br><small style='color:#3a6a8a'>Óptimo: 22–26°C para a maioria das espécies</small>
    </div>
    <div class="info-box">
        ⚗️ <b>pH da água</b> &nbsp;
        <span class="badge {cls_p}">{lbl_p}</span> &nbsp; {ph}
        <br><small style='color:#3a6a8a'>Óptimo: 7.0–7.5</small>
    </div>
    <div class="info-box">
        💧 <b>Oxigénio dissolvido</b> &nbsp;
        <span class="badge {cls_o}">{lbl_o}</span> &nbsp; {oxigenio} mg/L
        <br><small style='color:#3a6a8a'>Mínimo recomendado: 6.0 mg/L</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Previsão</div>',
                unsafe_allow_html=True)

    # A cor muda conforme o peso previsto:
    # verde (≥400g) → próximo do alvo | azul (≥200g) → em crescimento | amarelo → ainda pequeno
    cor_peso = "#4ade80" if peso_previsto >= 400 else \
               "#00b4ff" if peso_previsto >= 200 else "#fbbf24"

    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-main" style="color:{cor_peso}">{peso_previsto:.0f} g</div>
        <div class="prediction-sub">ao dia {dias} de criação</div>
        <br>
        <div style='font-size:0.8rem; color:#3a6a8a;'>
            Margem de erro do modelo: ± 23g &nbsp;|&nbsp; R² = 0.88
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# quais os fatores que o modelo considera mais relevantes.
st.markdown('<div class="section-title">🧠 O que o modelo valoriza</div>',
            unsafe_allow_html=True)

importancias = {
    "Dias de criação":  0.612,   # domina — principal fator de crescimento
    "Ração diária":     0.198,   # segundo mais importante
    "Densidade tanque": 0.071,
    "Oxigénio":         0.058,
    "Temperatura":      0.038,
    "pH":               0.023,
}

fig2, ax2 = plt.subplots(figsize=(10, 2.2))
fig2.patch.set_facecolor('#0a1628')
ax2.set_facecolor('#0a1628')

nomes  = list(importancias.keys())
vals   = list(importancias.values())
# Destaque azul brilhante para a variável mais importante
colors = ['#00b4ff' if v == max(vals) else '#1a4a7a' for v in vals]

bars = ax2.barh(nomes, vals, color=colors, height=0.55, edgecolor='none')

# Valor percentual ao lado de cada barra
for bar, val in zip(bars, vals):
    ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f"{val:.1%}", va='center', color='#a8c8e8', fontsize=8.5)

ax2.set_xlim(0, 0.75)
ax2.tick_params(colors='#5a8aaa', labelsize=8.5)
ax2.spines[:].set_visible(False)
ax2.grid(False)

plt.tight_layout()
st.pyplot(fig2)
plt.close()