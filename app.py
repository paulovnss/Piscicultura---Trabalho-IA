import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Caminho relativo — funciona em qualquer computador
BASE = Path(__file__).parent

# Título, ícone e layout
st.set_page_config(
    page_title="AquaPredict | Piscicultura Inteligente",
    page_icon="🐟",
    layout="wide",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0a1628; color: #e8f4f8; }

section[data-testid="stSidebar"] {
    background: #0d1f3c;
    border-right: 1px solid #1a3a5c;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span { color: #a8c8e8 !important; }

.hero {
    background: linear-gradient(135deg, #0d2a4a 0%, #0a3d6b 50%, #0d2a4a 100%);
    border: 1px solid #1a4a7a;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
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

.badge {
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-size: 0.78rem;
    font-weight: 600; letter-spacing: 0.5px;
}
.badge-ok   { background: #0d3a2a; color: #4ade80; border: 1px solid #1a6040; }
.badge-warn { background: #3a2a0d; color: #fbbf24; border: 1px solid #6a4a1a; }
.badge-bad  { background: #3a0d0d; color: #f87171; border: 1px solid #6a1a1a; }

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 700;
    color: #a8c8e8; text-transform: uppercase; letter-spacing: 2px;
    margin-bottom: 16px; padding-bottom: 8px;
    border-bottom: 1px solid #1a3a5c;
}

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

.info-box {
    background: #0d1f3c;
    border-left: 3px solid #00b4ff;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px; margin-bottom: 12px;
    font-size: 0.88rem; color: #a8c8e8;
}

/* Caixa de meta personalizada */
.meta-box {
    background: #0d2a4a;
    border: 1px solid #1a4a7a;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
    font-size: 0.85rem;
    color: #a8c8e8;
    text-align: center;
}
.meta-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem; font-weight: 800;
    color: #fbbf24;
}
</style>
""", unsafe_allow_html=True)


# CARREGAR MODELO
@st.cache_resource
def load_model():
    return joblib.load(BASE / "modelo.pkl")

modelo = load_model()


# SIDEBAR — PARÂMETROS DO TANQUE + META DE PESO
with st.sidebar:
    st.markdown("### 🎛️ Parâmetros do Tanque")
    st.markdown("---")

    temperatura = st.slider("🌡️ Temperatura da água (°C)", 18.0, 30.0, 24.0, 0.1)
    oxigenio    = st.slider("💧 Oxigénio dissolvido (mg/L)", 5.0, 10.0, 7.5, 0.1)
    racao       = st.slider("🍽️ Ração diária (g)", 10.0, 80.0, 45.0, 0.5)
    dias        = st.slider("📅 Dias desde criação", 30, 730, 180, 1)
    densidade   = st.slider("🐟 Densidade do tanque (peixe/m³)", 5.0, 40.0, 20.0, 0.5)
    ph          = st.slider("⚗️ pH da água", 6.5, 8.5, 7.2, 0.05)

    st.markdown("---")

    # META DE PESO 
    st.markdown("### 🎯 Meta de Colheita")
    st.markdown("<small style='color:#3a6a8a'>Defina o peso alvo para colheita</small>",
                unsafe_allow_html=True)
    # Slider de 100g a 2000g, default 500g, passo 50g
    meta_peso = st.slider("⚖️ Peso alvo (g)", 100, 2000, 500, 50)
    st.markdown(f"""
    <div class='meta-box'>
        Meta definida<br>
        <span class='meta-value'>{meta_peso}g</span><br>
        <small style='color:#3a6a8a'>{meta_peso/1000:.2f} kg por peixe</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#3a6a8a; text-align:center;'>
    AquaPredict v1.0<br>Powered by Random Forest
    </div>
    """, unsafe_allow_html=True)


# PREVISÃO EM TEMPO REAL
X_input = np.array([[temperatura, oxigenio, racao, dias, densidade, ph]])
peso_previsto = modelo.predict(X_input)[0]
peso_kg = peso_previsto / 1000

# Taxa de crescimento e dias para atingir a META
ja_atingiu_meta = peso_previsto >= meta_peso

dias_para_meta = None
dia_meta = None

for d in range(dias, 731, 5):

    x = np.array([[
        temperatura,
        oxigenio,
        racao,
        d,
        densidade,
        ph
    ]])

    peso_futuro = modelo.predict(x)[0]

    if peso_futuro >= meta_peso:
        dia_meta = d
        dias_para_meta = d - dias
        break


if dias_para_meta is None:
    dias_para_meta = "N/A"
    dia_meta = None


# Mantém apenas para exibição
crescimento_dia = peso_previsto / max(dias, 1)


# FUNÇÕES DE ESTADO DO TANQUE
def status_temperatura(t):
    if 22 <= t <= 26:                       return "badge-ok",   "Óptima"
    elif 20 <= t < 22 or 26 < t <= 28:     return "badge-warn", "Aceitável"
    else:                                   return "badge-bad",  "Crítica"

def status_ph(p):
    if 7.0 <= p <= 7.5:                     return "badge-ok",   "Óptimo"
    elif 6.8 <= p < 7.0 or 7.5 < p <= 7.8: return "badge-warn", "Aceitável"
    else:                                   return "badge-bad",  "Fora do intervalo"

def status_oxigenio(o):
    if o >= 7.0:    return "badge-ok",   "Adequado"
    elif o >= 6.0:  return "badge-warn", "Baixo"
    else:           return "badge-bad",  "Crítico"


# HERO BANNER
st.markdown("""
<div class="hero">
    <h1>🐟 AquaPredict</h1>
    <p>Sistema inteligente de previsão de crescimento para piscicultura · Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# MÉTRICAS NO TOPO
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
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{crescimento_dia:.1f}g</div>
        <div class="metric-label">Crescimento/dia</div>
    </div>""", unsafe_allow_html=True)

with col4:
    # Agora usa a meta definida pelo utilizador
    if ja_atingiu_meta:
        label_meta = "Atingida ✓"
    elif dias_para_meta == "N/A":
        label_meta = "--"
    else:
        label_meta = f"{dias_para_meta}d"
    cor_meta   = '#4ade80' if ja_atingiu_meta else '#fbbf24'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{cor_meta}">{label_meta}</div>
        <div class="metric-label">Para meta ({meta_peso}g)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# GRÁFICO + ESTADO DO TANQUE
col_esq, col_dir = st.columns([1.2, 1])

with col_esq:
    st.markdown('<div class="section-title">📊 Projeção de Crescimento</div>',
                unsafe_allow_html=True)

    # Projetar até 730 dias (2 anos) para mostrar o impacto da meta de peso
    dias_proj  = np.arange(30, 731, 5)
    pesos_proj = []
    for d in dias_proj:
        x = np.array([[temperatura, oxigenio, racao, d, densidade, ph]])
        pesos_proj.append(modelo.predict(x)[0])

    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor('#0a1628')
    ax.set_facecolor('#0d2a4a')

    ax.fill_between(dias_proj, pesos_proj, alpha=0.15, color='#00b4ff')
    ax.plot(dias_proj, pesos_proj, color='#00b4ff', linewidth=2.2)

    # Posição atual
    ax.axvline(dias, color='#4ade80', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.axhline(peso_previsto, color='#4ade80', linewidth=1, linestyle=':', alpha=0.6)

    # Linha da META escolhida pelo utilizador (amarela, dinâmica)
    ax.axhline(meta_peso, color='#fbbf24', linewidth=1.5, linestyle='--',
               alpha=0.85, label=f'Meta: {meta_peso}g')

    ax.scatter([dias], [peso_previsto], color='#4ade80', s=80, zorder=5)

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
    st.pyplot(fig)
    plt.close()


with col_dir:
    st.markdown('<div class="section-title">🔬 Estado do Tanque</div>',
                unsafe_allow_html=True)

    cls_t, lbl_t = status_temperatura(temperatura)
    cls_p, lbl_p = status_ph(ph)
    cls_o, lbl_o = status_oxigenio(oxigenio)

    st.markdown(f"""
    <div class="info-box">
        🌡️ <b>Temperatura</b> &nbsp;
        <span class="badge {cls_t}">{lbl_t}</span> &nbsp; {temperatura}°C
        <br><small style='color:#3a6a8a'>Óptimo: 22–26°C</small>
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

    cor_peso = "#4ade80" if peso_previsto >= meta_peso else \
               "#00b4ff" if peso_previsto >= meta_peso * 0.6 else "#fbbf24"

    progresso = min(100, int((peso_previsto / meta_peso) * 100))

    if ja_atingiu_meta:
        texto_meta = f"Meta de {meta_peso}g já atingida."
    elif dia_meta is not None:
        texto_meta = (
            f"Meta prevista para o dia "
            f"{dia_meta} ({dias_para_meta} dias restantes)."
        )
    else:
        texto_meta = (
            f"Meta de {meta_peso}g não é atingida "
            f"até ao dia 730."
        )

    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-main" style="color:{cor_peso}">{peso_previsto:.0f} g</div>
        <div class="prediction-sub">ao dia {dias} de criação</div>
        <br>
        <div style='background:#0a1628; border-radius:8px; overflow:hidden; height:8px; margin-bottom:8px;'>
            <div style='background:{cor_peso}; width:{progresso}%; height:100%; border-radius:8px; transition:width 0.3s;'></div>
        </div>
        <div style='font-size:0.8rem; color:#5a8a6a;'>
        {progresso}% da meta ({meta_peso}g)
        </div>
        <br>
        <div style='font-size:0.75rem; color:#3a6a8a;'>
            Margem de erro: ± 23g &nbsp;|&nbsp; R² = 0.88
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
    🎯 {texto_meta}
    </div>
    """, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# IMPORTÂNCIA DAS FEATURES
st.markdown('<div class="section-title">🧠 O que o modelo valoriza</div>',
            unsafe_allow_html=True)

importancias = {
    "Dias de criação":  0.824,
    "Ração diária":     0.116,
    "Densidade tanque": 0.022,
    "Oxigénio":         0.020,
    "Temperatura":      0.010,
    "pH":               0.009,
}

fig2, ax2 = plt.subplots(figsize=(10, 2.2))
fig2.patch.set_facecolor('#0a1628')
ax2.set_facecolor('#0a1628')

nomes  = list(importancias.keys())
vals   = list(importancias.values())
colors = ['#00b4ff' if v == max(vals) else '#1a4a7a' for v in vals]

bars = ax2.barh(nomes, vals, color=colors, height=0.55, edgecolor='none')
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