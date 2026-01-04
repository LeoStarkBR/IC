# Painel de Monitoramento da Dengue - Bauru
# Versão Final: Gráficos Empilhados (Stack) e Chat com Scroll Independente

import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Painel Dengue Bauru",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Tentativa de importação da IA
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# =========================
# CARREGAMENTO DE DADOS
# =========================
@st.cache_data
def carregar_dados():
    try:
        dados = pd.read_csv("dados/dengue_bauru.csv")
        clima = pd.read_csv("dados/clima_bauru.csv")
        bairros = gpd.read_file("dados/bairros_bauru.geojson")
    except FileNotFoundError:
        dates = pd.date_range(start='2023-01-01', end='2025-12-31', freq='D')
        dados = pd.DataFrame({
            'data': dates,
            'bairro': np.random.choice(['Centro', 'Vila Universitária', 'Mary Dota', 'Geisel', 'Falcao', 'Redentor'], len(dates)),
            'casos': np.random.poisson(5, len(dates))
        })
        clima = pd.DataFrame({
            'data': dates,
            'chuva_mm': np.random.gamma(2, 10, len(dates)),
            'temperatura_media': np.random.normal(28, 4, len(dates))
        })
        bairros = gpd.GeoDataFrame() 
    return dados, clima, bairros

def preparar_dados(dados, clima):
    dados['data'] = pd.to_datetime(dados['data'])
    clima['data'] = pd.to_datetime(clima['data'])
    df = dados.merge(clima, on='data', how='left')
    df['mes'] = df['data'].dt.month
    df['ano'] = df['data'].dt.year
    return df

dados_raw, clima_raw, bairros_raw = carregar_dados()
df = preparar_dados(dados_raw, clima_raw)

# =========================
# CABEÇALHO E FILTROS
# =========================
st.title("Painel de Monitoramento da Dengue")
st.markdown("### Inteligência Epidemiológica")

with st.container(border=True):
    col_f1, col_f2 = st.columns(2)
    
    lista_bairros = sorted(df['bairro'].unique().tolist())
    lista_bairros.insert(0, "Todos")
    lista_anos = sorted(df['ano'].unique().tolist(), reverse=True)
    lista_anos.insert(0, "Todos")

    with col_f1:
        bairro_selecionado = st.selectbox("Filtro de Bairro", lista_bairros)
    with col_f2:
        ano_selecionado = st.selectbox("Filtro de Ano", lista_anos)

# =========================
# LÓGICA DE DADOS
# =========================
df_filtrado = df.copy()

if bairro_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['bairro'] == bairro_selecionado]
if ano_selecionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado['ano'] == ano_selecionado]

# Cálculos de Delta
delta_casos, delta_media, delta_pico, delta_chuva = None, None, None, None
cor_delta_casos, cor_delta_pico = "off", "off"

if ano_selecionado != "Todos":
    ano_anterior = int(ano_selecionado) - 1
    df_prev = df[df['ano'] == ano_anterior]
    if bairro_selecionado != "Todos":
        df_prev = df_prev[df_prev['bairro'] == bairro_selecionado]
    
    total_atual = df_filtrado['casos'].sum()
    total_prev = df_prev['casos'].sum() if not df_prev.empty else 0
    diff_casos = total_atual - total_prev
    
    if total_prev > 0:
        perc = (diff_casos / total_prev) * 100
        delta_casos = f"{diff_casos} ({perc:+.1f}%)"
    else:
        delta_casos = f"{diff_casos}"
    cor_delta_casos = "inverse"

    media_atual = df_filtrado['casos'].mean()
    media_prev = df_prev['casos'].mean() if not df_prev.empty else 0
    delta_media = round(media_atual - media_prev, 2)

    pico_atual = df_filtrado['casos'].max()
    pico_prev = df_prev['casos'].max() if not df_prev.empty else 0
    delta_pico = int(pico_atual - pico_prev)
    cor_delta_pico = "inverse" 

    chuva_atual = df_filtrado['chuva_mm'].sum()
    chuva_prev = df_prev['chuva_mm'].sum() if not df_prev.empty else 0
    delta_chuva = f"{int(chuva_atual - chuva_prev)} mm"

# =========================
# INDICADORES (FULL WIDTH)
# =========================
st.markdown("#### Indicadores Gerais")
l1, l2, l3, l4 = st.columns(4)

with l1:
    with st.container(border=True):
        st.metric("Total Casos", int(df_filtrado['casos'].sum()), delta=delta_casos, delta_color=cor_delta_casos)
with l2:
    with st.container(border=True):
        val_media = df_filtrado['casos'].mean()
        st.metric("Média/Dia", round(val_media, 2) if not pd.isna(val_media) else 0, delta=delta_media, delta_color=cor_delta_casos)
with l3:
    with st.container(border=True):
        val_max = df_filtrado['casos'].max()
        st.metric("Pico (Dia)", int(val_max) if not pd.isna(val_max) else 0, delta=delta_pico, delta_color=cor_delta_pico)
with l4:
    with st.container(border=True):
        val_chuva = df_filtrado['chuva_mm'].sum()
        st.metric("Chuva (mm)", int(val_chuva), delta=delta_chuva, delta_color="normal")

st.divider()

# =========================
# LAYOUT PRINCIPAL: DASHBOARD (Esq) vs IA (Dir)
# =========================
col_dashboard, col_ia = st.columns([2.5, 1]) 

# ---------------------------------------------------------
# COLUNA ESQUERDA: GRÁFICOS EMPILHADOS (STACK)
# ---------------------------------------------------------
with col_dashboard:
    
    # 1. Gráfico de Linha (Tendência)
    with st.container(border=True):
        st.markdown("##### Evolução Temporal")
        df_linha = df_filtrado.groupby('data', as_index=False)['casos'].sum()
        fig_linha = px.line(df_linha, x='data', y='casos', markers=True)
        fig_linha.update_layout(height=350, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_linha, use_container_width=True)

    # 2. Gráfico Combo (Clima)
    with st.container(border=True):
        st.markdown("##### 🌦️ Influência Climática")
        df_combo = df_filtrado.groupby('data', as_index=False).agg({'casos': 'sum', 'chuva_mm': 'mean'}).sort_values('data')
        fig_combo = go.Figure()
        fig_combo.add_trace(go.Bar(x=df_combo['data'], y=df_combo['chuva_mm'], name='Chuva', marker_color='#A0C4FF', opacity=0.5, yaxis='y2'))
        fig_combo.add_trace(go.Scatter(x=df_combo['data'], y=df_combo['casos'], name='Casos', mode='lines', line=dict(color='#FF6B6B', width=2)))
        
        fig_combo.update_layout(
            height=350,
            yaxis=dict(title="Casos", side="left"),
            yaxis2=dict(title="Chuva", side="right", overlaying="y", showgrid=False),
            legend=dict(x=0, y=1.1, orientation='h'),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_combo, use_container_width=True)

    # 3. Módulo Geográfico (Mapa + Tabela)
    with st.container(border=True):
        st.markdown("##### Análise Geográfica e Risco")
        
        col_mapa, col_tabela = st.columns([1, 1])
        
        # Mapa na esquerda
        if not bairros_raw.empty:
                df_mapa = df_filtrado.groupby('bairro', as_index=False)['casos'].sum()
                mapa = bairros_raw.merge(df_mapa, left_on='nome', right_on='bairro')
                if not mapa.empty:
                    fig_map = px.choropleth_mapbox(
                        mapa, geojson=mapa.geometry, locations=mapa.index, color='casos',
                        mapbox_style="carto-positron", zoom=10, center={"lat": -22.3145, "lon": -49.0586},
                        opacity=0.6
                    )
                    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=300)
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.warning("Sem dados.")
        else:
                st.warning("GeoJSON ausente.")

        # Tabela na direita (Corrigida)
        resumo = df_filtrado.groupby('bairro').agg({'casos':'mean', 'chuva_mm':'mean'}).reset_index()
        def calc_risco(r):
                if r['casos'] > 50 and r['chuva_mm'] > 120: return 'Alto 🔴'
                elif r['casos'] > 20: return 'Médio 🟡'
                else: return 'Baixo 🟢'
            
        resumo['Risco'] = resumo.apply(calc_risco, axis=1)
            # Renomeação explícita para evitar KeyError
        resumo.columns = ['Bairro', 'Média Casos', 'Média Chuva', 'Risco']
            
        st.dataframe(
                resumo[['Bairro', 'Risco', 'Média Casos']].sort_values('Média Casos', ascending=False), 
                use_container_width=True, 
                hide_index=True,
        )


# ---------------------------------------------------------
# COLUNA DIREITA: MÓDULO IA COM SCROLL INDEPENDENTE
# ---------------------------------------------------------
with col_ia:

    with st.container(border=True):
        st.subheader(" Assistente IA")
    
        # 1. Configuração (Sempre no topo)
        with st.expander(" Chave API"):
            api_key = st.text_input("Gemini Key", type="password", label_visibility="collapsed")

        if HAS_GENAI and api_key:
            try:
                genai.configure(api_key=api_key)
                try:
                    modelos = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    nome_modelo = modelos[0] if modelos else "models/gemini-1.5-flash"
                except:
                    nome_modelo = "models/gemini-1.5-flash"
                
                st.caption(f"Modelo: {nome_modelo}")

                # 2. JANELA DE HISTÓRICO (Com Scroll Próprio)
                # O st.container(height=...) cria a barra de rolagem interna
                historico_container = st.container(height=600, border=True)

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Renderiza as mensagens DENTRO da janela de histórico
                with historico_container:
                    for msg in st.session_state.messages:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

                # 3. INPUT (Estilo Pill)
                # O chat_input fica fixo no rodapé da coluna.
                if prompt := st.chat_input("Pergunte sobre os dados..."):
                    
                    # Renderiza pergunta do usuário na janela
                    with historico_container:
                        st.chat_message("user").markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    # Lógica IA
                    stats = df_filtrado.describe().to_string()
                    top_bairros = ""
                    if bairro_selecionado == "Todos":
                        ranking = df_filtrado.groupby('bairro')['casos'].sum().sort_values(ascending=False).head(5)
                        top_bairros = f"\nTOP 5 BAIRROS:\n{ranking.to_string()}"

                    contexto = f"""
                    Analista Dengue Bauru, com respostas curtas e coesas para facilitar o entendimento dos usuários. 
                    Filtros: {bairro_selecionado} ({ano_selecionado}).
                    Dados: {stats}
                    {top_bairros}
                    Pergunta: {prompt}
                    """

                    with historico_container:
                        with st.chat_message("assistant"):
                            with st.spinner("..."):
                                try:
                                    model = genai.GenerativeModel(nome_modelo)
                                    resp = model.generate_content(contexto)
                                    st.markdown(resp.text)
                                    st.session_state.messages.append({"role": "assistant", "content": resp.text})
                                except Exception as e:
                                    st.error("Erro IA.")
            
            except Exception as e:
                st.error(f"Erro Config: {e}")
        else:
            st.info("Configure a API Key acima.")