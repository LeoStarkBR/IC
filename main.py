# Painel de Monitoramento da Dengue - Bauru

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
    except FileNotFoundError as e:
        st.error(
            f"Erro ao carregar os dados.\n"
            f"Arquivo não encontrado: {e.filename}\n\n"
            "Verifique se a pasta 'dados/' e os arquivos CSV/GeoJSON existem."
        )
        st.stop()
    except Exception as e:
        st.error(
            "Erro inesperado ao carregar os dados.\n\n"
            f"Detalhes: {e}"
        )
        st.stop()

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
# COLUNA ESQUERDA: GRÁFICOS EMPILHADOS
# ---------------------------------------------------------
with col_dashboard:
    
    # 1. Gráfico de Linha
    with st.container(border=True):
        st.markdown("##### Evolução Temporal")
        df_linha = df_filtrado.groupby('data', as_index=False)['casos'].sum()
        fig_linha = px.line(df_linha, x='data', y='casos', markers=True)
        fig_linha.update_layout(height=350, margin=dict(l=20, r=20, t=10, b=20))
        st.plotly_chart(fig_linha, use_container_width=True)

   # 2. Gráfico Combo (Atualizado com Temperatura)
    with st.container(border=True):
        st.markdown("##### Influência Climática")
        
        # 1. Adicionamos 'temperatura_media' na agregação
        df_combo = df_filtrado.groupby('data', as_index=False).agg({
            'casos': 'sum', 
            'chuva_mm': 'mean', 
            'temperatura_media': 'mean'
        }).sort_values('data')
        
        fig_combo = go.Figure()
        
        # Barras de Chuva (Eixo Y2 - Direita)
        fig_combo.add_trace(go.Bar(
            x=df_combo['data'], 
            y=df_combo['chuva_mm'], 
            name='Chuva (mm)', 
            marker_color='#A0C4FF', 
            opacity=0.5, 
            yaxis='y2'
        ))
        
        # Nova Linha: Temperatura Média (Eixo Y2 - Direita)
        # Compartilha o eixo da chuva para manter a correlação climática
        fig_combo.add_trace(go.Scatter(
            x=df_combo['data'], 
            y=df_combo['temperatura_media'], 
            name='Temp. Média (°C)', 
            mode='lines', 
            line=dict(color='#FFA500', width=2, dash='dot'), # Laranja e pontilhada para destacar
            yaxis='y2' 
        ))

        # Linha de Casos (Eixo Y1 - Esquerda)
        fig_combo.add_trace(go.Scatter(
            x=df_combo['data'], 
            y=df_combo['casos'], 
            name='Casos', 
            mode='lines', 
            line=dict(color='#FF6B6B', width=2)
        ))
        
        fig_combo.update_layout(
            height=350,
            yaxis=dict(title="Casos", side="left"),
            # Atualizei o título do eixo direito para refletir as duas métricas
            yaxis2=dict(title="Clima (mm / °C)", side="right", overlaying="y", showgrid=False),
            legend=dict(x=0, y=1.1, orientation='h'),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_combo, use_container_width=True)

    # 3. Módulo Geográfico
    with st.container(border=True):
        st.markdown("##### Análise Geográfica e Risco")
        
        col_mapa, col_tabela = st.columns([1, 1])
        
        # Mapa
        with col_mapa:
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

        # Tabela (Pré-calculo do resumo para usar no chat depois)
        with col_tabela:
            resumo = df_filtrado.groupby('bairro').agg({'casos':'mean', 'chuva_mm':'mean'}).reset_index()
            def calc_risco(r):
                if r['casos'] > 50 and r['chuva_mm'] > 120: return 'Alto 🔴'
                elif r['casos'] > 20: return 'Médio 🟡'
                else: return 'Baixo 🟢'
            
            resumo['Risco'] = resumo.apply(calc_risco, axis=1)
            resumo.columns = ['Bairro', 'Média Casos', 'Média Chuva', 'Risco']
            
            st.dataframe(
                resumo[['Bairro', 'Risco', 'Média Casos']].sort_values('Média Casos', ascending=False), 
                use_container_width=True, 
                hide_index=True,
                height=300
            )


# ---------------------------------------------------------
# COLUNA DIREITA: MÓDULO IA
# ---------------------------------------------------------
with col_ia:
    with st.container(border=True, height=850):
        st.subheader("💬 Assistente IA")
    
        with st.expander("⚙️ Chave API"):
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

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                if prompt := st.chat_input("Pergunte sobre os dados..."):
                    st.chat_message("user").markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    # --- ENGENHARIA DE CONTEXTO (AQUI ESTÁ A MÁGICA) ---
                    
                    # 1. Resumo Estatístico Geral (Para perguntas genéricas)
                    stats = df_filtrado.describe().to_string()
                    
                    # 2. Visão do Gráfico de Linha (Agrupado por Mês para a IA entender a tendência)
                    # Convertemos a série temporal em texto
                    df_temporal = df_filtrado.set_index('data').resample('ME')['casos'].sum()
                    vis_temporal = f"TENDÊNCIA MENSAL DE CASOS:\n{df_temporal.to_string()}"

                    # 3. Visão do Mapa e Tabela de Risco (Passamos a tabela inteira)
                    vis_geografica = f"RISCO POR BAIRRO:\n{resumo.to_string(index=False)}"

                    # 4. Dados Climáticos (Correlação)
                    corr = df_filtrado['casos'].corr(df_filtrado['chuva_mm'])
                    vis_clima = f"CORRELAÇÃO CHUVA vs CASOS: {corr:.2f} (Escala -1 a 1)"

                    # Montagem do Prompt Rico
                    contexto = f"""
                    Você é um Epidemiologista Senior analisando dados de Dengue em Bauru.
                    O usuário está vendo gráficos na tela e você deve explicar o que eles mostram.
                    
                    FILTROS ATUAIS: {bairro_selecionado} ({ano_selecionado})

                    [DADOS DO GRÁFICO DE LINHA - TEMPO]
                    {vis_temporal}

                    [DADOS DO MAPA E TABELA - LOCAIS]
                    {vis_geografica}

                    [DADOS DE CLIMA]
                    {vis_clima}

                    [ESTATÍSTICAS GERAIS]
                    {stats}
                    
                    PERGUNTA DO USUÁRIO: {prompt}
                    
                    Responda de forma direta, citando números e meses específicos que aparecem nos dados acima.
                    """

                    with st.chat_message("assistant"):
                        with st.spinner("Analisando gráficos..."):
                            try:
                                model = genai.GenerativeModel(nome_modelo)
                                resp = model.generate_content(contexto)
                                st.markdown(resp.text)
                                st.session_state.messages.append({"role": "assistant", "content": resp.text})
                            except Exception as e:
                                st.error(f"Erro IA: {e}")
            
            except Exception as e:
                st.error(f"Erro Config: {e}")
        else:
            st.info("Configure a API Key acima.")