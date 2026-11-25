import gc
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Previsão Inteligente de FIPE com IA",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
PARQUET_PATH = BASE_DIR / "fipex-prices-latest.parquet"
YOLO_PATH = BASE_DIR / "runs" / "detect" / "train-9" / "weights" / "best.pt"
MODELO_PATH = BASE_DIR / "modelo_fipe_rf.joblib"
UPLOAD_DIR = BASE_DIR / "upload_temp"
UPLOAD_DIR.mkdir(exist_ok=True)

PESOS_GRAVIDADE = {
    "Scratch": 0.015,  # 1,5%
    "Paint chip": 0.015,  # 1,5%
    "Dent": 0.030,  # 3,0%
    "Broken part": 0.045,  # 4,5%
    "Missing part": 0.050,  # 5,0%
}
DESCONTO_MAXIMO = 0.40


@st.cache_data

def carregar_base_fipe(caminho_parquet: Path) -> pd.DataFrame:
  colunas_necessarias = [
      "tipo_veiculo",
      "nome_marca",
      "nome_modelo",
      "ano_modelo",
      "nome_combustivel",
      "valor_formatado",
  ]
  df = pd.read_parquet(caminho_parquet, columns=colunas_necessarias)

  df_carros = df[
      df["tipo_veiculo"].astype(str).str.lower().str.strip() == "carro"
  ]
  del df

  preco_limpo = (
      df_carros["valor_formatado"]
      .astype(str)
      .str.replace("R$", "", regex=False)
      .str.replace(".", "", regex=False)
      .str.replace(",", ".", regex=False)
      .str.strip()
  )

  df_model = df_carros[[
      "nome_marca",
      "nome_modelo",
      "ano_modelo",
      "nome_combustivel",
  ]].copy()
  del df_carros

  df_model["preco_num"] = pd.to_numeric(preco_limpo, errors="coerce")
  df_model = df_model.dropna(subset=[
      "nome_marca",
      "nome_modelo",
      "ano_modelo",
      "nome_combustivel",
      "preco_num",
  ])

  df_model["eh_1_0"] = (
      df_model["nome_modelo"]
      .astype(str)
      .str.lower()
      .str.contains(r"1[.,]0", regex=True)
      .astype(int)
  )

  df_model["marca_cod"] = df_model["nome_marca"].astype("category").cat.codes
  df_model["modelo_cod"] = df_model["nome_modelo"].astype("category").cat.codes
  df_model["combustivel_cod"] = (
      df_model["nome_combustivel"].astype("category").cat.codes
  )

  return df_model


@st.cache_resource
def carregar_modelos(df_model: pd.DataFrame):
  if MODELO_PATH.exists():
    modelo_rf = joblib.load(MODELO_PATH)
  else:
    features = [
        "marca_cod",
        "modelo_cod",
        "ano_modelo",
        "combustivel_cod",
        "eh_1_0",
    ]
    X = df_model[features]
    y = df_model["preco_num"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    modelo_rf = RandomForestRegressor(
        n_estimators=20, random_state=42, n_jobs=-1
    )
    modelo_rf.fit(X_train, y_train)
    joblib.dump(modelo_rf, MODELO_PATH)

  modelo_yolo = YOLO(str(YOLO_PATH))
  return modelo_rf, modelo_yolo


st.markdown(
    """
    <style>
        .main-header {
            padding: 1.2rem 1.5rem;
            border-radius: 14px;
            background: linear-gradient(90deg, #111827 0%, #1f2937 100%);
            color: white;
            margin-bottom: 1.2rem;
            border: 1px solid #374151;
        }
        .motivo-box {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            background-color: #111827;
            border-left: 5px solid #10b981;
            color: #f9fafb;
            margin-top: 0.8rem;
        }
    </style>
    <div class="main-header">
        <h1 style="margin:0; font-size: 1.8rem;">🚗 Avaliador Inteligente de Veículos (FIPE + YOLOv8)</h1>
        <p style="margin:0.3rem 0 0 0; color:#9ca3af;">
            Estimativa de tabela FIPE com Machine Learning e desconto automático por avarias detectadas na foto.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not PARQUET_PATH.exists():
  st.error(f"Arquivo não encontrado: {PARQUET_PATH}")
  st.stop()

if not YOLO_PATH.exists():
  st.error(f"Peso do YOLO não encontrado: {YOLO_PATH}")
  st.stop()

try:
  with st.spinner("Carregando base FIPE e modelos de IA..."):
    df_model = carregar_base_fipe(PARQUET_PATH)
    gc.collect()
    modelo_fipe, modelo_yolo = carregar_modelos(df_model)
    gc.collect()
except Exception as e:
  st.error("Erro ao carregar base ou modelos:")
  st.exception(e)
  st.stop()

col_esq, col_dir = st.columns([1, 1.35], gap="large")

with col_esq:
  st.subheader("📋 Dados do Veículo")
  marcas = sorted(df_model["nome_marca"].dropna().unique().tolist())
  marca_escolhida = st.selectbox("Marca", marcas)

  modelos_da_marca = sorted(
      df_model[df_model["nome_marca"] == marca_escolhida]["nome_modelo"]
      .dropna()
      .unique()
      .tolist()
  )
  modelo_escolhido = st.selectbox("Modelo", modelos_da_marca)

  c1, c2 = st.columns(2)
  with c1:
    ano_escolhido = st.number_input(
        "Ano do Modelo", min_value=1980, max_value=2027, value=2020, step=1
    )
  with c2:
    cambio_escolhido = st.selectbox("Câmbio", ["Automático", "Manual"])

  confianca = st.slider(
      "Sensibilidade da IA (confiança)", 0.05, 0.80, 0.15, 0.05
  )

  st.subheader("📸 Foto para Inspeção de Avarias")
  arquivo_foto = st.file_uploader(
      "Clique ou arraste a foto do carro",
      type=["jpg", "jpeg", "png", "webp"],
  )

  calcular_btn = st.button(
      "Calcular Preço estimado de Venda", type="primary", use_container_width=True
  )

with col_dir:
  st.subheader("Resultado da Avaliação")

  if calcular_btn:
    if arquivo_foto is None:
      st.warning("Envia uma foto do carro antes de calcular.")
    else:
      caminho_foto = UPLOAD_DIR / arquivo_foto.name
      caminho_foto.write_bytes(arquivo_foto.getvalue())

      veiculo_filtrado = df_model[
          (df_model["nome_marca"] == marca_escolhida)
          & (df_model["nome_modelo"] == modelo_escolhido)
      ]

      if veiculo_filtrado.empty:
        st.error("Veículo não encontrado com essa combinação.")
      else:
        cod_marca = int(veiculo_filtrado["marca_cod"].iloc[0])
        cod_modelo = int(veiculo_filtrado["modelo_cod"].iloc[0])
        cod_combustivel = int(veiculo_filtrado["combustivel_cod"].iloc[0])
        eh_1_0 = int(any(veiculo_filtrado["eh_1_0"]))

        entrada = pd.DataFrame([{
                    "marca_cod": cod_marca,
                    "modelo_cod": cod_modelo,
                    "ano_modelo": int(ano_escolhido),
                    "combustivel_cod": cod_combustivel,
                    "eh_1_0": eh_1_0,
                }])

        try:
                    with st.spinner("Calculando valor FIPE e analisando avarias na foto..."):
                        fipe_prevista = float(modelo_fipe.predict(entrada)[0])
                        resultados = modelo_yolo.predict(
                        source=str(caminho_foto),
                        conf=confianca,
                        imgsz=320,
                        device="cpu",
                        save=False,
                        verbose=False,
                    )
        except Exception as e:
                    st.error("Deu erro na hora de calcular / rodar a IA:")
                    st.exception(e)
                    st.stop()

        contagem = {}
        desconto_bruto = 0.0
        imagem_anotada_rgb = None

        for r in resultados:
          imagem_anotada_rgb = r.plot()[:, :, ::-1]  # BGR -> RGB
          for box in r.boxes:
            cls_id = int(box.cls[0])
            nome_classe = modelo_yolo.names[cls_id]
            contagem[nome_classe] = contagem.get(nome_classe, 0) + 1
            desconto_bruto += PESOS_GRAVIDADE.get(nome_classe, 0.02)

        del resultados
        gc.collect()

        desconto_final = min(desconto_bruto, DESCONTO_MAXIMO)
        sugestao_anuncio = fipe_prevista * (1.0 - desconto_final)
        diferenca_reais = sugestao_anuncio - fipe_prevista

        total_avarias = sum(contagem.values())
        if total_avarias == 0:
          motivo = (
              "Nenhuma avaria foi detectada na foto; o valor ficou igual da"
              " FIPE prevista."
          )
        else:
          resumo_classes = ", ".join(
              [f"{qtd}x {cls}" for cls, qtd in contagem.items()]
          )
          teto_txt = (
              f" (limitado ao teto de {DESCONTO_MAXIMO*100:.0f}%)"
              if desconto_bruto > DESCONTO_MAXIMO
              else ""
          )
          motivo = (
              f"O valor baixou {desconto_final*100:.1f}% por causa de"
              f" {total_avarias} avaria(s) detectada(s) na imagem:"
              f" {resumo_classes}{teto_txt}."
          )

        st.markdown(
            f"**Veículo:** {marca_escolhida} — {modelo_escolhido} | **Ano:**"
            f" {ano_escolhido} | **Câmbio:** {cambio_escolhido}"
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Tabela FIPE (ML)", f"R$ {fipe_prevista:,.2f}")
        m2.metric("Desconto IA", f"{desconto_final*100:.1f}%")
        m3.metric(
            "Sugestão de Venda",
            f"R$ {sugestao_anuncio:,.2f}",
            delta=f"R$ {diferenca_reais:+,.2f}",
        )

        st.markdown(
            f"""
                <div class="motivo-box">
                    <strong>Motivo da avaliação:</strong><br>
                    {motivo}
                </div>
                """,
            unsafe_allow_html=True,
        )

        if imagem_anotada_rgb is not None:
          st.image(
              imagem_anotada_rgb,
              caption="Inspeção visual do YOLOv8",
              use_container_width=True,
          )