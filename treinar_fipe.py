from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
PARQUET_PATH = BASE_DIR / "fipex-prices-latest.parquet"
MODELO_PATH = BASE_DIR / "modelo_fipe_rf.joblib"

colunas_necessarias = [
    "tipo_veiculo",
    "nome_marca",
    "nome_modelo",
    "ano_modelo",
    "nome_combustivel",
    "valor_formatado",
]

df = pd.read_parquet(PARQUET_PATH, columns=colunas_necessarias)
df_carros = df[
    df["tipo_veiculo"].astype(str).str.lower().str.strip() == "carro"
].copy()
del df

preco_limpo = (
    df_carros["valor_formatado"]
    .astype(str)
    .str.replace("R$", "", regex=False)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.strip()
)

df_model = df_carros[
    ["nome_marca", "nome_modelo", "ano_modelo", "nome_combustivel"]
].copy()
del df_carros

df_model["preco_num"] = pd.to_numeric(preco_limpo, errors="coerce")
df_model = df_model.dropna(
    subset=[
        "nome_marca",
        "nome_modelo",
        "ano_modelo",
        "nome_combustivel",
        "preco_num",
    ]
)

df_model["eh_1_0"] = (
    df_model["nome_modelo"]
    .astype(str)
    .str.lower()
    .str.contains(r"1[.,]0", regex=True)
    .astype("int8")
)

df_model["marca_cod"] = (
    df_model["nome_marca"].astype("category").cat.codes.astype("int32")
)
df_model["modelo_cod"] = (
    df_model["nome_modelo"].astype("category").cat.codes.astype("int32")
)
df_model["combustivel_cod"] = (
    df_model["nome_combustivel"].astype("category").cat.codes.astype("int32")
)
df_model["ano_modelo"] = pd.to_numeric(
    df_model["ano_modelo"], errors="coerce"
).astype("int32")
df_model["preco_num"] = df_model["preco_num"].astype("float32")

features = [
    "marca_cod",
    "modelo_cod",
    "ano_modelo",
    "combustivel_cod",
    "eh_1_0",
]
X = df_model[features]
y = df_model["preco_num"]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)

joblib.dump(modelo_rf, MODELO_PATH, compress=3)
print("Modelo salvo em:", MODELO_PATH)