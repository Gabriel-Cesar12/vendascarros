# vendascarros

Aplicativo Streamlit para estimar o valor de um veiculo com dados FIPE,
Random Forest e deteccao de avarias com YOLOv8.

## Ambiente local (Windows / PowerShell)

Use Python 3.11 ou superior. Crie o ambiente virtual:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

O comando instala as versões CPU do PyTorch; a aplicação executa a inferência
YOLO na CPU. Usar o executável diretamente evita depender da política de
execução do PowerShell para ativar `Activate.ps1`.

Inicie a aplicação Streamlit com:

```powershell
.\.venv\Scripts\python.exe -m streamlit run index.py
```

A aplicação lê `fipex-prices-latest.parquet`, `modelo_fipe_rf.joblib` e os pesos
YOLO em `runs/detect/train-9/weights/best.pt`, relativos à pasta do projeto.
O notebook `analisefipe.ipynb` também requer as dependências listadas em
`requirements.txt`.

## Hospedagem

A aplicacao deve ser publicada no Streamlit Community Cloud, nao diretamente
na Vercel. Streamlit precisa de um processo Python persistente e de WebSockets.
Consulte o procedimento em [docs/DEPLOY.md](docs/DEPLOY.md).