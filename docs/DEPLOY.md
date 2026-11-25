# Implantacao do Avaliador Inteligente FIPE + IA

## Plataforma recomendada

Use o Streamlit Community Cloud. O aplicativo depende de processos persistentes,
WebSockets e de arquivos locais para carregar os modelos. Essas caracteristicas
sao incompativeis com o modelo serverless da Vercel.

O projeto nao deve ser convertido para Vercel apenas alterando configuracoes:
um FastAPI ou Flask seria necessario, juntamente com uma estrategia de frontend,
armazenamento externo e cache dos modelos.

## Arquivos necessarios em producao

- `index.py`
- `requirements.txt`
- `packages.txt`
- `fipex-prices-latest.parquet`
- `modelo_fipe_rf.joblib`
- `runs/detect/train-9/weights/best.pt`

O notebook, o script de treinamento, os datasets de treinamento e os demais
pesos de execucoes nao sao necessarios para a interface publica.

## Conexao com o GitHub no Streamlit Community Cloud

1. Abra https://share.streamlit.io/.
2. Autentique-se com a conta do Streamlit.
3. Selecione o repositorio `Gabriel-Cesar12/vendascarros`.
4. Selecione a branch `main`.
5. Defina `index.py` como arquivo principal.
6. Confirme `requirements.txt` e `packages.txt`.
7. Inicie o deploy.

Nao faca deploy antes de revisar o status do Git e garantir que os arquivos
necessarios estejam publicados no repositorio.

## Atualizacoes

Envie as alteracoes para a branch configurada no Streamlit Community Cloud.
O aplicativo sera reconstruido automaticamente conforme a integracao do
projeto.

## Limites conhecidos

- O Streamlit requer uma sessao persistente e conexoes WebSocket.
- A imagem enviada pelo usuario deve respeitar os formatos definidos no app.
- Os modelos podem exigir mais memoria durante a inicializacao.
- Arquivos escritos em `upload_temp/` sao temporarios e podem ser perdidos
  quando o container reiniciar.
