# PulseLab v4 aprovado + segurança

Este kit usa **a v4 aprovada** do ensaio e adiciona por cima:
- login OIDC
- whitelist de e-mails
- admins
- painel admin
- bloqueio de usuários
- limite diário de ensaios
- auditoria local em SQLite

## Importante
- O ensaio é a **base da v4 aprovada**
- O SQLite local serve para MVP
- No Streamlit Community Cloud, armazenamento local não é persistente; para produção, use banco externo

## Secrets
Use o arquivo `.streamlit/secrets.example.toml` como modelo.

## Deploy
1. Suba estes arquivos para o GitHub
2. Faça deploy no Streamlit Cloud apontando para `app.py`
3. Cole os secrets no painel do app
4. Ajuste `redirect_uri` para sua URL publicada + `/oauth2callback`
