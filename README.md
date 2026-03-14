# PulseLab v5 - Admin interno + IA LED experimental

## O que esta versão entrega
- Login interno com SQLite
- Bootstrap do primeiro admin dentro do próprio app
- Admin controla usuários sem editar Secrets
- Cadastro interno de usuários
- Ativar / bloquear usuários
- Alterar papel (técnico/admin)
- Reset de senha
- Auditoria local
- Módulo de ensaio estilo v4
- IA experimental de detecção de LED por captura

## Limitação
Esta versão NÃO faz contagem contínua automática por vídeo.
A "IA de pulso" aqui é experimental e assistida por imagem capturada.

## Rodar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar
Pode publicar direto no Streamlit Cloud. Como a autenticação é interna, não depende de OIDC.

## Primeiro acesso
No primeiro acesso, o sistema vai pedir para criar o administrador master.

## Próximo passo técnico
Para contagem real automática em vídeo:
- streamlit-webrtc
- OpenCV
- debounce óptico contínuo
