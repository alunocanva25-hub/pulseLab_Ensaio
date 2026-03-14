# PulseLab - Kit de Publicação no Streamlit Community Cloud

Este kit foi preparado para publicar o módulo de ensaio do PulseLab no Streamlit Community Cloud.

## Arquivos
- `app.py` -> app principal
- `requirements.txt` -> dependências Python
- `.streamlit/config.toml` -> configuração visual básica do Streamlit

## Publicar no Streamlit Community Cloud
1. Crie um repositório no GitHub.
2. Envie estes arquivos para a raiz do repositório.
3. Acesse o Streamlit Community Cloud.
4. Clique em **Deploy an app**.
5. Escolha seu repositório.
6. Em **Main file path**, selecione `app.py`.
7. Clique em **Deploy**.

## Rodar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Rodar localmente aceitando acesso pela rede
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## Testar no celular
- PC e celular devem estar na mesma rede Wi-Fi.
- Rode o app no PC.
- Abra no celular o endereço `http://IP_DO_PC:8501`.
- Se estiver publicado no Streamlit Cloud, abra a URL `https://seu-app.streamlit.app`.

## Observação importante
A câmera no navegador costuma funcionar melhor quando a app está publicada em HTTPS.
