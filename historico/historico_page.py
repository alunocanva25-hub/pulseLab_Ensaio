from __future__ import annotations

import pandas as pd
import streamlit as st

def render_historico_page():
    st.subheader("Histórico local")
    historico = st.session_state.get("historico_local", [])
    if not historico:
        st.info("Ainda não há ensaios registrados.")
        return
    st.caption(f"Total de registros: {len(historico)}")
    resumo = pd.DataFrame(historico)
    cols_existentes = [c for c in ["datahora", "usuario", "captura_modo", "classe", "pulsos", "erro", "status", "robustez"] if c in resumo.columns]
    if cols_existentes:
        st.dataframe(resumo[cols_existentes], use_container_width=True, hide_index=True)
    st.markdown("---")
    st.markdown("### Detalhamento")
    for i, item in enumerate(historico):
        titulo = f"{item.get('datahora', '-')} | {item.get('status', '-')} | erro {item.get('erro', '-')}%"
        with st.expander(titulo, expanded=(i == 0)):
            df = pd.DataFrame([{"Campo": k, "Valor": v} for k, v in item.items()])
            st.dataframe(df, use_container_width=True, hide_index=True)
