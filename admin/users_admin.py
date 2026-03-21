from __future__ import annotations
import sqlite3
import streamlit as st

def render_users_admin(auth_user, get_user_by_username, create_user, update_user_status, update_user_role, update_user_password, list_users_df, log_event):
    st.subheader("Painel administrativo")
    with st.form("create_user_form"):
        new_username = st.text_input("Usuário")
        new_full_name = st.text_input("Nome completo")
        new_password = st.text_input("Senha inicial", type="password")
        new_role = st.selectbox("Papel", ["tecnico", "admin"])
        new_active = st.toggle("Ativo", value=True)
        save_user = st.form_submit_button("Criar usuário")
    if save_user:
        try:
            if not new_username.strip() or not new_full_name.strip() or not new_password:
                st.error("Preencha usuário, nome e senha.")
            else:
                create_user(new_username.strip().lower(), new_full_name.strip(), new_password, new_role, new_active)
                log_event(auth_user["username"], "create_user", f"user={new_username.strip().lower()}; role={new_role}; active={new_active}")
                st.success("Usuário criado com sucesso.")
                st.rerun()
        except sqlite3.IntegrityError:
            st.error("Esse usuário já existe.")
    users_df = list_users_df()
    st.dataframe(users_df, use_container_width=True, hide_index=True)
