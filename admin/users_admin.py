from __future__ import annotations

import sqlite3
import streamlit as st

def render_users_admin(auth_user, get_user_by_username, create_user, update_user_status, update_user_role, update_user_password, list_users_df, log_event):
    st.subheader("Painel administrativo")
    tab1, tab2 = st.tabs(["Usuários", "Minha senha"])
    with tab1:
        st.markdown("### Cadastro interno de usuários")
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
        st.markdown("### Lista de usuários")
        users_df = list_users_df()
        st.dataframe(users_df, use_container_width=True, hide_index=True)
        usernames = [str(x) for x in users_df["username"].tolist()] if not users_df.empty else []
        if usernames:
            st.markdown("### Alterar usuário existente")
            sel_user = st.selectbox("Selecionar usuário", usernames)
            user_row = get_user_by_username(sel_user)
            col1, col2, col3 = st.columns(3)
            with col1:
                status_now = bool(user_row["is_active"])
                new_status = st.toggle("Ativo", value=status_now, key="admin_status_toggle")
                if st.button("Salvar status", use_container_width=True):
                    update_user_status(sel_user, new_status)
                    log_event(auth_user["username"], "update_user_status", f"user={sel_user}; active={new_status}")
                    st.success("Status atualizado.")
                    st.rerun()
            with col2:
                role_now = str(user_row["role"])
                new_user_role = st.selectbox("Papel", ["tecnico", "admin"], index=0 if role_now == "tecnico" else 1, key="admin_role_select")
                if st.button("Salvar papel", use_container_width=True):
                    update_user_role(sel_user, new_user_role)
                    log_event(auth_user["username"], "update_user_role", f"user={sel_user}; role={new_user_role}")
                    st.success("Papel atualizado.")
                    st.rerun()
            with col3:
                new_pwd = st.text_input("Nova senha", type="password")
                if st.button("Resetar senha", use_container_width=True):
                    if not new_pwd:
                        st.error("Informe a nova senha.")
                    else:
                        update_user_password(sel_user, new_pwd)
                        log_event(auth_user["username"], "reset_password", f"user={sel_user}")
                        st.success("Senha alterada.")
                        st.rerun()
    with tab2:
        st.markdown("### Alterar minha senha")
        st.caption("A troca com validação da senha atual continua no app.py.")
