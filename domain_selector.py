"""Domain Selection Page"""
import streamlit as st
from domain_config import get_all_domains

def show_domain_selector():
    st.set_page_config(page_title="e-KYC Domain Selector", layout="wide")
    st.title(" e-KYC Domain Selector")
    st.markdown("Select your sector/domain to proceed with KYC verification.")
    st.divider()
    
    domains = get_all_domains()
    col1, col2 = st.columns(2)
    
    domain_list = list(domains.items())
    for i, (domain_key, domain_info) in enumerate(domain_list[:2]):
        with col1:
            if st.button(f"{domain_info['name']}\n{domain_info['face_threshold']}%", key=f"btn_{domain_key}", use_container_width=True):
                st.session_state.selected_domain = domain_key
                st.session_state.domain_config = domain_info
                st.rerun()
    
    for i, (domain_key, domain_info) in enumerate(domain_list[2:]):
        with col2:
            if st.button(f"{domain_info['name']}\n{domain_info['face_threshold']}%", key=f"btn_{domain_key}", use_container_width=True):
                st.session_state.selected_domain = domain_key
                st.session_state.domain_config = domain_info
                st.rerun()

def main():
    if "selected_domain" not in st.session_state:
        st.session_state.selected_domain = None
    show_domain_selector()

if __name__ == "__main__":
    main()
