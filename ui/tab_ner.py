import streamlit as st
from visuals import charts 

def render_ner_tab(df_display):
    st.subheader("Named Entity Recognition (NER)")
    st.info("This section identifies the key people (PER), organizations (ORG), and locations (LOC) mentioned in the chat.")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_people = charts.plot_frequent_named_entities(df_display, top_n=15, entity_types=['PER'])
        if fig_people:
            st.plotly_chart(fig_people, use_container_width=True)
        else:
            st.caption("No 'Person' entities found in the current selection.")

    with col2:
        fig_orgs_locs = charts.plot_frequent_named_entities(df_display, top_n=15, entity_types=['ORG', 'LOC'])
        if fig_orgs_locs:
            st.plotly_chart(fig_orgs_locs, use_container_width=True)
        else:
            st.caption("No 'Organization' or 'Location' entities found.")