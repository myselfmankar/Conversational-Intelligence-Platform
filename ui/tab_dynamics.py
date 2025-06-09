import streamlit as st 
from visuals import charts 

def render_dynamics_tab(df_display):
    st.subheader("Social Network Analysis")
    st.info("This graph visualizes communication patterns. An arrow from User A to User B means A often sent a message right before B. Larger nodes represent more central users.")
    
    fig_network = charts.create_interaction_network_graph(df_display)
    if fig_network:
        st.plotly_chart(fig_network, use_container_width=True)
    else:
        st.warning("Could not generate a network graph. The chat may be too short or have too few interactions in the current selection.")