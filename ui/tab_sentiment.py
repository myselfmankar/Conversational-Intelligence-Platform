import streamlit as st
from visuals import charts 

def render_sentiment_tab(df_display):
    st.subheader("Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_pie = charts.plot_sentiment_distribution_pie(df_display)
        if fig_pie: st.plotly_chart(fig_pie, use_container_width=True)
        else: st.info("No sentiment data to plot with current filters.")
    
    with col2:
        fig_author_sent = charts.plot_sentiment_per_author(df_display)
        if fig_author_sent: st.plotly_chart(fig_author_sent, use_container_width=True)
        else: st.info("No sentiment data per author to plot with current filters.")