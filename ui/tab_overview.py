import streamlit as st 
from visuals import charts 

def render_overview_tab(df_display):
    st.subheader("Message Volume")
    fig_daily = charts.plot_message_activity_timeline(df_display)
    if fig_daily: st.plotly_chart(fig_daily, use_container_width=True)

    fig_hourly = charts.plot_hourly_activity(df_display)
    if fig_hourly: st.plotly_chart(fig_hourly, use_container_width=True)
    
    st.subheader("Top Contributors")
    fig_auth_act =charts. plot_author_activity(df_display, top_n=10)
    if fig_auth_act: st.plotly_chart(fig_auth_act, use_container_width=True)

    st.subheader("Author Activity Ranking")
    ranked_activity_df = charts.get_ranked_author_activity_df(df_display, top_n=10)
    if not ranked_activity_df.empty: st.dataframe(ranked_activity_df, use_container_width=True)