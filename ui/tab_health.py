import streamlit as st 
from visuals import charts 

def render_health_tab(df_display):
    st.subheader("Community Health & Moderation Dashboard")
    st.info("This dashboard helps identify potentially harmful content and recognizes positive community members.")

    st.markdown("---")

    # Section 1: Flagged Messages
    st.markdown("#### ⚠️ Messages Flagged for Review")
    st.caption("Messages automatically flagged as potentially toxic, sorted by confidence score.")

    if 'toxicity_label' in df_display.columns:
        flagged_df = df_display[df_display['toxicity_label'] == 'toxic'].sort_values('toxicity_score', ascending=False)
        
        if not flagged_df.empty:
            st.dataframe(
                flagged_df[['datetime', 'author', 'message', 'toxicity_score']],
                use_container_width=True,
                column_config={"toxicity_score": st.column_config.ProgressColumn("Toxicity Score", format="%.2f", min_value=0, max_value=1)}
            )
        else:
            st.success("✅ No toxic messages were detected in the current selection.")
    else:
        st.warning("Toxicity analysis was not performed. Data not available.")

    st.markdown("---")
    
    # Section 2: Community Champions
    st.markdown("#### 🏆 Community Champions Leaderboard")
    st.caption("Users ranked by a 'Contribution Score' based on their activity and positivity.")

    champions_df = charts.get_community_champions_df(df_display, top_n=10)
    if not champions_df.empty:
        st.dataframe(champions_df, use_container_width=True)
    else:
        st.info("Not enough data to rank community champions in the current selection.")