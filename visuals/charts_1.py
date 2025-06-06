import plotly.express as px
import pandas as pd
from collections import Counter


def plot_message_activity_timeline(df_display: pd.DataFrame):
    """Generates an interactive line chart for daily message activity using Plotly."""
    if 'datetime' not in df_display.columns or df_display['datetime'].isnull().all():
        return None
    
    df_copy = df_display.copy()
    df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
    
    # Handle potential timezone issues for resampling
    if df_copy['datetime'].dt.tz is not None:
        df_copy['datetime_naive'] = df_copy['datetime'].dt.tz_localize(None)
        daily_activity = df_copy.set_index('datetime_naive').resample('D')['message'].count().reset_index()
        date_column = 'datetime_naive'
    else:
        daily_activity = df_copy.set_index('datetime').resample('D')['message'].count().reset_index()
        date_column = 'datetime'
    
    daily_activity.rename(columns={'message': 'Number of Messages', date_column: 'Date'}, inplace=True)

    if daily_activity.empty:
        return None

    fig = px.line(daily_activity, x='Date', y='Number of Messages', 
                  title="Daily Message Count")
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Messages")
    return fig

def plot_hourly_activity(df_display: pd.DataFrame):
    """Generates an interactive bar chart for hourly message activity using Plotly."""
    if 'datetime' not in df_display.columns or df_display['datetime'].isnull().all():
        return None
    
    df_copy = df_display.copy()
    df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
    hourly_activity_counts = df_copy['datetime'].dt.hour.value_counts().sort_index()
    
    # Create a DataFrame for Plotly Express
    hourly_activity_df = pd.DataFrame({
        'Hour of Day': hourly_activity_counts.index,
        'Number of Messages': hourly_activity_counts.values
    })

    if hourly_activity_df.empty:
        return None

    fig = px.bar(hourly_activity_df, x='Hour of Day', y='Number of Messages', 
                 title="Messages by Hour of Day")
    fig.update_layout(xaxis_title="Hour of Day (0-23)", yaxis_title="Number of Messages",
                      xaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 1)) # Ensure all hours are shown
    return fig

def plot_sentiment_distribution_pie(df_display: pd.DataFrame):
    """Generates an interactive pie chart for overall sentiment distribution using Plotly."""
    # Filter out entries where sentiment might be 'ERROR' or default 'NEUTRAL' for non-text
    sentiment_counts = df_display[
        (df_display['sentiment_label'] != 'ERROR') & 
        (df_display['message_type'] == 'text') # Only consider actual text messages for sentiment pie
    ]['sentiment_label'].value_counts()
    
    if not sentiment_counts.empty:
        fig = px.pie(sentiment_counts, values=sentiment_counts.values, 
                     names=sentiment_counts.index, title="Overall Sentiment Distribution (Text Messages)")
        return fig
    return None


# def plot_frequent_named_entities(df_display: pd.DataFrame, top_n=20):
#     """Generates an interactive bar chart for the most frequent named entities using Plotly."""
#     all_entities = []
#     # Ensure 'entities' column exists and contains lists
#     if 'entities' not in df_display.columns:
#         return None

#     for entity_list in df_display['entities']:
#         if isinstance(entity_list, list):
#             for entity in entity_list:
#                 # Check if entity is a dict and has the required keys
#                 if isinstance(entity, dict) and 'entity_group' in entity and 'word' in entity:
#                     if entity['entity_group'] in ['PER', 'LOC', 'ORG', 'MISC']:
#                          all_entities.append(f"{entity['word']} ({entity['entity_group']})")
    
#     if all_entities:
#         entity_counts = Counter(all_entities).most_common(top_n)
#         df_entities = pd.DataFrame(entity_counts, columns=['Entity', 'Count'])
#         if df_entities.empty:
#             return None
#         fig = px.bar(df_entities, x='Count', y='Entity', orientation='h', 
#                      title=f"Most Frequent Named Entities (Top {top_n})")
#         fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Show highest count at the top
#         return fig
#     return None

def plot_sentiment_per_author(df_display: pd.DataFrame):
    """Generates an interactive bar chart for sentiment distribution per author using Plotly."""
    author_sentiment = df_display[
        (~df_display['is_system']) & 
        (df_display['sentiment_label'] != 'ERROR') &
        (df_display['message_type'] == 'text')
    ].groupby('author')['sentiment_label'].value_counts(normalize=True).mul(100).rename('percentage').round(1).reset_index()
    
    if not author_sentiment.empty:
        # Ensure the 'author' column is treated as a string/category for plotting
        author_sentiment['author'] = author_sentiment['author'].astype(str) # <--- ADD THIS LINE

        fig = px.bar(author_sentiment, x='author', y='percentage', color='sentiment_label', 
                     title="Sentiment Distribution per Author (Text Messages, %)", 
                     labels={'percentage':'Percentage of Messages', 'author':'Author', 'sentiment_label':'Sentiment'},
                     barmode='group')
        # To ensure all categories are shown and not treated numerically if they still look like numbers
        fig.update_xaxes(type='category') # <--- ADD THIS LINE
        return fig
    return None

def plot_author_activity(df_display: pd.DataFrame, top_n=10):
    """Generates an interactive bar chart for top N most active authors using Plotly."""
    author_msg_counts = df_display[~df_display['is_system']]['author'].value_counts().reset_index()
    author_msg_counts.columns = ['Author', 'Message Count'] 
    
    if not author_msg_counts.empty:
        # Ensure the 'Author' column is treated as a string/category
        author_msg_counts['Author'] = author_msg_counts['Author'].astype(str) # <--- ADD THIS LINE

        fig = px.bar(author_msg_counts.head(top_n), x='Author', y='Message Count', 
                     title=f"Top {top_n} Most Active Authors")
        # To ensure all categories are shown and not treated numerically
        fig.update_xaxes(type='category') # <--- ADD THIS LINE
        return fig
    return None

def plot_average_message_length(df_display: pd.DataFrame, top_n=10):
    """Generates an interactive bar chart for top N authors by average message length using Plotly."""
    df_copy = df_display.copy()
    
    if 'message' not in df_copy.columns:
        return None
        
    df_copy['message_length'] = df_copy['message'].apply(lambda x: len(str(x)))
    avg_len_author = df_copy[~df_copy['is_system']].groupby('author')['message_length'].mean().round(1).sort_values(ascending=False).reset_index()
    avg_len_author.columns = ['Author', 'Average Message Length']
    
    if not avg_len_author.empty:
        # Ensure the 'Author' column is treated as a string/category
        avg_len_author['Author'] = avg_len_author['Author'].astype(str) # <--- ADD THIS LINE

        fig = px.bar(avg_len_author.head(top_n), x='Author', y='Average Message Length', 
                     title=f"Top {top_n} Authors by Average Message Length")
        # To ensure all categories are shown and not treated numerically
        fig.update_xaxes(type='category') # <--- ADD THIS LINE
        return fig
    return None

def get_ranked_author_activity_df(df_display: pd.DataFrame, top_n=10) -> pd.DataFrame:
    """Prepares a DataFrame for a ranked list of most active authors."""
    if df_display.empty or 'author' not in df_display.columns:
        return pd.DataFrame(columns=['Rank', 'Author', 'Message Count']).set_index('Rank')

    author_msg_counts = df_display[~df_display['is_system']]['author'].value_counts().reset_index()
    # value_counts().reset_index() gives 'index' (for authors) and 'author' (for counts)
    # if series name was 'author'. Let's be explicit:
    author_msg_counts.columns = ['Author', 'Message Count'] 
    
    if not author_msg_counts.empty:
        top_authors = author_msg_counts.head(top_n).copy() # Use .copy() to avoid SettingWithCopyWarning
        top_authors['Author'] = top_authors['Author'].astype(str) # Ensure author is string
        top_authors.insert(0, 'Rank', range(1, 1 + len(top_authors)))
        return top_authors.set_index('Rank')
    return pd.DataFrame(columns=['Rank', 'Author', 'Message Count']).set_index('Rank')


def get_ranked_avg_message_length_df(df_display: pd.DataFrame, top_n=10) -> pd.DataFrame:
    """Prepares a DataFrame for a ranked list of authors by average message length."""
    if df_display.empty or 'author' not in df_display.columns or 'message' not in df_display.columns:
        return pd.DataFrame(columns=['Rank', 'Author', 'Average Message Length']).set_index('Rank')
        
    df_copy = df_display.copy()
    df_copy['message_length'] = df_copy['message'].apply(lambda x: len(str(x)))
    avg_len_author = df_copy[~df_copy['is_system']].groupby('author')['message_length'].mean().round(1).sort_values(ascending=False).reset_index()
    avg_len_author.columns = ['Author', 'Average Message Length'] 
    
    if not avg_len_author.empty:
        top_authors_avg_len = avg_len_author.head(top_n).copy()
        top_authors_avg_len['Author'] = top_authors_avg_len['Author'].astype(str)
        top_authors_avg_len.insert(0, 'Rank', range(1, 1 + len(top_authors_avg_len)))
        return top_authors_avg_len.set_index('Rank')