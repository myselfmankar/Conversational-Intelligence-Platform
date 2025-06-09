import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import networkx as nx

# --- Activity Charts ---

def plot_message_activity_timeline(df_display: pd.DataFrame):
    """Generates an interactive line chart for daily message activity using Plotly."""
    if 'datetime' not in df_display.columns or df_display['datetime'].isnull().all():
        return None
    
    df_copy = df_display.copy()
    df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
    
    # Handle potential timezone issues for resampling
    if df_copy['datetime'].dt.tz is not None:
        daily_activity = df_copy.set_index('datetime').resample('D').size().reset_index(name='count')
    else:
        daily_activity = df_copy.set_index('datetime').resample('D').size().reset_index(name='count')
    
    daily_activity.rename(columns={'count': 'Number of Messages', 'datetime': 'Date'}, inplace=True)

    if daily_activity.empty:
        return None

    fig = px.line(daily_activity, x='Date', y='Number of Messages', title="Daily Message Count")
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Messages")
    return fig

def plot_hourly_activity(df_display: pd.DataFrame):
    """Generates an interactive bar chart for hourly message activity using Plotly."""
    if 'datetime' not in df_display.columns or df_display['datetime'].isnull().all():
        return None
    
    df_copy = df_display.copy()
    df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
    hourly_activity_counts = df_copy['datetime'].dt.hour.value_counts().sort_index()
    
    hourly_activity_df = pd.DataFrame({'Hour of Day': hourly_activity_counts.index, 'Number of Messages': hourly_activity_counts.values})

    if hourly_activity_df.empty:
        return None

    fig = px.bar(hourly_activity_df, x='Hour of Day', y='Number of Messages', title="Messages by Hour of Day")
    fig.update_layout(xaxis_title="Hour of Day (0-23)", yaxis_title="Number of Messages", xaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 1))
    return fig

# --- Sentiment Charts ---

def plot_sentiment_distribution_pie(df_display: pd.DataFrame):
    """Generates an interactive pie chart for overall sentiment distribution using Plotly."""
    sentiment_counts = df_display[(df_display['sentiment_label'] != 'ERROR') & (df_display['message_type'] == 'text')]['sentiment_label'].value_counts()
    
    if not sentiment_counts.empty:
        fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, title="Overall Sentiment Distribution")
        return fig
    return None

def plot_sentiment_per_author(df_display: pd.DataFrame):
    """Generates an interactive bar chart for sentiment distribution per author using Plotly."""
    author_sentiment = df_display[(~df_display['is_system']) & (df_display['sentiment_label'] != 'ERROR') & (df_display['message_type'] == 'text')].groupby('author')['sentiment_label'].value_counts(normalize=True).mul(100).rename('percentage').round(1).reset_index()
    
    if not author_sentiment.empty:
        author_sentiment['author'] = author_sentiment['author'].astype(str)
        fig = px.bar(author_sentiment, x='author', y='percentage', color='sentiment_label', title="Sentiment Distribution per Author (%)", labels={'percentage':'Percentage', 'author':'Author'}, barmode='group')
        fig.update_xaxes(type='category')
        return fig
    return None

# --- Author Stats ---

def plot_author_activity(df_display: pd.DataFrame, top_n=10):
    """Generates an interactive bar chart for top N most active authors using Plotly."""
    author_msg_counts = df_display[~df_display['is_system']]['author'].value_counts().nlargest(top_n).reset_index()
    author_msg_counts.columns = ['Author', 'Message Count'] 
    
    if not author_msg_counts.empty:
        author_msg_counts['Author'] = author_msg_counts['Author'].astype(str)
        fig = px.bar(author_msg_counts, x='Author', y='Message Count', title=f"Top {top_n} Most Active Authors")
        fig.update_xaxes(type='category')
        return fig
    return None

def get_ranked_author_activity_df(df_display: pd.DataFrame, top_n=10):
    """Prepares a ranked DataFrame of most active authors."""
    author_msg_counts = df_display[~df_display['is_system']]['author'].value_counts().reset_index()
    author_msg_counts.columns = ['Author', 'Message Count']
    top_authors = author_msg_counts.head(top_n).copy()
    top_authors.insert(0, 'Rank', range(1, len(top_authors) + 1))
    return top_authors.set_index('Rank')

# --- NEW: Named Entity Recognition (NER) Charts ---

def plot_frequent_named_entities(df_display: pd.DataFrame, top_n=20, entity_types=None):
    """Parses 'entities' and creates a bar chart for the most frequent named entities."""
    if 'entities' not in df_display.columns:
        return None
    if entity_types is None:
        entity_types = ['PER', 'ORG', 'LOC', 'MISC']

    all_entities = []
    for entity_list in df_display['entities'].dropna():
        if isinstance(entity_list, list):
            for entity in entity_list:
                if isinstance(entity, dict) and entity.get('entity_group') in entity_types:
                    word = entity.get('word', '').replace('##', '').strip()
                    if word:
                        all_entities.append(word)

    if not all_entities:
        return None

    entity_counts = Counter(all_entities).most_common(top_n)
    df_entities = pd.DataFrame(entity_counts, columns=['Entity', 'Count'])

    if df_entities.empty:
        return None

    title = f"Top {top_n} Most Frequent {'/'.join(entity_types)} Entities"
    fig = px.bar(df_entities, x='Count', y='Entity', orientation='h', title=title, labels={'Count': 'Frequency'})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig


def create_interaction_network_graph(df_display: pd.DataFrame):
    """Creates an interactive network graph of user interactions."""
    user_df = df_display[~df_display['is_system']].copy()
    if len(user_df) < 2:
        return None

    user_df['previous_author'] = user_df['author'].shift(1)
    interactions = user_df.dropna(subset=['previous_author'])
    interactions = interactions[interactions['author'] != interactions['previous_author']]

    if interactions.empty:
        return None

    interaction_counts = interactions.groupby(['previous_author', 'author']).size().reset_index(name='weight')
    G = nx.from_pandas_edgelist(interaction_counts, source='previous_author', target='author', edge_attr='weight', create_using=nx.DiGraph())

    if not G.nodes:
        return None

    pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y, node_text, node_size = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        degree = G.degree(node)
        node_x.append(x)
        node_y.append(y)
        node_size.append(10 + degree * 5)
        node_text.append(f"{node}<br># of connections: {degree}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(showscale=True, colorscale='YlGnBu', size=node_size, color=[G.degree(node) for node in G.nodes()],
                    colorbar=dict(thickness=15, title='Node Connections', xanchor='left'))
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Chat Social Network', showlegend=False, hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[dict(text="Network of who messages after whom. Node size and color indicate a user's centrality.",
                                      showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

def get_community_champions_df(df_display: pd.DataFrame, top_n=10):
    """
    Calculates a "Contribution Score" for each author and returns a ranked DataFrame.
    The score is based on message activity and positive sentiment ratio.
    """
    if df_display.empty or 'author' not in df_display.columns:
        return pd.DataFrame()

    user_df = df_display[~df_display['is_system']].copy()
    
    if user_df.empty:
        return pd.DataFrame()

    # 1. Calculate total messages per author
    author_activity = user_df['author'].value_counts().reset_index()
    author_activity.columns = ['Author', 'Message Count']

    # 2. Calculate positive sentiment ratio per author
    sentiment_ratios = user_df[user_df['sentiment_label'] == 'POSITIVE'].groupby('author').size() / user_df.groupby('author').size()
    sentiment_ratios = sentiment_ratios.fillna(0).reset_index(name='Positive Ratio')
    sentiment_ratios.columns = ['Author', 'Positive Ratio (%)']
    sentiment_ratios['Positive Ratio (%)'] = (sentiment_ratios['Positive Ratio (%)'] * 100).round(1)

    # 3. Merge the stats
    champions_df = pd.merge(author_activity, sentiment_ratios, on='Author', how='left').fillna(0)

    # 4. Normalize metrics and calculate a final score
    if not champions_df.empty:
        # Normalize between 0 and 1
        champions_df['Normalized Activity'] = champions_df['Message Count'] / champions_df['Message Count'].max()
        champions_df['Normalized Positivity'] = champions_df['Positive Ratio (%)'] / 100 # Already a 0-100 scale

        # Calculate a weighted score
        champions_df['Contribution Score'] = (0.4 * champions_df['Normalized Activity'] + 0.6 * champions_df['Normalized Positivity']) * 100
        champions_df = champions_df.sort_values(by='Contribution Score', ascending=False).head(top_n)

        # Format for display
        champions_df['Contribution Score'] = champions_df['Contribution Score'].round(1)
        champions_df.insert(0, 'Rank', range(1, len(champions_df) + 1))
        
        return champions_df[['Rank', 'Author', 'Message Count', 'Positive Ratio (%)', 'Contribution Score']].set_index('Rank')

    return pd.DataFrame()

def get_topic_metrics(df_display: pd.DataFrame, topic: str):
    """
    Analyzes the DataFrame for a specific topic and calculates key metrics.

    This function filters the DataFrame for messages containing the given topic
    and computes statistics like mention count, sentiment breakdown, and average
    toxicity.

    Args:
        df_display (pd.DataFrame): The DataFrame to analyze (can be pre-filtered).
        topic (str): The keyword/topic to search for (case-insensitive).

    Returns:
        dict: A dictionary containing the calculated metrics.
              Example: 
              {
                  'mentions': 50, 
                  'authors': 12, 
                  'positive_ratio': 65.5, 
                  'negative_ratio': 10.2, 
                  'avg_toxicity': 0.15
              }
        None: If the topic is not found or the input is invalid.
    """
    # --- Input Validation ---
    if df_display.empty or not isinstance(topic, str) or not topic.strip():
        return None

    # --- Filtering for the Topic ---
    # Use .str.contains() for a case-insensitive search.
    # na=False ensures that any potential NaN values in 'message' don't cause an error.
    topic_df = df_display[df_display['message'].str.contains(topic, case=False, na=False)]

    # If no messages contain the topic, there's nothing to analyze.
    if topic_df.empty:
        return None

    # --- Metric Calculation ---
    metrics = {
        'mentions': topic_df.shape[0],
        'authors': topic_df['author'].nunique(),
    }

    # --- Conditional Calculation for NLP-derived columns ---
    # These checks make the function robust, even if NLP steps failed.

    # Calculate sentiment breakdown if the column exists
    if 'sentiment_label' in topic_df.columns:
        sentiment_counts = topic_df['sentiment_label'].value_counts(normalize=True)
        # Use .get(key, 0.0) as a safe way to access counts that might not exist
        metrics['positive_ratio'] = sentiment_counts.get('POSITIVE', 0.0) * 100
        metrics['negative_ratio'] = sentiment_counts.get('NEGATIVE', 0.0) * 100
        metrics['neutral_ratio'] = sentiment_counts.get('NEUTRAL', 0.0) * 100

    # Calculate average toxicity if the column exists
    if 'toxicity_score' in topic_df.columns:
        # .mean() will safely ignore NaN values if any exist
        metrics['avg_toxicity'] = topic_df['toxicity_score'].mean()

    return metrics

def get_suggested_topics(df_display: pd.DataFrame, top_n=10):
    """
    Scans NER results to suggest potential brand/product topics to track.
    Prioritizes 'ORG' and 'MISC' entities.
    """
    if 'entities' not in df_display.columns:
        return []

    suggestions = []
    entity_types = ['ORG', 'MISC']

    for entity_list in df_display['entities'].dropna():
        if isinstance(entity_list, list):
            for entity in entity_list:
                if isinstance(entity, dict) and entity.get('entity_group') in entity_types:
                    word = entity.get('word', '').replace('##', '').strip()
                    # Exclude very short, likely unhelpful words
                    if len(word) > 2:
                        suggestions.append(word)

    if not suggestions:
        return []

    most_common_suggestions = [item for item, count in Counter(suggestions).most_common(top_n)]
    return most_common_suggestions