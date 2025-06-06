# charts.py
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

# --- NEW: Group Dynamics & Network Graph ---

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