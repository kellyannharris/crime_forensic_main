import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Load crime data"""
    if not Path(data_path).exists():
        print(f"Can't find data at {data_path}")
        raise FileNotFoundError(f"Data not found at {data_path}")
    
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Got {df.shape[0]} records")
    
    if 'DATE OCC' in df.columns:
        df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
        df = df.sort_values('DATE OCC', ascending=False).head(10000)
    else:
        df = df.tail(10000)
    
    print(f"Using {df.shape[0]} recent records")
    return df

def build_criminal_network(df):
    """Build network from crime data"""
    df = df.dropna(subset=['LOCATION', 'Mocodes'])
    edges = []
    
    for idx, row in df.iterrows():
        location = row['LOCATION']
        mocodes = [m.strip() for m in str(row['Mocodes']).split(',') if m.strip()]
        for m in mocodes:
            edges.append((location, m))
    
    G = nx.Graph()
    G.add_edges_from(edges)
    print(f"Built network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def analyze_network(G):
    """Analyze the criminal network"""
    centrality = nx.degree_centrality(G)
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:100]
    
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        print(f"Found {len(communities)} communities")
    except ImportError:
        communities = []
        print("Community detection not available")
    
    return {
        'top_nodes': top_nodes,
        'communities': communities,
        'stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G)
        }
    }

def visualize_network(G, top_nodes, output_file='data/analysis/criminal_network.png'):
    """Create network visualization"""
    top_node_names = [node[0] for node in top_nodes[:50]]
    subgraph = G.subgraph(top_node_names)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph, k=1, iterations=50)
    
    nx.draw(subgraph, pos, 
            node_color='lightblue',
            node_size=500,
            font_size=8,
            font_weight='bold',
            with_labels=True)
    
    plt.title('Criminal Network Analysis - Top Nodes')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Network visualization saved to {output_file}")

def run_network_analysis(data_path="data/structured/crime_data.csv"):
    """Run complete network analysis"""
    print("=== Criminal Network Analysis ===")
    
    # Load data
    df = load_data(data_path)
    
    # Build network
    G = build_criminal_network(df)
    
    if G.number_of_nodes() == 0:
        print("No network could be built from the data")
        return None
    
    # Analyze network
    results = analyze_network(G)
    
    # Visualize
    visualize_network(G, results['top_nodes'])
    
    # Print results
    print(f"\nNetwork Statistics:")
    print(f"- Nodes: {results['stats']['nodes']}")
    print(f"- Edges: {results['stats']['edges']}")
    print(f"- Density: {results['stats']['density']:.4f}")
    
    print(f"\nTop 10 Central Nodes:")
    for i, (node, centrality) in enumerate(results['top_nodes'][:10]):
        print(f"{i+1}. {node} (centrality: {centrality:.4f})")
    
    return results

if __name__ == "__main__":
    results = run_network_analysis() 