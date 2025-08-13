"""
Criminal Network Analysis
Kelly-Ann Harris
"""

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class CriminalNetworkAnalyzer:
    """Network analysis for crime data"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path or "data/structured/crime_data.csv"
        self.network = nx.Graph()
        self.crime_data = None
        
        self._load_data()
    
    def _load_data(self):
        """Load crime data"""
        try:
            if Path(self.data_path).exists():
                self.crime_data = pd.read_csv(self.data_path)
                print(f"Loaded {len(self.crime_data)} records")
            else:
                print(f"Data file not found: {self.data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def build_network(self, min_connections=2):
        """Build network from crime data"""
        try:
            if self.crime_data is None:
                return self._create_sample_network(min_connections)
            
            self.network.clear()
            
            area_crimes = self.crime_data.groupby('AREA')
            connections_found = 0
            
            for area1, crimes1 in area_crimes:
                for area2, crimes2 in area_crimes:
                    if area1 >= area2:
                        continue
                    
                    crime_types1 = set(crimes1.get('Crm Cd', []))
                    crime_types2 = set(crimes2.get('Crm Cd', []))
                    
                    shared_crimes = crime_types1.intersection(crime_types2)
                    if len(shared_crimes) >= min_connections:
                        weight = len(shared_crimes)
                        self.network.add_edge(f"Area_{area1}", f"Area_{area2}", weight=weight)
                        connections_found += 1
            
            stats = {
                "nodes": len(self.network.nodes()),
                "edges": len(self.network.edges()),
                "connections_found": connections_found,
                "network_density": nx.density(self.network) if len(self.network.nodes()) > 1 else 0,
                "average_connections": round(connections_found / max(len(self.network.nodes()), 1), 2)
            }
            
            print(f"Network built: {stats['nodes']} areas, {stats['edges']} connections")
            return stats
            
        except Exception as e:
            print(f"Error building network: {e}")
            return self._create_sample_network(min_connections)
    
    def _create_sample_network(self, min_connections=2):
        """Create sample network"""
        self.network.clear()
        
        # Create sample network structure
        sample_areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5"]
        
        # Add connections representing criminal network relationships
        connections = [
            ("Area_1", "Area_2", 3),  # 3 shared crime patterns
            ("Area_2", "Area_3", 2),  # 2 shared crime patterns  
            ("Area_3", "Area_4", 4),  # 4 shared crime patterns
            ("Area_1", "Area_5", 2),  # 2 shared crime patterns
        ]
        
        for area1, area2, weight in connections:
            if weight >= min_connections:
                self.network.add_edge(area1, area2, weight=weight)
        
        return {
            "nodes": len(self.network.nodes()),
            "edges": len(self.network.edges()),
            "connections_found": len(connections),
            "network_density": nx.density(self.network) if len(self.network.nodes()) > 1 else 0,
            "average_connections": 2.0,
            "note": "Sample network created for demonstration"
        }
    
    def analyze_centrality_measures(self, top_n=5):
        """
        Analyze centrality measures as specified in proposal
        Calculates degree, betweenness, and eigenvector centrality for key player identification
        """
        try:
            if len(self.network.nodes()) == 0:
                return {"error": "No network built yet"}
            
            # Calculate different centrality measures as per proposal
            centrality_results = {}
            
            # Degree centrality - identifies most connected nodes
            degree_centrality = nx.degree_centrality(self.network)
            sorted_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
            centrality_results['degree_centrality'] = [
                {"area": area, "centrality_score": round(score, 3)} 
                for area, score in sorted_degree[:top_n]
            ]
            
            # Betweenness centrality - identifies bridge nodes
            if len(self.network.nodes()) > 2:
                betweenness_centrality = nx.betweenness_centrality(self.network)
                sorted_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
                centrality_results['betweenness_centrality'] = [
                    {"area": area, "centrality_score": round(score, 3)} 
                    for area, score in sorted_betweenness[:top_n]
                ]
            
            # Eigenvector centrality - identifies influential nodes
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.network, max_iter=1000)
                sorted_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)
                centrality_results['eigenvector_centrality'] = [
                    {"area": area, "centrality_score": round(score, 3)} 
                    for area, score in sorted_eigenvector[:top_n]
                ]
            except:
                centrality_results['eigenvector_centrality'] = "Not computed - network structure"
            
            result = {
                "centrality_analysis": centrality_results,
                "total_areas": len(self.network.nodes()),
                "method": "Centrality Measures Analysis"
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error in centrality analysis: {e}"}
    
    def detect_communities(self):
        """
        Community detection algorithms as specified in proposal
        Identifies groups of closely connected nodes in the criminal network
        """
        try:
            if len(self.network.nodes()) == 0:
                return {"error": "No network built yet"}
            
            # Apply community detection algorithms
            communities_results = {}
            
            # Greedy modularity community detection
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(self.network))
                
                communities_results['greedy_modularity'] = {
                    "number_of_communities": len(communities),
                    "communities": [
                        {
                            "community_id": i,
                            "areas": list(community),
                            "size": len(community)
                        }
                        for i, community in enumerate(communities)
                    ],
                    "modularity": nx.community.modularity(self.network, communities)
                }
            except:
                # Fallback to connected components
                communities = list(nx.connected_components(self.network))
                communities_results['connected_components'] = {
                    "number_of_communities": len(communities),
                    "communities": [
                        {
                            "community_id": i,
                            "areas": list(community),
                            "size": len(community)
                        }
                        for i, community in enumerate(communities)
                    ]
                }
            
            # Label propagation community detection (alternative algorithm)
            try:
                from networkx.algorithms.community import label_propagation_communities
                lp_communities = list(label_propagation_communities(self.network))
                
                communities_results['label_propagation'] = {
                    "number_of_communities": len(lp_communities),
                    "communities": [
                        {
                            "community_id": i,
                            "areas": list(community),
                            "size": len(community)
                        }
                        for i, community in enumerate(lp_communities)
                    ]
                }
            except:
                pass
            
            result = {
                "community_detection": communities_results,
                "method": "Community Detection Algorithms"
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Error in community detection: {e}"}
    
    def get_network_summary(self):
        """Get comprehensive network analysis summary using graph analytics"""
        try:
            if len(self.network.nodes()) == 0:
                return {"error": "No network built yet"}
            
            # Basic network statistics from graph analytics
            num_nodes = len(self.network.nodes())
            num_edges = len(self.network.edges())
            
            # Calculate network density
            density = nx.density(self.network) if num_nodes > 1 else 0
            
            # Calculate clustering coefficient
            clustering = nx.average_clustering(self.network) if num_nodes > 0 else 0
            
            # Analyze edge weights (connection strengths)
            if num_edges > 0:
                edge_weights = [data['weight'] for _, _, data in self.network.edges(data=True)]
                avg_weight = sum(edge_weights) / len(edge_weights)
                max_weight = max(edge_weights)
                min_weight = min(edge_weights)
            else:
                avg_weight = 0
                max_weight = 0
                min_weight = 0
            
            # Connected components analysis
            connected_components = list(nx.connected_components(self.network))
            largest_component_size = max([len(comp) for comp in connected_components]) if connected_components else 0
            
            return {
                "network_statistics": {
                    "total_areas": num_nodes,
                    "total_connections": num_edges,
                    "network_density": round(density, 3),
                    "clustering_coefficient": round(clustering, 3),
                    "connected_components": len(connected_components),
                    "largest_component_size": largest_component_size
                },
                "connection_analysis": {
                    "average_shared_crimes": round(avg_weight, 2),
                    "max_shared_crimes": max_weight,
                    "min_shared_crimes": min_weight
                },
                "analysis_ready": True,
                "method": "Graph Analytics"
            }
            
        except Exception as e:
            return {"error": f"Error in network summary: {e}"}
    
    def find_key_players(self, top_n=5):
        """
        Identify key players in criminal network using centrality measures
        Combines multiple centrality metrics to rank most important nodes
        """
        try:
            if len(self.network.nodes()) == 0:
                return {"error": "No network built yet"}
            
            # Calculate multiple centrality measures
            degree_centrality = nx.degree_centrality(self.network)
            
            # Calculate other centrality measures if network is large enough
            if len(self.network.nodes()) > 2:
                try:
                    betweenness_centrality = nx.betweenness_centrality(self.network)
                    eigenvector_centrality = nx.eigenvector_centrality(self.network, max_iter=1000)
                except:
                    betweenness_centrality = {node: 0 for node in self.network.nodes()}
                    eigenvector_centrality = {node: 0 for node in self.network.nodes()}
            else:
                betweenness_centrality = {node: 0 for node in self.network.nodes()}
                eigenvector_centrality = {node: 0 for node in self.network.nodes()}
            
            # Combine centrality measures for composite ranking
            key_players = []
            for node in self.network.nodes():
                # Weighted combination of centrality measures
                composite_score = (
                    0.4 * degree_centrality.get(node, 0) +
                    0.3 * betweenness_centrality.get(node, 0) +
                    0.3 * eigenvector_centrality.get(node, 0)
                )
                
                key_players.append({
                    "area": node,
                    "composite_score": round(composite_score, 3),
                    "degree_centrality": round(degree_centrality.get(node, 0), 3),
                    "betweenness_centrality": round(betweenness_centrality.get(node, 0), 3),
                    "eigenvector_centrality": round(eigenvector_centrality.get(node, 0), 3),
                    "connections": list(self.network.neighbors(node))
                })
            
            # Sort by composite score
            key_players.sort(key=lambda x: x['composite_score'], reverse=True)
            
            return {
                "key_players": key_players[:top_n],
                "total_analyzed": len(key_players),
                "method": "Centrality Measures Analysis"
            }
            
        except Exception as e:
            return {"error": f"Error finding key players: {e}"}
    
    def analyze_crime_patterns(self, area):
        """
        Analyze crime patterns for specific area using network context
        """
        try:
            if self.crime_data is None:
                return {"error": "No crime data available"}
            
            # Filter data for this area
            area_data = self.crime_data[self.crime_data['AREA'] == area]
            
            if len(area_data) == 0:
                return {"error": f"No data found for area {area}"}
            
            # Count crime types
            crime_counts = area_data['Crm Cd'].value_counts().head(5)
            
            # Time patterns
            if 'TIME OCC' in area_data.columns:
                # Extract hour from time
                area_data['hour'] = area_data['TIME OCC'].astype(str).str.zfill(4).str[:2].astype(int)
                time_pattern = area_data['hour'].value_counts().sort_index().head(5)
            else:
                time_pattern = {}
            
            # Network context - find connected areas
            area_name = f"Area_{area}"
            connected_areas = []
            if area_name in self.network.nodes():
                for neighbor in self.network.neighbors(area_name):
                    weight = self.network[area_name][neighbor]['weight']
                    connected_areas.append({
                        "connected_area": neighbor,
                        "shared_patterns": weight
                    })
            
            return {
                "area": area,
                "total_crimes": len(area_data),
                "top_crime_types": crime_counts.to_dict(),
                "peak_hours": time_pattern.to_dict() if hasattr(time_pattern, 'to_dict') else {},
                "network_connections": connected_areas,
                "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "method": "Graph Analytics with Crime Pattern Analysis"
            }
            
        except Exception as e:
            return {"error": f"Error analyzing patterns: {e}"}
    
    def get_area_connections(self, area):
        """Get network connections for a specific area"""
        try:
            area_name = f"Area_{area}"
            
            if area_name not in self.network.nodes():
                return {"error": f"Area {area} not found in network"}
            
            # Get all connections for this area
            connections = []
            for neighbor in self.network.neighbors(area_name):
                weight = self.network[area_name][neighbor]['weight']
                connections.append({
                    "connected_area": neighbor,
                    "shared_crime_patterns": weight,
                    "connection_strength": "strong" if weight > 3 else "medium" if weight > 1 else "weak"
                })
            
            return {
                "area": area,
                "total_connections": len(connections),
                "connections": connections,
                "method": "Graph Analytics"
            }
            
        except Exception as e:
            return {"error": f"Error getting connections: {e}"} 