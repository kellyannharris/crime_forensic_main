import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatialCrimeMapping:
    """Crime mapping with K-means clustering"""

    def __init__(self, data_path="/Users/harrisk/Documents/forensic-application-capstone/data/structured/crime_data.csv"):
        self.data_path = Path(data_path)

    def load_data(self):
        """Load crime data"""
        if not self.data_path.exists():
            print(f"Can't find data at {self.data_path}")
            raise FileNotFoundError(f"Data not found at {self.data_path}")
        
        print(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path, low_memory=False)
        print(f"Got {df.shape[0]} rows of data")
        return df

    def prepare_spatial_data(self, df):
        """Clean up the coordinate data"""
        valid_coords = (
            df['LAT'].between(33, 35) &
            df['LON'].between(-119, -117)
        )
        clean_data = df.loc[valid_coords, ['LAT', 'LON']].dropna().values
        print(f"Got {clean_data.shape[0]} valid coordinates")
        return clean_data

    def perform_kmeans_clustering(self, spatial_data, n_clusters=5):
        """Do K-means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(spatial_data)
        print(f"Created {n_clusters} clusters")
        return kmeans

    def plot_clusters(self, spatial_data, kmeans, df, valid_idx):
        """Plot the clusters on a map"""
        import geopandas as gpd
        import contextily as ctx
        import matplotlib.gridspec as gridspec
        import matplotlib.colors as mcolors
        
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        n_clusters = centers.shape[0]
        
        gdf = gpd.GeoDataFrame({
            'LAT': spatial_data[:, 0],
            'LON': spatial_data[:, 1],
            'cluster': labels,
            'crime_type': df.iloc[valid_idx]['Crm Cd Desc'].values
        }, geometry=gpd.points_from_xy(spatial_data[:, 1], spatial_data[:, 0]), crs='EPSG:4326')
        gdf = gdf.to_crs(epsg=3857)
        
        centers_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(centers[:, 1], centers[:, 0]), crs='EPSG:4326').to_crs(epsg=3857)
        
        top_crimes = gdf.groupby('cluster')['crime_type'].agg(lambda x: x.value_counts().head(3).index.tolist())
        
        cmap = plt.get_cmap('tab10', n_clusters)
        colors = [mcolors.to_hex(cmap(i)) for i in range(n_clusters)]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        for i in range(n_clusters):
            crime_list = f"Cluster {i+1}:\n" + "\n".join([f"{j+1}. {ct}" for j, ct in enumerate(top_crimes[i])])
            gdf[gdf['cluster'] == i].plot(ax=ax, markersize=8, color=colors[i], label=crime_list)
        
        centers_gdf.plot(ax=ax, marker='X', color='black', markersize=200, label='Centers')
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        ax.set_title('LA Crime Hotspots (K-means)', fontsize=16)
        ax.set_axis_off()
        ax.legend(loc='upper left', fontsize=10, title='Crime Clusters')
        plt.tight_layout()
        plt.savefig('data/analysis/spatial_crime_mapping_map.png')
        plt.close()
        print("Saved cluster map")

if __name__ == "__main__":
    analyzer = SpatialCrimeMapping()
    df = analyzer.load_data()
    
    valid = (df['LAT'].between(33, 35) & df['LON'].between(-119, -117))
    spatial_data = df.loc[valid, ['LAT', 'LON']].dropna().values
    valid_idx = df.loc[valid, ['LAT', 'LON']].dropna().index
    
    kmeans = analyzer.perform_kmeans_clustering(spatial_data)
    analyzer.plot_clusters(spatial_data, kmeans, df, valid_idx) 