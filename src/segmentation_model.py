
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import umap

class PatientSegmenter:
    def __init__(self, data_path='data/processed/full_patient_dataset.csv', output_dir='data/processed'):
        self.data_path = data_path
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.model_pipeline = None
        self.umap_reducer = None
        self.kmeans = None

    def load_data(self):
        print("Loading processed data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} patients.")

    def preprocess(self):
        print("Preprocessing features...")
        
        # Define features
        self.num_features = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'Glucose', 'TotalCholesterol', 
                             'Calories', 'Protein', 'Carbs', 'Sugar', 'Fiber', 'Fat', 'MedicationCount']
        self.cat_features = ['Gender', 'Race', 'Education', 'Diabetes', 'Smoker']
        
        # Create pipeline
        # Note: UMAP inputs need to be numeric. OneHotEncoder helps.
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.cat_features)
            ])
        
        self.X_processed = preprocessor.fit_transform(self.df)
        print(f"Processed feature matrix shape: {self.X_processed.shape}")
        
        # Save preprocessor
        joblib.dump(preprocessor, os.path.join(self.output_dir, 'preprocessor.pkl'))
        return self.X_processed

    def train_umap(self):
        print("Training UMAP...")
        # Reduce to 2 dimensions for visualization and clustering
        self.umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        self.umap_embedding = self.umap_reducer.fit_transform(self.X_processed)
        
        # Save UMAP model
        joblib.dump(self.umap_reducer, os.path.join(self.output_dir, 'umap_reducer.pkl'))
        
        # Add to dataframe
        self.df['UMAP_1'] = self.umap_embedding[:, 0]
        self.df['UMAP_2'] = self.umap_embedding[:, 1]
        print("UMAP reduction complete.")

    def train_kmeans(self, k=5):
        print(f"Training KMeans with k={k}...")
        # Cluster on the UMAP embeddings usually produces cleaner visual clusters, 
        # but clustering on high-dim data is also valid. 
        # Often UMAP+KMeans is good for discovering manifold structures.
        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.umap_embedding)
        
        # Add to dataframe
        self.df['Cluster'] = self.cluster_labels
        
        # Save KMeans model
        joblib.dump(self.kmeans, os.path.join(self.output_dir, 'kmeans_model.pkl'))
        print("KMeans clustering complete.")

    def save_results(self):
        output_path = os.path.join(self.output_dir, 'umap_clustered_dataset.csv')
        self.df.to_csv(output_path, index=False)
        print(f"Saved clustered dataset to {output_path}")
        
        # Summary
        summary = self.df.groupby('Cluster')[self.num_features].mean()
        summary_path = os.path.join(self.output_dir, 'cluster_summary.csv')
        summary.to_csv(summary_path)
        print(f"Saved cluster summary to {summary_path}")

    def plot_clusters(self):
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.df['UMAP_1'], self.df['UMAP_2'], c=self.df['Cluster'], cmap='viridis', s=5, alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Patient Segments (UMAP + KMeans)')
        plt.xlabel('UMAP_1')
        plt.ylabel('UMAP_2')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_plot.png'))
        print("Saved cluster plot.")

    def run(self):
        self.load_data()
        self.preprocess()
        self.train_umap()
        self.train_kmeans()
        self.save_results()
        self.plot_clusters()

if __name__ == "__main__":
    segmenter = PatientSegmenter()
    segmenter.run()
