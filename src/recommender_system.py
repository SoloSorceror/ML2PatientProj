import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns


class HealthRecommender:
    def __init__(self, data_path, preprocessor_path, umap_path, kmeans_path):
        """
        Initialize the Health Recommender System.
        """
        # Load Data and Models
        self.df = pd.read_csv(data_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.umap_reducer = joblib.load(umap_path)
        self.kmeans = joblib.load(kmeans_path)

        # Define Target State (Cluster 2 - Average/Healthy Young Adults)
        # We calculate the mean vector of Cluster 2 in the ORIGINAL feature space
        self.target_cluster = 2
        self.target_stats = self.df[self.df['Cluster'] == self.target_cluster].mean(numeric_only=True)
        
        # Define available actions
        self.actions = {
            'reduce_calories_10%': {'Calories': 0.90, 'Sugar': 0.90, 'Fat': 0.90, 'Carbs': 0.90, 'BMI': 0.995, 'Glucose': 0.99},
            'increase_exercise': {'BMI': 0.98, 'SystolicBP': 0.98, 'DiastolicBP': 0.98, 'Glucose': 0.97},
            'improve_diet_fiber': {'Fiber': 1.10, 'Sugar': 0.90, 'TotalCholesterol': 0.99, 'Glucose': 0.991},
            'reduce_sugar_20%': {'Sugar': 0.80, 'Glucose': 0.98, 'BMI': 0.998},
            'medication_adherence': {'SystolicBP': 0.95, 'DiastolicBP': 0.95, 'Glucose': 0.95} 
        }

    def _get_distance_to_target(self, current_vector_df):
        """
        Calculates the Euclidean distance from the current patient vector to the target mean vector.
        We only compare on the NUMERICAL features used in clustering.
        """
        # Identify numerical columns that match our target stats
        # We need to ensure we are comparing apples to apples. 
        # The preprocessor expects specific columns.
        
        # Transform current vector to latent space for true "distance" or compare in raw space?
        # Comparing in raw space is more interpretable for the reward function.
        
        relevant_features = [col for col in self.target_stats.index if col in current_vector_df.columns]
        
        # Calculate Euclidean distance on normalized values would be best, 
        # but for simplicity/interpretability, let's use weighted difference on key metrics
        
        dist = 0
        diffs = {}
        
        # Key metrics to optimize
        key_metrics = ['BMI', 'Glucose', 'SystolicBP', 'TotalCholesterol', 'Calories']
        
        for feat in key_metrics:
            if feat in current_vector_df.columns:
                target_val = self.target_stats[feat]
                current_val = current_vector_df[feat].values[0]
                
                # Penalty if we are WORSE than target
                # E.g. Higher BMI is bad.
                if current_val > target_val:
                     dist += (current_val - target_val)**2
                
                # For Calories, too low is also bad, but usually we are reducing.
                # Let's simple use squared error for now.
                
        return np.sqrt(dist)

    def recommend(self, patient_id=None, patient_row=None, max_steps=5):
        """
        Generates a sequence of recommendations. 
        Accepts either patient_id (lookup) OR patient_row (direct input).
        """
        if patient_row is None:
            patient_row = self.df[self.df['SEQN'] == patient_id].copy()
        
        if patient_row.empty:
            return "Patient not found."

        current_state = patient_row.copy()
        trajectory = [current_state.copy()]
        actions_taken = []
        
        # Context-Aware Thresholds (Avoid recommending reductions if already low)
        SAFE_MINS = {
            'Calories': 1500,
            'Sugar': 25,
            'Fat': 40,
            'SystolicBP': 110,
            'BMI': 22
        }

        for step in range(max_steps):
            best_action = None
            best_improvement = 0
            best_new_state = None

            current_dist = self._get_distance_to_target(current_state)

            # Evaluate all possible actions
            for action_name, effects in self.actions.items():
                
                # --- Feasibility Check ---
                # 1. Don't reduce Calories if < 1500
                if 'reduce_calories' in action_name and current_state['Calories'].values[0] < SAFE_MINS['Calories']:
                    continue
                # 2. Don't reduce Sugar if < 25g
                if 'reduce_sugar' in action_name and current_state['Sugar'].values[0] < SAFE_MINS['Sugar']:
                    continue
                # 3. Don't focus on BP meds if BP is normal
                if 'medication' in action_name and current_state['SystolicBP'].values[0] < 120:
                    continue
                
                # 4. Don't repeat actions immediately (Basic check)
                if action_name in actions_taken:
                    continue

                temp_state = current_state.copy()
                
                # Apply effects
                for col, multiplier in effects.items():
                    if col in temp_state.columns:
                        temp_state[col] = temp_state[col] * multiplier
                
                new_dist = self._get_distance_to_target(temp_state)
                improvement = current_dist - new_dist
                
                # Greedy selection
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_action = action_name
                    best_new_state = temp_state

            # If we found an improvement, commit to it
            if best_action and best_improvement > 0.05: 
                current_state = best_new_state
                actions_taken.append(best_action)
                trajectory.append(current_state.copy())
            else:
                break 

        return {
            'patient_id': patient_id if patient_id else "API_User",
            'initial_cluster': patient_row['Cluster'].values[0] if 'Cluster' in patient_row.columns else 0,
            'actions': actions_taken,
            'final_state': current_state,
            'trajectory': trajectory
        }

    def visualize_trajectory(self, plan, output_path='e:/ML2PatientProj/data/processed/patient_journey.png'):
        """
        Visualizes the patient's journey on the UMAP plane.
        """
        # 1. Transform trajectory states to UMAP coordinates
        trajectory_points = []
        for state in plan['trajectory']:
            # We need to preprocess then UMAP transform
            # Note: This implies our preprocessor and UMAP can handle single samples efficiently
            # The preprocessor expects the original dataframe structure
            
            # Ensure categorical columns match training data requirements if OneHotEncoder is strict
            # For simplicity, we assume the 'state' df has correct structure
            
            try:
                processed = self.preprocessor.transform(state)
                # UMAP transform usually expects 2D array
                coords = self.umap_reducer.transform(processed)
                trajectory_points.append(coords[0])
            except Exception as e:
                print(f"Error transforming state for visualization: {e}")
                return

        trajectory_points = np.array(trajectory_points)

        # 2. Plot Background Clusters
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            data=self.df, x='UMAP_1', y='UMAP_2', hue='Cluster', 
            palette='viridis', alpha=0.3, s=50, legend='full'
        )

        # 3. Plot Trajectory
        # Start
        plt.scatter(trajectory_points[0,0], trajectory_points[0,1], c='red', s=200, marker='X', label='Start', edgecolors='black')
        # End
        plt.scatter(trajectory_points[-1,0], trajectory_points[-1,1], c='lime', s=200, marker='*', label='Goal', edgecolors='black')
        
        # Path
        plt.plot(trajectory_points[:,0], trajectory_points[:,1], c='white', linewidth=3, linestyle='--', alpha=0.8)
        
        # Arrows
        for i in range(len(trajectory_points)-1):
            plt.arrow(
                trajectory_points[i,0], trajectory_points[i,1],
                trajectory_points[i+1,0] - trajectory_points[i,0],
                trajectory_points[i+1,1] - trajectory_points[i,1],
                color='white', width=0.05, head_width=0.2
            )

        plt.title(f"Patient {plan['patient_id']} Health Journey", fontsize=16)
        plt.legend()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory visualization saved to {output_path}")

if __name__ == "__main__":
    # Test Run
    rec = HealthRecommender(
        data_path='e:/ML2PatientProj/data/processed/umap_clustered_dataset.csv',
        preprocessor_path='e:/ML2PatientProj/data/processed/preprocessor.pkl',
        umap_path='e:/ML2PatientProj/data/processed/umap_reducer.pkl',
        kmeans_path='e:/ML2PatientProj/data/processed/kmeans_model.pkl'
    )
    
    # Pick a random patient from Cluster 4 (At-Risk)
    # We filter for high BMI to ensure a 'bad' starting state for demo purposes
    high_risk_patients = rec.df[(rec.df['Cluster'] == 4) & (rec.df['BMI'] > 30)]
    if not high_risk_patients.empty:
        sample_patient = high_risk_patients.sample(1)['SEQN'].values[0]
    else:
         sample_patient = rec.df[rec.df['Cluster'] == 4].sample(1)['SEQN'].values[0]

    plan = rec.recommend(sample_patient)
    
    print("\n--- Recommendation Plan ---")
    print(f"Patient: {plan['patient_id']}")
    print(f"Initial Cluster: {plan['initial_cluster']}")
    print("Recommended Actions:")
    for i, action in enumerate(plan['actions']):
        print(f"{i+1}. {action}")
    
    # Show Before/After stats
    initial = plan['trajectory'][0]
    final = plan['final_state']
    
    cols_to_show = ['BMI', 'SystolicBP', 'Glucose', 'Calories']
    print("\nExpected Outcomes:")
    print(f"{'Feature':<15} | {'Before':<10} | {'After':<10}")
    print("-" * 40)
    for col in cols_to_show:
        if col in initial.columns:
            print(f"{col:<15} | {initial[col].values[0]:.1f}       | {final[col].values[0]:.1f}")
            
    # Visualize
    rec.visualize_trajectory(plan)

