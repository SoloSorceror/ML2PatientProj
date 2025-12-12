
import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self, raw_dir='data/raw', processed_dir='data/processed'):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def load_data(self):
        print("Loading raw data...")
        self.demo = pd.read_csv(os.path.join(self.raw_dir, 'demographic.csv'))
        self.diet = pd.read_csv(os.path.join(self.raw_dir, 'diet.csv'))
        self.exam = pd.read_csv(os.path.join(self.raw_dir, 'examination.csv'))
        self.labs = pd.read_csv(os.path.join(self.raw_dir, 'labs.csv'))
        self.meds = pd.read_csv(os.path.join(self.raw_dir, 'medications.csv'), encoding='latin1')
        self.quest = pd.read_csv(os.path.join(self.raw_dir, 'questionnaire.csv'))
        print("Data loaded successfully.")

    def process_medications(self):
        print("Processing medications...")
        # Aggregate to patient level: Count of medications
        med_counts = self.meds.groupby('SEQN').size().reset_index(name='MedicationCount')
        return med_counts

    def clean_demographics(self):
        print("Cleaning demographics...")
        # Keep: SEQN, Gender, Age, Race, Education
        # RIAGENDR: 1=Male, 2=Female
        # RIDAGEYR: Age in years
        # RIDRETH1: Race/Ethnicity (1=Mexican, 2=Other Hispanic, 3=White, 4=Black, 5=Other)
        cols = ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH1', 'DMDEDUC2']
        df = self.demo[cols].copy()
        df.rename(columns={
            'RIAGENDR': 'Gender', 
            'RIDAGEYR': 'Age', 
            'RIDRETH1': 'Race',
            'DMDEDUC2': 'Education'
        }, inplace=True)
        return df

    def clean_diet(self):
        print("Cleaning diet...")
        # Keep macronutrients from Day 1
        # DR1TKCAL (Calories), DR1TPROT (Protein), DR1TCARB (Carbs), DR1TSUGR (Sugar), DR1TFIBE (Fiber), DR1TTFAT (Fat)
        cols = ['SEQN', 'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TSUGR', 'DR1TFIBE', 'DR1TTFAT']
        df = self.diet[cols].copy()
        df.rename(columns={
            'DR1TKCAL': 'Calories', 
            'DR1TPROT': 'Protein', 
            'DR1TCARB': 'Carbs', 
            'DR1TSUGR': 'Sugar', 
            'DR1TFIBE': 'Fiber', 
            'DR1TTFAT': 'Fat'
        }, inplace=True)
        return df

    def clean_examination(self):
        print("Cleaning examination...")
        # Keep BMI and BP
        # BMXBMI (BMI)
        # BPXSY1 (Systolic BP 1st reading)
        # BPXDI1 (Diastolic BP 1st reading)
        cols = ['SEQN', 'BMXBMI', 'BPXSY1', 'BPXDI1']
        df = self.exam[cols].copy()
        df.rename(columns={
            'BMXBMI': 'BMI', 
            'BPXSY1': 'SystolicBP', 
            'BPXDI1': 'DiastolicBP'
        }, inplace=True)
        return df

    def clean_labs(self):
        print("Cleaning labs...")
        # Keep Glucose, Cholesterol
        # LBXGLT (Glucose, refrigerated serum)
        # LBXTC (Total Cholesterol)
        # LBXSCH (Cholesterol in labs.csv? Check schema report - LBXTC is in one of them)
        # From report: LBXTC is in labs.csv. LBXGLT is in labs.csv.
        
        # Note: Columns might have NaNs, handled later
        cols = ['SEQN', 'LBXGLT', 'LBXTC']
        # Filter columns that actually exist to avoid errors if I misidentified one
        available_cols = [c for c in cols if c in self.labs.columns]
        df = self.labs[available_cols].copy()
        df.rename(columns={
            'LBXGLT': 'Glucose', 
            'LBXTC': 'TotalCholesterol'
        }, inplace=True)
        return df

    def clean_questionnaire(self):
        print("Cleaning questionnaire...")
        # Keep Diabetes diagnosis, Smoking status
        # DIQ010: Doctor told you have diabetes? (1=Yes, 2=No, 3=Borderline)
        # SMQ020: Smoked at least 100 cigarettes in life? (1=Yes, 2=No)
        cols = ['SEQN', 'DIQ010', 'SMQ020']
        available_cols = [c for c in cols if c in self.quest.columns]
        df = self.quest[available_cols].copy()
        df.rename(columns={
            'DIQ010': 'Diabetes', 
            'SMQ020': 'Smoker'
        }, inplace=True)
        return df

    def run(self):
        self.load_data()
        
        df_demo = self.clean_demographics()
        df_diet = self.clean_diet()
        df_exam = self.clean_examination()
        df_labs = self.clean_labs()
        df_quest = self.clean_questionnaire()
        df_meds = self.process_medications()

        # Merge all
        print("Merging datasets...")
        main_df = df_demo.merge(df_diet, on='SEQN', how='left')
        main_df = main_df.merge(df_exam, on='SEQN', how='left')
        main_df = main_df.merge(df_labs, on='SEQN', how='left')
        main_df = main_df.merge(df_quest, on='SEQN', how='left')
        main_df = main_df.merge(df_meds, on='SEQN', how='left')

        # Fill NaNs
        # Med count NaN -> 0
        main_df['MedicationCount'] = main_df['MedicationCount'].fillna(0)
        
        # For numerical cols, fill with median
        num_cols = ['Age', 'Calories', 'Protein', 'Carbs', 'Sugar', 'Fiber', 'Fat', 
                   'BMI', 'SystolicBP', 'DiastolicBP', 'Glucose', 'TotalCholesterol']
        for c in num_cols:
            if c in main_df.columns:
                main_df[c] = main_df[c].fillna(main_df[c].median())
        
        # For categorical, fill with mode or special value
        # Education, Diabetes, Smoker
        cat_cols = ['Education', 'Diabetes', 'Smoker']
        for c in cat_cols:
             if c in main_df.columns:
                main_df[c] = main_df[c].fillna(main_df[c].mode()[0])

        print(f"Final dataset shape: {main_df.shape}")
        
        output_path = os.path.join(self.processed_dir, 'full_patient_dataset.csv')
        main_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.run()
