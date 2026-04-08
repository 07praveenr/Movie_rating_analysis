# src/data_loader.py
# PURPOSE: Load the raw dataset and do a first inspection
# Think of this as "opening the box and seeing what's inside"

import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV dataset into a pandas DataFrame.
    
    Args:
        filepath: Path to the CSV file
    Returns:
        Raw DataFrame
    """
    print("=" * 50)
    print("STEP 1: LOADING DATA")
    print("=" * 50)

    # Load the CSV
    df = pd.read_csv(filepath, encoding='latin-1')  
    # encoding='latin-1' handles special characters in movie names

    # --- First look ---
    print(f"\n✅ Dataset loaded successfully!")
    print(f"\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n📋 Column Names:")
    for col in df.columns:
        print(f"   • {col}")

    print("\n👀 First 5 Rows:")
    print(df.head())

    print("\n🔍 Data Types:")
    print(df.dtypes)

    print("\n❓ Missing Values (count per column):")
    print(df.isnull().sum())

    return df


# --- Run this file directly to test ---
if __name__ == "__main__":
    df = load_data("data/raw/movies2.csv")