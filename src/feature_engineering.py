# src/feature_engineering.py
# PURPOSE: Create new columns that help the model learn better
# "Features" = inputs to a machine learning model

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features:
    1. Rating category  (High / Medium / Low)
    2. Popularity score (composite of votes + rating)
    3. Movie age        (how old is the movie?)
    4. Encoded genre    (ML needs numbers, not text)
    """
    print("\n" + "=" * 50)
    print("STEP 5: FEATURE ENGINEERING")
    print("=" * 50)

    df = df.copy()

    # ── 1. Rating Category ───────────────────────────────
    # Convert continuous rating into 3 meaningful buckets
    def categorise_rating(r):
        if r >= 7.5:
            return 'High'
        elif r >= 5.5:
            return 'Medium'
        else:
            return 'Low'

    df['rating_category'] = df['rating'].apply(categorise_rating)
    print("✅ 'rating_category' created: High / Medium / Low")
    print(df['rating_category'].value_counts())

    # ── 2. Popularity Score ──────────────────────────────
    # Combines rating AND votes into one score
    # Formula: log(votes+1) × rating   (log smooths huge vote differences)
    df['popularity_score'] = (
        np.log1p(df['votes']) * df['rating']
    ).round(3)
    print(f"\n✅ 'popularity_score' created")
    print(f"   Range: {df['popularity_score'].min():.1f} – {df['popularity_score'].max():.1f}")

    # ── 3. Movie Age ─────────────────────────────────────
    CURRENT_YEAR = 2025
    df['movie_age'] = CURRENT_YEAR - df['year']
    print(f"✅ 'movie_age' created (years since release)")

    # ── 4. Is Blockbuster? (binary feature) ─────────────
    # 1 if votes > 10,000 (significant audience attention)
    df['is_blockbuster'] = (df['votes'] > 10000).astype(int)
    print(f"✅ 'is_blockbuster' created: {df['is_blockbuster'].sum()} blockbusters found")

    # ── 5. Encode Genre for ML ───────────────────────────
    # Label Encoding: Drama=0, Action=1, Comedy=2... etc.
    le = LabelEncoder()
    df['genre_encoded'] = le.fit_transform(df['primary_genre'].astype(str))
    print(f"✅ 'genre_encoded' created ({df['primary_genre'].nunique()} unique genres encoded)")

    # ── 6. Votes (log-transformed) ───────────────────────
    # Raw votes have extreme outliers — log scale normalises them
    df['log_votes'] = np.log1p(df['votes'])
    print("✅ 'log_votes' created (log scale of votes)")

    # ── Preview new features ─────────────────────────────
    print("\n📋 New features preview:")
    print(df[['title', 'rating', 'rating_category', 'popularity_score',
              'movie_age', 'is_blockbuster', 'genre_encoded', 'log_votes']].head(5))

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned/movies_cleaned.csv")
    df = engineer_features(df)
    # Save enriched dataset
    df.to_csv("data/cleaned/movies_features.csv", index=False)
    print("\n💾 Feature-enriched dataset saved!")