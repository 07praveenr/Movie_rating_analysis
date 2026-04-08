# src/data_cleaner.py
import pandas as pd
import numpy as np
import re
import os


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 50)
    print("STEP 2: CLEANING DATA (TMDB Dataset)")
    print("=" * 50)

    df = df.copy()
    df.columns = df.columns.str.strip()

    # ── 1. Rename TMDB columns to our standard names ─────
    rename_map = {
        'title':         'title',
        'vote_average':  'rating',
        'vote_count':    'votes',
        'runtime':       'duration',
        'release_date':  'year',       # will extract year below
        'genres':        'genre',
        'popularity':    'popularity',
        'original_language': 'language',
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=rename_map, inplace=True)
    print("✅ Columns renamed")

    # ── 2. Extract year from release_date (e.g. '2021-03-15') ──
    def extract_year(val):
        if pd.isna(val):
            return np.nan
        match = re.search(r'(\d{4})', str(val))
        return int(match.group(1)) if match else np.nan

    df['year'] = df['year'].apply(extract_year)
    df.dropna(subset=['year'], inplace=True)
    df['year'] = df['year'].astype(int)
    # Keep only reasonable years
    df = df[(df['year'] >= 1900) & (df['year'] <= 2026)]
    print(f"✅ 'year' extracted | Range: {df['year'].min()}–{df['year'].max()}")

    # ── 3. Drop duplicates ───────────────────────────────
    before = len(df)
    df.drop_duplicates(subset=['title', 'year'], inplace=True)
    print(f"✅ Removed {before - len(df)} duplicates")

    # ── 4. Clean duration ────────────────────────────────
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df['duration'].fillna(df['duration'].median(), inplace=True)
    df['duration'] = df['duration'].astype(int)
    print("✅ 'duration' cleaned")

    # ── 5. Clean votes ───────────────────────────────────
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce')
    df.dropna(subset=['votes'], inplace=True)
    df['votes'] = df['votes'].astype(int)
    print("✅ 'votes' cleaned")

    # ── 6. Clean rating ──────────────────────────────────
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[(df['rating'] >= 1) & (df['rating'] <= 10)]
    print("✅ 'rating' cleaned")

    # ── 7. Filter: only movies with enough votes ─────────
    # TMDB has many obscure entries — keep min 10 votes
    df = df[df['votes'] >= 10]
    print(f"✅ Filtered to movies with 10+ votes | Rows: {len(df)}")

    # ── 8. Handle missing values ─────────────────────────
    df.dropna(subset=['rating', 'votes', 'title'], inplace=True)
    df['genre'].fillna('Unknown', inplace=True)
    print("✅ Missing values handled")

    # ── 9. Extract primary genre ─────────────────────────
    # TMDB genres look like: "Action, Drama, Thriller"
    df['primary_genre'] = df['genre'].apply(
        lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown'
    )
    print("✅ 'primary_genre' extracted")

    print(f"\n✅ Cleaning complete! Shape: {df.shape}")
    print(f"   Year range: {df['year'].min()} – {df['year'].max()}")
    print(f"   Rating range: {df['rating'].min()} – {df['rating'].max()}")
    print(f"   Sample:\n{df[['title','year','rating','votes','primary_genre']].head(3)}")

    return df


def save_cleaned(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    from data_loader import load_data

    df_raw = load_data("data/raw/movies2.csv")
    df_clean = clean_data(df_raw)
    save_cleaned(df_clean, "data/cleaned/movies_cleaned.csv")