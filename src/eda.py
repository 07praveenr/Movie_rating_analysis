# src/eda.py
# PURPOSE: Understand patterns and answer business questions
# EDA = "exploring" the data before drawing conclusions

import pandas as pd
import numpy as np

def run_eda(df: pd.DataFrame):
    """
    Exploratory Data Analysis:
    - Summary statistics
    - Top-rated movies
    - Genre distribution
    - Year-wise trends
    - Correlation analysis
    """
    print("\n" + "=" * 50)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    # ── 1. Summary Statistics ────────────────────────────
    print("\n📊 SUMMARY STATISTICS:")
    print(df[['rating', 'votes', 'duration', 'year']].describe().round(2))
    # This tells you: mean, std, min, max, quartiles
    # Look for: is the average rating around 6-7? Are there outliers?

    # ── 2. Rating Distribution ───────────────────────────
    print("\n⭐ RATING DISTRIBUTION:")
    # Count movies in each rating bucket
    bins = [0, 4, 6, 7, 8, 10]
    labels = ['Very Low (<4)', 'Low (4-6)', 'Average (6-7)', 'Good (7-8)', 'Excellent (8-10)']
    df['rating_bucket'] = pd.cut(df['rating'], bins=bins, labels=labels)
    print(df['rating_bucket'].value_counts())

    # ── 3. Top-Rated Movies ──────────────────────────────
    print("\n🏆 TOP 10 RATED MOVIES (min 1000 votes for reliability):")
    top_rated = (
        df[df['votes'] >= 1000]          # filter low-vote movies
        .sort_values('rating', ascending=False)
        .head(10)[['title', 'year', 'rating', 'votes', 'primary_genre']]
    )
    print(top_rated.to_string(index=False))
    # WHY min 1000 votes? A movie with 1 vote rated 10 isn't meaningful

    # ── 4. Most Popular Genres ───────────────────────────
    print("\n🎬 TOP 10 GENRES BY MOVIE COUNT:")
    genre_counts = df['primary_genre'].value_counts().head(10)
    print(genre_counts)

    print("\n🎬 TOP 10 GENRES BY AVERAGE RATING:")
    genre_rating = (
        df.groupby('primary_genre')['rating']
        .agg(['mean', 'count'])
        .query('count >= 20')        # only genres with 20+ movies
        .sort_values('mean', ascending=False)
        .head(10)
        .round(2)
    )
    print(genre_rating)

    # ── 5. Year-wise Trends ──────────────────────────────
    print("\n📅 MOVIES PER YEAR (last 20 years):")
    yearly = df[df['year'] >= 2000].groupby('year').agg(
        movie_count=('title', 'count'),
        avg_rating=('rating', 'mean')
    ).round(2)
    print(yearly)

    # ── 6. Correlation Analysis ──────────────────────────
    print("\n🔗 CORRELATION MATRIX (numeric columns):")
    corr = df[['rating', 'votes', 'duration', 'year']].corr().round(3)
    print(corr)
    print("\n💡 Key insight: Does more votes = higher rating? Check rating-votes correlation.")
    print(f"   Votes ↔ Rating correlation: {corr.loc['votes','rating']:.3f}")
    # Positive = more votes tend to come with higher ratings (popular = good?)
    # Near 0 = no relationship

    return df  # return df with added rating_bucket column


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned/movies_cleaned.csv")
    df = run_eda(df)