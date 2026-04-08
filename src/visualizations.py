# src/visualizations.py
# PURPOSE: Turn numbers into visual stories
# Good visualizations reveal insights instantly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

# ── Global style settings ────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
OUTPUT_DIR = "outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(filename: str):
    """Helper to save each plot."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    print(f"💾 Saved: {path}")
    plt.close()


def plot_rating_histogram(df: pd.DataFrame):
    """
    PLOT 1: Distribution of movie ratings
    Tells us: Are most movies rated high or low? Is the distribution normal?
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(df['rating'], bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(df['rating'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean: {df['rating'].mean():.2f}")
    ax.axvline(df['rating'].median(), color='orange', linestyle='--', linewidth=1.5, label=f"Median: {df['rating'].median():.2f}")
    
    ax.set_title("Distribution of Movie Ratings", fontsize=14, fontweight='bold')
    ax.set_xlabel("IMDb Rating")
    ax.set_ylabel("Number of Movies")
    ax.legend()
    
    save_plot("01_rating_histogram.png")


def plot_top_genres(df: pd.DataFrame):
    """
    PLOT 2: Bar chart of top genres by movie count
    Tells us: What genres are most commonly produced?
    """
    genre_counts = df['primary_genre'].value_counts().head(12)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(genre_counts.index[::-1], genre_counts.values[::-1], 
                   color=sns.color_palette("viridis", len(genre_counts)))
    
    # Add value labels on bars
    for bar, val in zip(bars, genre_counts.values[::-1]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', fontsize=9)
    
    ax.set_title("Top 12 Movie Genres by Count", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Movies")
    ax.set_ylabel("Genre")
    
    save_plot("02_top_genres.png")


def plot_avg_rating_by_genre(df: pd.DataFrame):
    """
    PLOT 3: Average rating per genre (for genres with 20+ movies)
    Tells us: Which genres are consistently rated highest?
    """
    genre_rating = (
        df.groupby('primary_genre')['rating']
        .agg(['mean', 'count'])
        .query('count >= 20')
        .sort_values('mean', ascending=False)
        .head(12)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("RdYlGn", len(genre_rating))
    bars = ax.barh(genre_rating.index[::-1], genre_rating['mean'].values[::-1], color=palette[::-1])
    
    for bar, val in zip(bars, genre_rating['mean'].values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9)
    
    ax.set_xlim(0, 10)
    ax.axvline(df['rating'].mean(), color='gray', linestyle='--', alpha=0.5, label='Overall mean')
    ax.set_title("Average Rating by Genre (min 20 movies)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Average Rating")
    ax.legend()
    
    save_plot("03_avg_rating_by_genre.png")


def plot_movies_per_year(df: pd.DataFrame):
    """
    PLOT 4: Line chart — movies released per year
    Tells us: Is the industry growing? Any dip during certain years?
    """
    yearly = df[df['year'] >= 1990].groupby('year').agg(
        count=('title', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()  # second y-axis for average rating

    ax1.fill_between(yearly['year'], yearly['count'], alpha=0.3, color='steelblue')
    ax1.plot(yearly['year'], yearly['count'], color='steelblue', linewidth=2, label='Movie Count')
    ax2.plot(yearly['year'], yearly['avg_rating'], color='tomato', linewidth=2, linestyle='--', label='Avg Rating')

    ax1.set_title("Movies Released Per Year & Average Rating Trend", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Movies", color='steelblue')
    ax2.set_ylabel("Average Rating", color='tomato')
    ax2.set_ylim(0, 10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    save_plot("04_movies_per_year.png")


def plot_votes_vs_rating(df: pd.DataFrame):
    """
    PLOT 5: Scatter plot — votes vs rating
    Tells us: Do popular movies (high votes) tend to be better rated?
    """
    # Sample to avoid overplotting if dataset is large
    sample = df[df['votes'] > 0].sample(min(3000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        np.log10(sample['votes'] + 1),   # log scale for votes (huge range)
        sample['rating'],
        alpha=0.4, s=15,
        c=sample['rating'], cmap='RdYlGn', vmin=1, vmax=10
    )
    plt.colorbar(scatter, ax=ax, label='Rating')

    ax.set_title("Votes vs Rating (log scale for votes)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Log₁₀(Votes)")
    ax.set_ylabel("IMDb Rating")

    # Add trend line
    z = np.polyfit(np.log10(sample['votes'] + 1), sample['rating'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(np.log10(sample['votes'].min()+1), np.log10(sample['votes'].max()+1), 100)
    ax.plot(x_line, p(x_line), 'b--', linewidth=1.5, label='Trend line')
    ax.legend()
    
    save_plot("05_votes_vs_rating.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    PLOT 6: Heatmap of numeric correlations
    Tells us: Which features are related to each other?
    """
    numeric_cols = ['rating', 'votes', 'duration', 'year']
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".3f",
        cmap='coolwarm', center=0,
        square=True, linewidths=0.5,
        ax=ax
    )
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
    
    save_plot("06_correlation_heatmap.png")


def run_all_plots(df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("=" * 50)
    plot_rating_histogram(df)
    plot_top_genres(df)
    plot_avg_rating_by_genre(df)
    plot_movies_per_year(df)
    plot_votes_vs_rating(df)
    plot_correlation_heatmap(df)
    print("\n✅ All 6 plots saved to outputs/plots/")


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned/movies_cleaned.csv")
    run_all_plots(df)