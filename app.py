# app.py — Modern Cinematic Dark UI Dashboard
# Replace your existing app.py with this file

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import os

# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="CineStats · Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS — Cinematic Dark Theme ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg:       #0A0A0F;
    --surface:  #111118;
    --card:     #16161F;
    --border:   #ffffff0f;
    --accent:   #E8C547;
    --accent2:  #FF6B6B;
    --accent3:  #4ECDC4;
    --text:     #E8E8F0;
    --muted:    #6B6B80;
    --font-display: 'Bebas Neue', sans-serif;
    --font-body:    'DM Sans', sans-serif;
    --font-mono:    'DM Mono', monospace;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1400px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--font-display) !important;
    color: var(--accent) !important;
    letter-spacing: 2px;
}
[data-testid="stSidebar"] label {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600 !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }
.stSlider [data-testid="stTickBar"] { display: none; }

/* ── Multiselect ── */
[data-baseweb="tag"] {
    background: var(--accent) !important;
    color: #000 !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
}
[data-baseweb="select"] > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Number Input ── */
[data-testid="stNumberInput"] input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
iframe { border-radius: 12px !important; }

/* ── Warning / Info ── */
[data-testid="stAlert"] {
    background: #1A1A2E !important;
    border: 1px solid var(--accent) !important;
    border-radius: 10px !important;
    color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# ── matplotlib dark theme ────────────────────────────────
plt.style.use('dark_background')
rcParams['figure.facecolor']  = '#16161F'
rcParams['axes.facecolor']    = '#16161F'
rcParams['axes.edgecolor']    = '#ffffff0f'
rcParams['axes.labelcolor']   = '#6B6B80'
rcParams['xtick.color']       = '#6B6B80'
rcParams['ytick.color']       = '#6B6B80'
rcParams['grid.color']        = '#ffffff08'
rcParams['grid.linestyle']    = '--'
rcParams['font.family']       = 'DejaVu Sans'
rcParams['text.color']        = '#E8E8F0'

ACCENT  = '#E8C547'
ACCENT2 = '#FF6B6B'
ACCENT3 = '#4ECDC4'
MUTED   = '#6B6B80'

# ── Load Data ────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "data/cleaned/movies_features.csv"
    if not os.path.exists(path):
        st.error("❌ Run the pipeline first: `python src/feature_engineering.py`")
        st.stop()
    return pd.read_csv(path)

df = load_data()

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 1.5rem 0;'>
        <div style='font-family:"Bebas Neue",sans-serif; font-size:2rem;
                    color:#E8C547; letter-spacing:3px; line-height:1;'>
            CINE<span style='color:#FF6B6B;'>STATS</span>
        </div>
        <div style='font-size:0.68rem; color:#6B6B80; letter-spacing:2px;
                    text-transform:uppercase; margin-top:4px;'>
            Movie Intelligence Platform
        </div>
    </div>
    <hr style='border-color:#ffffff0f; margin-bottom:1.5rem;'/>
    """, unsafe_allow_html=True)

    year_min = int(df['year'].min())
    year_max = max(int(df['year'].max()), 2026)
    year_range = st.slider("📅 Year Range", year_min, year_max, (2000, year_max))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    all_genres = sorted(df['primary_genre'].dropna().unique())
    default_genres = [g for g in ["Drama","Action","Comedy","Thriller","Crime"] if g in all_genres]
    genres = st.multiselect("🎭 Genres", options=all_genres, default=default_genres[:5])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    min_votes = st.number_input("🗳️ Min Votes", 0, 500000, 500, step=100)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    rating_filter = st.slider("⭐ Min Rating", 0.0, 10.0, 0.0, step=0.1)

    st.markdown("""
    <hr style='border-color:#ffffff0f; margin:1.5rem 0 1rem 0;'/>
    <div style='font-size:0.65rem; color:#6B6B80; letter-spacing:1px;'>
        TMDB DATASET · POWERED BY PYTHON
    </div>
    """, unsafe_allow_html=True)

# ── Filter ───────────────────────────────────────────────
mask = df['year'].between(*year_range) & (df['votes'] >= min_votes) & (df['rating'] >= rating_filter)
if genres:
    mask &= df['primary_genre'].isin(genres)
filtered = df[mask]

# ── HEADER ───────────────────────────────────────────────
st.markdown(f"""
<div style='margin-bottom: 2rem;'>
    <div style='font-family:"Bebas Neue",sans-serif; font-size:3.2rem;
                letter-spacing:4px; line-height:1; color:#E8E8F0;'>
        MOVIE RATING
        <span style='color:#E8C547;'>INTELLIGENCE</span>
    </div>
    <div style='color:#6B6B80; font-size:0.8rem; letter-spacing:2px;
                text-transform:uppercase; margin-top:6px;'>
        {len(filtered):,} films · {year_range[0]}–{year_range[1]} · Real-time analysis
    </div>
</div>
""", unsafe_allow_html=True)

if filtered.empty:
    st.markdown("""
    <div style='background:#16161F; border:1px solid #E8C54733; border-radius:12px;
                padding:2rem; text-align:center; color:#E8C547;'>
        ⚠️ No movies match your filters. Try widening the year range or genre selection.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── KPI CARDS ────────────────────────────────────────────
def kpi_card(label, value, sub, color):
    return f"""
    <div style='background:#16161F; border:1px solid #ffffff0f; border-radius:14px;
                padding:1.4rem 1.6rem; border-top:3px solid {color};
                transition:transform .2s;'>
        <div style='font-size:0.65rem; color:#6B6B80; letter-spacing:2px;
                    text-transform:uppercase; margin-bottom:6px;'>{label}</div>
        <div style='font-family:"Bebas Neue",sans-serif; font-size:2.4rem;
                    color:{color}; line-height:1; letter-spacing:1px;'>{value}</div>
        <div style='font-size:0.72rem; color:#6B6B80; margin-top:4px;'>{sub}</div>
    </div>"""

top_genre = filtered['primary_genre'].mode()[0] if not filtered.empty else "—"
high_pct   = round((filtered['rating_category'] == 'High').mean() * 100, 1) if 'rating_category' in filtered.columns else 0
blockbuster_count = filtered['is_blockbuster'].sum() if 'is_blockbuster' in filtered.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, "Total Films",    f"{len(filtered):,}",                     "in selection",                 ACCENT),
    (c2, "Avg Rating",     f"{filtered['rating'].mean():.2f}",        "out of 10.0",                  ACCENT3),
    (c3, "Avg Votes",      f"{filtered['votes'].mean():,.0f}",        "audience reach",               ACCENT2),
    (c4, "Top Genre",      top_genre,                                 f"{high_pct}% rated High",      "#A78BFA"),
    (c5, "Blockbusters",   f"{blockbuster_count:,}",                  "10k+ votes",                   "#34D399"),
]
for col, label, val, sub, color in cards:
    with col:
        st.markdown(kpi_card(label, val, sub, color), unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ── ROW 1: Rating Distribution + Genre Ratings ───────────
col1, col2 = st.columns([1.1, 0.9], gap="medium")

with col1:
    st.markdown("<div style='font-family:\"Bebas Neue\",sans-serif; font-size:1.3rem; letter-spacing:2px; color:#E8E8F0; margin-bottom:0.8rem;'>RATING DISTRIBUTION</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    n, bins, patches = ax.hist(filtered['rating'], bins=25, edgecolor='none')
    # Color bars by value
    norm = plt.Normalize(bins[:-1].min(), bins[:-1].max())
    colors = plt.cm.YlOrRd(norm(bins[:-1]))
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    mean_r = filtered['rating'].mean()
    ax.axvline(mean_r, color=ACCENT, linewidth=1.5, linestyle='--', alpha=0.8)
    ax.text(mean_r + 0.05, ax.get_ylim()[1] * 0.92,
            f'μ = {mean_r:.2f}', color=ACCENT, fontsize=8, fontweight='bold')
    ax.set_xlabel("IMDb Rating", fontsize=9, color=MUTED)
    ax.set_ylabel("Count", fontsize=9, color=MUTED)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col2:
    st.markdown("<div style='font-family:\"Bebas Neue\",sans-serif; font-size:1.3rem; letter-spacing:2px; color:#E8E8F0; margin-bottom:0.8rem;'>GENRE RATINGS</div>", unsafe_allow_html=True)
    genre_avg = (
        filtered.groupby('primary_genre')['rating']
        .agg(['mean','count'])
        .query('count >= 3')
        .sort_values('mean', ascending=True)
        .tail(10)
    )
    if not genre_avg.empty:
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        bars = ax2.barh(genre_avg.index, genre_avg['mean'],
                        color=[ACCENT3 if v >= genre_avg['mean'].median() else MUTED
                               for v in genre_avg['mean']],
                        alpha=0.85, height=0.6)
        for bar, val in zip(bars, genre_avg['mean']):
            ax2.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height()/2,
                     f'{val:.2f}', va='center', fontsize=8, color=ACCENT3, fontweight='bold')
        ax2.set_xlim(0, 10.5)
        ax2.set_xlabel("Avg Rating", fontsize=9, color=MUTED)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params(left=False)
        ax2.grid(axis='x', alpha=0.2)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── ROW 2: Movies Per Year ───────────────────────────────
st.markdown("<div style='font-family:\"Bebas Neue\",sans-serif; font-size:1.3rem; letter-spacing:2px; color:#E8E8F0; margin-bottom:0.8rem;'>YEARLY PRODUCTION TREND</div>", unsafe_allow_html=True)

yearly = filtered.groupby('year').agg(
    count=('title','count'),
    avg_rating=('rating','mean')
).reset_index()

if len(yearly) > 1:
    fig3, ax3 = plt.subplots(figsize=(14, 3.2))
    ax3b = ax3.twinx()

    ax3.fill_between(yearly['year'], yearly['count'],
                     alpha=0.15, color=ACCENT)
    ax3.plot(yearly['year'], yearly['count'],
             color=ACCENT, linewidth=2, zorder=3)
    ax3.scatter(yearly['year'], yearly['count'],
                color=ACCENT, s=20, zorder=4)

    ax3b.plot(yearly['year'], yearly['avg_rating'],
              color=ACCENT2, linewidth=1.5, linestyle='--', alpha=0.7)

    ax3.set_ylabel("Movies Released", fontsize=9, color=ACCENT)
    ax3b.set_ylabel("Avg Rating", fontsize=9, color=ACCENT2)
    ax3b.set_ylim(0, 10)
    ax3.set_xlabel("Year", fontsize=9, color=MUTED)

    for spine in ['top']:
        ax3.spines[spine].set_visible(False)
    ax3.spines['right'].set_color(ACCENT2 + '44')
    ax3.grid(axis='y', alpha=0.15)

    legend = [
        mpatches.Patch(color=ACCENT,  label='Movie Count'),
        mpatches.Patch(color=ACCENT2, label='Avg Rating'),
    ]
    ax3.legend(handles=legend, loc='upper left',
               framealpha=0, fontsize=8, labelcolor='white')

    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()
else:
    st.info("Select a wider year range to see the trend.")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── ROW 3: Votes vs Rating + Rating Category Donut ───────
col3, col4 = st.columns([1.2, 0.8], gap="medium")

with col3:
    st.markdown("<div style='font-family:\"Bebas Neue\",sans-serif; font-size:1.3rem; letter-spacing:2px; color:#E8E8F0; margin-bottom:0.8rem;'>VOTES vs RATING</div>", unsafe_allow_html=True)
    sample = filtered[filtered['votes'] > 0].sample(min(2000, len(filtered)), random_state=42)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sc = ax4.scatter(
        np.log10(sample['votes'] + 1),
        sample['rating'],
        c=sample['rating'], cmap='YlOrRd',
        alpha=0.5, s=12, vmin=1, vmax=10, zorder=3
    )
    # Trend line
    x = np.log10(sample['votes'] + 1)
    z = np.polyfit(x, sample['rating'], 1)
    p = np.poly1d(z)
    xs = np.linspace(x.min(), x.max(), 100)
    ax4.plot(xs, p(xs), color=ACCENT3, linewidth=2, linestyle='--', label='Trend', zorder=4)
    ax4.set_xlabel("Log₁₀(Votes)", fontsize=9, color=MUTED)
    ax4.set_ylabel("Rating", fontsize=9, color=MUTED)
    ax4.legend(framealpha=0, fontsize=8, labelcolor='white')
    plt.colorbar(sc, ax=ax4, label='Rating', shrink=0.8)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(alpha=0.15)
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close()

with col4:
    st.markdown("<div style='font-family:\"Bebas Neue\",sans-serif; font-size:1.3rem; letter-spacing:2px; color:#E8E8F0; margin-bottom:0.8rem;'>RATING CATEGORIES</div>", unsafe_allow_html=True)
    if 'rating_category' in filtered.columns:
        cat_counts = filtered['rating_category'].value_counts()
        fig5, ax5 = plt.subplots(figsize=(5, 4))
        cat_colors = {'High': ACCENT3, 'Medium': ACCENT, 'Low': ACCENT2}
        colors_pie = [cat_colors.get(c, MUTED) for c in cat_counts.index]
        wedges, texts, autotexts = ax5.pie(
            cat_counts.values,
            labels=cat_counts.index,
            colors=colors_pie,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor='#0A0A0F', linewidth=2)
        )
        for t in texts:
            t.set_color('#E8E8F0')
            t.set_fontsize(10)
            t.set_fontweight('bold')
        for at in autotexts:
            at.set_color('#0A0A0F')
            at.set_fontsize(8)
            at.set_fontweight('bold')
        ax5.text(0, 0, f"{len(filtered):,}\nfilms",
                 ha='center', va='center',
                 fontsize=10, color='#E8E8F0', fontweight='bold', linespacing=1.4)
        fig5.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close()

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── TOP MOVIES TABLE ─────────────────────────────────────
st.markdown("<div style='font-family:\"Bebas Neue\",sans-serif; font-size:1.3rem; letter-spacing:2px; color:#E8E8F0; margin-bottom:0.8rem;'>🏆 TOP RATED FILMS</div>", unsafe_allow_html=True)

display_cols = ['title', 'year', 'primary_genre', 'rating', 'votes']
if 'rating_category' in filtered.columns:
    display_cols.append('rating_category')
if 'popularity_score' in filtered.columns:
    display_cols.append('popularity_score')

top_movies = (
    filtered[filtered['votes'] >= max(min_votes, 100)]
    [display_cols]
    .sort_values('rating', ascending=False)
    .head(50)
    .reset_index(drop=True)
)
top_movies.index += 1

st.dataframe(
    top_movies,
    use_container_width=True,
    height=380,
    column_config={
        "title":          st.column_config.TextColumn("🎬 Title", width="large"),
        "year":           st.column_config.NumberColumn("📅 Year", format="%d"),
        "primary_genre":  st.column_config.TextColumn("🎭 Genre"),
        "rating":         st.column_config.ProgressColumn("⭐ Rating", min_value=0, max_value=10, format="%.1f"),
        "votes":          st.column_config.NumberColumn("🗳️ Votes", format="%d"),
        "rating_category":st.column_config.TextColumn("📊 Category"),
        "popularity_score":st.column_config.NumberColumn("🔥 Score", format="%.1f"),
    }
)

# ── FOOTER ───────────────────────────────────────────────
st.markdown("""
<div style='margin-top:3rem; padding:1.5rem 0; border-top:1px solid #ffffff0f;
            display:flex; justify-content:space-between; align-items:center;'>
    <div style='font-family:"Bebas Neue",sans-serif; font-size:1rem;
                letter-spacing:3px; color:#6B6B80;'>
        CINESTATS · MOVIE INTELLIGENCE
    </div>
    <div style='font-size:0.7rem; color:#6B6B80; letter-spacing:1px;'>
        BUILT WITH PYTHON · PANDAS · MATPLOTLIB · STREAMLIT
    </div>
</div>
""", unsafe_allow_html=True)