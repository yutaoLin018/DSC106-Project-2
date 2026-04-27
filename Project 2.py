import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def add_title_and_subtitle(fig, title, subtitle):
    fig.suptitle(title, fontsize=18, y=0.992)
    fig.text(0.5, 0.948, subtitle, ha="center", fontsize=11)
    fig.subplots_adjust(top=0.84)

# =========================
# 1. LOAD DATA
# =========================
econ = pd.read_csv("data/economy-and-growth.csv")
edu = pd.read_csv("data/education.csv")
infra = pd.read_csv("data/infrastructure.csv")
poverty = pd.read_csv("data/poverty.csv")

# =========================
# 2. KEEP COUNTRIES ONLY
# =========================
def keep_countries_only(df):
    bad_keywords = [
        "income", "IDA", "IBRD", "total", "World", "OECD", "Euro area",
        "Europe & Central Asia", "East Asia & Pacific", "Latin America",
        "Middle East", "North America", "South Asia", "Sub-Saharan Africa",
        "Low & middle income", "Middle income", "High income",
        "Arab World", "European Union", "Early-demographic dividend",
        "Late-demographic dividend", "Post-demographic dividend",
        "Pre-demographic dividend", "Small states", "Fragile and conflict affected"
    ]
    pattern = "|".join(bad_keywords)
    mask = ~df["Country Name"].str.contains(pattern, case=False, na=False)
    return df[mask].copy()

econ = keep_countries_only(econ)
edu = keep_countries_only(edu)
infra = keep_countries_only(infra)
poverty = keep_countries_only(poverty)

# =========================
# 3. SELECT COLUMNS
# =========================
gdp_col = "average_value_GDP per capita (constant 2010 US$)"
school_col = "average_value_School enrollment, secondary (% gross)"
internet_col = "average_value_Individuals using the Internet (% of population)"
poverty_col = "average_value_Poverty headcount ratio at $3.20 a day (2011 PPP) (% of population)"

econ = econ[["Country Name", "Country Code", "Year", gdp_col]].copy()
edu = edu[["Country Name", "Country Code", "Year", school_col]].copy()
infra = infra[["Country Name", "Country Code", "Year", internet_col]].copy()
poverty = poverty[["Country Name", "Country Code", "Year", poverty_col]].copy()

# =========================
# 4. CLEAN YEAR + VALUES
# =========================
for df, value_col in [
    (econ, gdp_col),
    (edu, school_col),
    (infra, internet_col),
    (poverty, poverty_col)
]:
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df.dropna(subset=["Year"], inplace=True)
    df["Year"] = df["Year"].astype(int)

# =========================
# 5. HELPER: KEEP YEARS WITH ENOUGH COUNTRIES
# =========================
def mean_by_year_with_min_count(df, value_col, min_count=30):
    valid = df.dropna(subset=[value_col]).copy()
    counts = valid.groupby("Year")[value_col].count().reset_index(name="n")
    good_years = counts[counts["n"] >= min_count]["Year"]
    result = (
        valid[valid["Year"].isin(good_years)]
        .groupby("Year")[value_col]
        .mean()
        .reset_index()
    )
    return result

# =========================
# 6. GLOBAL TRENDS FOR FIGURE 1
# =========================
gdp_global = mean_by_year_with_min_count(econ, gdp_col, min_count=30)
school_global = mean_by_year_with_min_count(edu, school_col, min_count=30)
internet_global = mean_by_year_with_min_count(infra, internet_col, min_count=30)
poverty_global = mean_by_year_with_min_count(poverty, poverty_col, min_count=30)

# =========================
# 7. FIGURE 1
# PRO SIDE: GLOBAL TRENDS
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
add_title_and_subtitle(
    fig,
    "A Half-Century of Progress in Human Development",
    "Global country averages using years with at least 30 reporting countries"
)

# GDP
ax = axes[0, 0]
ax.plot(gdp_global["Year"], gdp_global[gdp_col], linewidth=2.5)
ax.scatter(
    [gdp_global["Year"].iloc[0], gdp_global["Year"].iloc[-1]],
    [gdp_global[gdp_col].iloc[0], gdp_global[gdp_col].iloc[-1]],
    s=30
)
ax.set_title("Incomes climbed steadily", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("GDP per capita")
ax.annotate(
    "Average GDP per capita\nroughly tripled",
    xy=(gdp_global["Year"].iloc[-1], gdp_global[gdp_col].iloc[-1]),
    xytext=(0.68, 0.74),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
)

# School enrollment
ax = axes[0, 1]
ax.plot(school_global["Year"], school_global[school_col], linewidth=2.5)
ax.scatter(
    [school_global["Year"].iloc[0], school_global["Year"].iloc[-1]],
    [school_global[school_col].iloc[0], school_global[school_col].iloc[-1]],
    s=30
)
ax.set_title("Secondary enrollment became far more common", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("% gross")
ax.annotate(
    "Enrollment rose from\naround 30% to nearly 80%",
    xy=(school_global["Year"].iloc[-1], school_global[school_col].iloc[-1]),
    xytext=(0.46, 0.84),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
)

# Internet
ax = axes[1, 0]
ax.plot(internet_global["Year"], internet_global[internet_col], linewidth=2.5)
ax.scatter(
    [internet_global["Year"].iloc[0], internet_global["Year"].iloc[-1]],
    [internet_global[internet_col].iloc[0], internet_global[internet_col].iloc[-1]],
    s=30
)
ax.set_title("Internet access transformed daily life", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("% of population")
ax.annotate(
    "A near-zero baseline became\nmajority access within two decades",
    xy=(internet_global["Year"].iloc[-1], internet_global[internet_col].iloc[-1]),
    xytext=(0.28, 0.54),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
)

# Poverty
ax = axes[1, 1]
ax.plot(poverty_global["Year"], poverty_global[poverty_col], linewidth=2.5)
ax.scatter(
    [poverty_global["Year"].iloc[0], poverty_global["Year"].iloc[-1]],
    [poverty_global[poverty_col].iloc[0], poverty_global[poverty_col].iloc[-1]],
    s=30
)
ax.set_title("Extreme poverty fell sharply", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("% of population")
ax.annotate(
    "Average poverty fell\nby more than half",
    xy=(poverty_global["Year"].iloc[-1], poverty_global[poverty_col].iloc[-1]),
    xytext=(0.22, 0.68),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
)

for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.savefig("Figure_1_improved.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 8. REGION MAP FOR FIGURE 2
# =========================
region_map = {
    "USA": "North America", "CAN": "North America", "MEX": "North America",
    "GBR": "Europe", "FRA": "Europe", "DEU": "Europe", "ITA": "Europe", "ESP": "Europe",
    "CHN": "East Asia & Pacific", "JPN": "East Asia & Pacific", "KOR": "East Asia & Pacific",
    "IDN": "East Asia & Pacific", "THA": "East Asia & Pacific",
    "IND": "South Asia", "PAK": "South Asia", "BGD": "South Asia", "LKA": "South Asia",
    "NGA": "Sub-Saharan Africa", "KEN": "Sub-Saharan Africa", "ETH": "Sub-Saharan Africa",
    "GHA": "Sub-Saharan Africa", "ZAF": "Sub-Saharan Africa",
    "BRA": "Latin America", "ARG": "Latin America", "CHL": "Latin America",
    "COL": "Latin America", "PER": "Latin America",
    "EGY": "Middle East & North Africa", "MAR": "Middle East & North Africa",
    "SAU": "Middle East & North Africa", "IRN": "Middle East & North Africa"
}

econ["Region"] = econ["Country Code"].map(region_map)
infra["Region"] = infra["Country Code"].map(region_map)

econ_reg = (
    econ.dropna(subset=["Region", gdp_col])
    .groupby(["Year", "Region"])[gdp_col]
    .mean()
    .reset_index()
)

infra_reg = (
    infra.dropna(subset=["Region", internet_col])
    .groupby(["Year", "Region"])[internet_col]
    .mean()
    .reset_index()
)

# =========================
# FIGURE 2
# PRO SIDE: REGIONAL TRENDS
# =========================
selected_regions = [
    "Europe",
    "North America",
    "East Asia & Pacific",
    "South Asia",
    "Sub-Saharan Africa"
]

# colorblind-friendlier palette
region_colors = {
    "Europe": "#1f77b4",              # blue
    "North America": "#ff7f0e",       # orange
    "East Asia & Pacific": "#9467bd", # purple
    "South Asia": "#8c564b",          # brown
    "Sub-Saharan Africa": "#e377c2"   # pink
}

econ_reg_small = econ_reg[econ_reg["Region"].isin(selected_regions)].copy()
infra_reg_small = infra_reg[infra_reg["Region"].isin(selected_regions)].copy()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
add_title_and_subtitle(
    fig,
    "Progress Was Not Isolated to One Part of the World",
    "Selected regional averages show rising incomes and internet access across very different starting points"
)
fig.subplots_adjust(top=0.82, wspace=0.32)

# manual end-label positions so they don't overlap
gdp_label_positions = {
    "Europe": (2020.5, 37500),
    "North America": (2020.5, 36700),
    "East Asia & Pacific": (2020.5, 11000),
    "South Asia": (2020.5, 2300),
    "Sub-Saharan Africa": (2020.5, 3200)
}

internet_label_positions = {
    "Europe": (2020.8, 94),
    "North America": (2020.8, 69.2),
    "East Asia & Pacific": (2020.8, 67.3),
    "South Asia": (2020.8, 37.5),
    "Sub-Saharan Africa": (2020.8, 41.0)
}

# Left panel: GDP
ax = axes[0]
for region in selected_regions:
    temp = econ_reg_small[econ_reg_small["Region"] == region]
    ax.plot(
        temp["Year"], temp[gdp_col],
        linewidth=2.5,
        color=region_colors[region]
    )

    if len(temp) > 0:
        x_end = temp["Year"].iloc[-1]
        y_end = temp[gdp_col].iloc[-1]
        label_x, label_y = gdp_label_positions[region]

        ax.annotate(
            region,
            xy=(x_end, y_end),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=9,
            va="center",
            color=region_colors[region],
            arrowprops=dict(arrowstyle="-", color=region_colors[region], lw=0.8),
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.2)
        )

ax.set_title("Regional GDP per Capita", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("GDP per capita")
ax.set_xlim(econ_reg_small["Year"].min(), econ_reg_small["Year"].max() + 8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.annotate(
    "All selected regions trend upward,\nthough from very different baselines",
    xy=(2000, 14500),
    xytext=(1976, 26000),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
)

# Right panel: Internet
ax = axes[1]
for region in selected_regions:
    temp = infra_reg_small[infra_reg_small["Region"] == region]
    ax.plot(
        temp["Year"], temp[internet_col],
        linewidth=2.5,
        color=region_colors[region]
    )

    if len(temp) > 0:
        x_end = temp["Year"].iloc[-1]
        y_end = temp[internet_col].iloc[-1]
        label_x, label_y = internet_label_positions[region]

        ax.annotate(
            region,
            xy=(x_end, y_end),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=9,
            va="center",
            color=region_colors[region],
            arrowprops=dict(arrowstyle="-", color=region_colors[region], lw=0.8),
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.2)
        )

ax.set_title("Regional Internet Access", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("% of population")
ax.set_xlim(infra_reg_small["Year"].min(), infra_reg_small["Year"].max() + 8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.annotate(
    "Even lower-access regions\nrose sharply after 2000",
    xy=(2012, 35),
    xytext=(1994, 70),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
)

plt.savefig("Figure_2_improved.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# 10. FIGURE 3
# CON SIDE: POVERTY RANKING
# =========================
poverty_nonmissing = poverty.dropna(subset=[poverty_col]).copy()

poverty_year_counts = (
    poverty_nonmissing.groupby("Year")[poverty_col]
    .count()
    .reset_index(name="count")
)

best_poverty_year = int(
    poverty_year_counts.sort_values(["count", "Year"], ascending=[False, False]).iloc[0]["Year"]
)

poverty_latest = poverty_nonmissing[poverty_nonmissing["Year"] == best_poverty_year].copy()
poverty_latest = poverty_latest.sort_values(by=poverty_col, ascending=False).head(20).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 8.5))
add_title_and_subtitle(
    fig,
    "Global Progress Still Bypassed Many Countries",
    f"Top 20 countries by poverty headcount ratio at $3.20/day in {best_poverty_year}"
)

fig.subplots_adjust(top=0.88, left=0.18)

# color logic
colors = []
for i, val in enumerate(poverty_latest[poverty_col]):
    if i < 2:
        colors.append("#d95f02")   # highlight top 2
    elif val >= 40:
        colors.append("#7570b3")   # highlight high-poverty countries
    else:
        colors.append("#1f77b4")   # default

ax.barh(poverty_latest["Country Name"], poverty_latest[poverty_col], color=colors)
ax.invert_yaxis()
ax.set_xlabel("Poverty headcount ratio at $3.20/day (% of population)")
ax.set_ylabel("Country")

top_country = poverty_latest.iloc[0]
ax.annotate(
    f"{top_country['Country Name']} approached {top_country[poverty_col]:.0f}% poverty",
    xy=(top_country[poverty_col], 0),
    xytext=(0.52, 0.78),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
)

ax.axvline(40, linestyle="--", linewidth=1.5, color="gray")
ax.text(40.7, len(poverty_latest) - 0.2, "40% threshold", fontsize=9, va="bottom")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig("Figure_3_improved.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# FIGURE 4
# CON SIDE: GROWTH WITH UNEVEN GAINS
# =========================
def distribution_stats_by_year(df, value_col, min_countries=50):
    rows = []

    valid = df.dropna(subset=[value_col]).copy()

    for year, grp in valid.groupby("Year"):
        vals = np.sort(grp[value_col].values)
        n = len(vals)

        if n < min_countries:
            continue

        p10 = np.percentile(vals, 10)
        p25 = np.percentile(vals, 25)
        p50 = np.percentile(vals, 50)
        p75 = np.percentile(vals, 75)
        p90 = np.percentile(vals, 90)
        mean_val = vals.mean()

        if p10 <= 0 or p25 <= 0:
            continue

        rows.append({
            "Year": year,
            "mean_gdp": mean_val,
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "p90_p10_ratio": p90 / p10,
            "n_countries": n
        })

    return pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)

dist_df = distribution_stats_by_year(econ, gdp_col, min_countries=50)

base_year = int(dist_df["Year"].iloc[0])

# Indexed series for left panel
dist_df["mean_pct_change"] = (
    (dist_df["mean_gdp"] / dist_df["mean_gdp"].iloc[0]) - 1
) * 100

dist_df["median_pct_change"] = (
    (dist_df["p50"] / dist_df["p50"].iloc[0]) - 1
) * 100

dist_df["gap_pct_change"] = (
    (dist_df["p90_p10_ratio"] / dist_df["p90_p10_ratio"].iloc[0]) - 1
) * 100

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
add_title_and_subtitle(
    fig,
    "Global Income Increased, But So Did Inequality",
    "Most countries got richer, but cross-country gaps remained wide"
)
fig.subplots_adjust(top=0.82, wspace=0.28)

# -------------------------
# Left panel: indexed trends
# -------------------------
ax = axes[0]

# leave extra room on the right for direct labels
ax.set_xlim(dist_df["Year"].min(), dist_df["Year"].max() + 4)

# Blue: average (solid, thinner, slightly transparent)
ax.plot(
    dist_df["Year"], dist_df["mean_pct_change"],
    linewidth=2.6, marker="o", markersize=3,
    color="#1f77b4", alpha=0.8,
    label="Global average GDP per capita"
)

# Green: median (dashed, slightly thicker, fully opaque)
ax.plot(
    dist_df["Year"], dist_df["median_pct_change"],
    linewidth=3.0, linestyle="--", marker="o", markersize=3,
    color="#2ca02c", alpha=1.0,
    label="Median country GDP per capita"
)

# Orange: 90/10 gap
ax.plot(
    dist_df["Year"], dist_df["gap_pct_change"],
    linewidth=3.0, marker="o", markersize=3,
    color="#d95f02", alpha=1.0,
    label="Gap between 90th and 10th percentiles"
)

ax.axhline(0, linestyle="--", linewidth=1, color="gray")
ax.set_title("Growth was broad, but gaps persisted", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel(f"% change since {base_year}")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# remove legend
# ax.legend(...)

# -------------------------
# Direct labels at line ends
# -------------------------
last_year = dist_df["Year"].iloc[-1]
label_x = last_year + 0.8

mean_y = dist_df["mean_pct_change"].iloc[-1]
median_y = dist_df["median_pct_change"].iloc[-1]
gap_y = dist_df["gap_pct_change"].iloc[-1]

ax.text(
    label_x, mean_y,
    "Global Average",
    color="#1f77b4",
    fontsize=10,
    va="center",
    ha="left"
)

ax.text(
    label_x, median_y,
    "Median Country",
    color="#2ca02c",
    fontsize=10,
    va="center",
    ha="left"
)

ax.text(
    label_x, gap_y,
    "90/10 Gap",
    color="#d95f02",
    fontsize=10,
    va="center",
    ha="left"
)

# -------------------------
# Short callout annotations
# -------------------------
ax.annotate(
    f"+{mean_y:.0f}%",
    xy=(last_year, mean_y),
    xytext=(0.67, 0.74),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none")
)

ax.annotate(
    f"+{median_y:.0f}%",
    xy=(last_year, median_y),
    xytext=(0.60, 0.60),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none")
)

ax.annotate(
    f"+{gap_y:.0f}%",
    xy=(last_year, gap_y),
    xytext=(0.64, 0.38),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none")
)

# -------------------------
# Right panel: distribution
# -------------------------
ax = axes[1]

ax.fill_between(
    dist_df["Year"], dist_df["p10"], dist_df["p90"],
    alpha=0.18, label="10th–90th percentile range"
)

ax.fill_between(
    dist_df["Year"], dist_df["p25"], dist_df["p75"],
    alpha=0.32, label="25th–75th percentile range"
)

ax.plot(
    dist_df["Year"], dist_df["p50"],
    linewidth=2.8, label="Median country"
)

ax.set_yscale("log")
ax.set_title("Income spread remained wide as countries grew richer", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("GDP per capita (log scale)")
ax.legend(loc="upper left", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.annotate(
    "Most countries got richer,\nbut the spread stayed wide",
    xy=(2008, dist_df.loc[dist_df["Year"] == 2008, "p75"].iloc[0]),
    xytext=(0.44, 0.80),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none")
)

plt.savefig("Figure_4.png", dpi=300, bbox_inches="tight")
plt.show()