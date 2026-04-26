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
# CON SIDE: INEQUALITY OVER TIME
# =========================
def decile_gap_by_year(df, value_col, min_countries=50):
    rows = []

    valid = df.dropna(subset=[value_col]).copy()

    for year, grp in valid.groupby("Year"):
        vals = np.sort(grp[value_col].values)
        n = len(vals)

        if n < min_countries:
            continue

        k = max(1, int(np.floor(n * 0.10)))

        bottom_mean = vals[:k].mean()
        top_mean = vals[-k:].mean()
        global_mean = vals.mean()

        if bottom_mean <= 0:
            continue

        gap_ratio = top_mean / bottom_mean

        rows.append({
            "Year": year,
            "global_mean_gdp": global_mean,
            "top_bottom_ratio": gap_ratio,
            "n_countries": n
        })

    out = pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)
    return out

gap_df = decile_gap_by_year(econ, gdp_col, min_countries=50)

# baseline year = first available year in the filtered series
base_year = int(gap_df["Year"].iloc[0])

gap_df["gdp_pct_change"] = (
    (gap_df["global_mean_gdp"] / gap_df["global_mean_gdp"].iloc[0]) - 1
) * 100

gap_df["gap_pct_change"] = (
    (gap_df["top_bottom_ratio"] / gap_df["top_bottom_ratio"].iloc[0]) - 1
) * 100

fig, ax = plt.subplots(figsize=(14, 9))
add_title_and_subtitle(
    fig,
    "Prosperity Rose, But So Did the Global Income Gap",
    f"Global average GDP per capita and the rich-poor country gap, indexed to {base_year} = 0%"
)

ax.plot(
    gap_df["Year"], gap_df["gdp_pct_change"],
    linewidth=3, marker="o", markersize=4,
    label="Global average GDP per capita",
    color="#1f77b4"
)

ax.plot(
    gap_df["Year"], gap_df["gap_pct_change"],
    linewidth=3, marker="o", markersize=4,
    label="Gap between richest 10% and poorest 10% of countries",
    color="#d95f02"
)

ax.axhline(0, linestyle="--", linewidth=1, color="gray")

ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel(f"% change since {base_year}", fontsize=13)
ax.set_title("Rising average prosperity did not mean equally shared gains", fontsize=15)
ax.legend(loc="upper left", fontsize=11)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# annotate final GDP growth
gdp_last = gap_df.iloc[-1]
ax.annotate(
    f"Average GDP per capita\n+{gdp_last['gdp_pct_change']:.0f}% since {base_year}",
    xy=(gdp_last["Year"], gdp_last["gdp_pct_change"]),
    xytext=(0.73, 0.78),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->"),
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none")
)

# annotate final inequality growth
gap_last = gap_df.iloc[-1]
ax.annotate(
    f"Rich-poor country gap\n+{gap_last['gap_pct_change']:.0f}% since {base_year}",
    xy=(gap_last["Year"], gap_last["gap_pct_change"]),
    xytext=(0.70, 0.48),
    textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->"),
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none")
)

plt.savefig("Figure_4_improved.png", dpi=300, bbox_inches="tight")
plt.show()