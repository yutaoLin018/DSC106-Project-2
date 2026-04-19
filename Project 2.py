import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
#    PRO SIDE: GLOBAL TRENDS
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Across Multiple Measures, Human Development Has Improved Worldwide", fontsize=18)

axes[0, 0].plot(gdp_global["Year"], gdp_global[gdp_col], linewidth=2)
axes[0, 0].set_title("GDP per Capita Rose Over Time")
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("GDP per capita")

axes[0, 1].plot(school_global["Year"], school_global[school_col], linewidth=2)
axes[0, 1].set_title("Secondary School Enrollment Increased")
axes[0, 1].set_xlabel("Year")
axes[0, 1].set_ylabel("% gross")

axes[1, 0].plot(internet_global["Year"], internet_global[internet_col], linewidth=2)
axes[1, 0].set_title("Internet Access Expanded Rapidly")
axes[1, 0].set_xlabel("Year")
axes[1, 0].set_ylabel("% of population")

axes[1, 1].plot(poverty_global["Year"], poverty_global[poverty_col], linewidth=2)
axes[1, 1].set_title("Poverty Rates Declined")
axes[1, 1].set_xlabel("Year")
axes[1, 1].set_ylabel("% of population")

plt.tight_layout()
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
# 9. FIGURE 2
#    PRO SIDE: REGIONAL TRENDS
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Improvement Has Been Widespread Across Multiple Regions", fontsize=18)

for region in sorted(econ_reg["Region"].dropna().unique()):
    temp = econ_reg[econ_reg["Region"] == region]
    axes[0].plot(temp["Year"], temp[gdp_col], label=region, linewidth=2)

axes[0].set_title("Regional GDP per Capita Trends")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("GDP per capita")
axes[0].legend(fontsize=8)

for region in sorted(infra_reg["Region"].dropna().unique()):
    temp = infra_reg[infra_reg["Region"] == region]
    axes[1].plot(temp["Year"], temp[internet_col], label=region, linewidth=2)

axes[1].set_title("Regional Internet Access Trends")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("% of population")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()

# =========================
# 10. FIGURE 3
#     CON SIDE: POVERTY RANKING
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
poverty_latest = poverty_latest.sort_values(by=poverty_col, ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(poverty_latest["Country Name"], poverty_latest[poverty_col])
plt.gca().invert_yaxis()
plt.title(f"Global Progress Has Left Many Countries Behind ({best_poverty_year})")
plt.xlabel("Poverty headcount ratio at $3.20/day (% of population)")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# =========================
# 11. FIGURE 4
#     CON SIDE: GDP + INTERNET EXTREMES
# =========================

# GDP year: latest year with enough country coverage
gdp_nonmissing = econ.dropna(subset=[gdp_col]).copy()
gdp_year_counts = gdp_nonmissing.groupby("Year")[gdp_col].count().reset_index(name="count")
best_gdp_year = int(
    gdp_year_counts[gdp_year_counts["count"] >= 30]
    .sort_values("Year", ascending=False)
    .iloc[0]["Year"]
)

gdp_latest = gdp_nonmissing[gdp_nonmissing["Year"] == best_gdp_year].copy()
gdp_latest = gdp_latest.sort_values(by=gdp_col)
gdp_bottom10 = gdp_latest.head(10).copy()
gdp_top10 = gdp_latest.tail(10).copy()

# Internet year: use a recent year with enough country coverage
internet_nonmissing = infra.dropna(subset=[internet_col]).copy()

internet_year_counts = (
    internet_nonmissing.groupby("Year")[internet_col]
    .count()
    .reset_index(name="count")
)

print(internet_year_counts.sort_values("Year", ascending=False).head(20))

# pick the latest year that still has at least 100 countries
possible_years = internet_year_counts[internet_year_counts["count"] >= 100]

best_internet_year = int(
    possible_years.sort_values("Year", ascending=False).iloc[0]["Year"]
)

internet_latest = internet_nonmissing[
    internet_nonmissing["Year"] == best_internet_year
].copy()

# keep only reasonable values
internet_latest = internet_latest[
    (internet_latest[internet_col] >= 0) & (internet_latest[internet_col] <= 100)
].copy()

internet_bottom10 = internet_latest.nsmallest(10, internet_col).copy()
internet_top10 = internet_latest.nlargest(10, internet_col).copy()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Average Gains Conceal Enormous Gaps Between Countries", fontsize=18)

axes[0, 0].barh(gdp_bottom10["Country Name"], gdp_bottom10[gdp_col])
axes[0, 0].set_title(f"Lowest GDP per Capita ({best_gdp_year})")
axes[0, 0].set_xlabel("GDP per capita")

axes[0, 1].barh(gdp_top10["Country Name"], gdp_top10[gdp_col])
axes[0, 1].set_title(f"Highest GDP per Capita ({best_gdp_year})")
axes[0, 1].set_xlabel("GDP per capita")

axes[1, 0].barh(internet_bottom10["Country Name"], internet_bottom10[internet_col])
axes[1, 0].set_title(f"Lowest Internet Access ({best_internet_year})")
axes[1, 0].set_xlabel("% using the Internet")

axes[1, 1].barh(internet_top10["Country Name"], internet_top10[internet_col])
axes[1, 1].set_title(f"Highest Internet Access ({best_internet_year})")
axes[1, 1].set_xlabel("% using the Internet")

plt.tight_layout()
plt.show()