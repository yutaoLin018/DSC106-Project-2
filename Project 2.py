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
# 2. SELECT COLUMNS
# =========================
gdp_col = "average_value_GDP per capita (constant 2010 US$)"
school_col = "average_value_School enrollment, secondary (% gross)"
internet_col = "average_value_Individuals using the Internet (% of population)"
poverty_col = "average_value_Poverty headcount ratio at $3.20 a day (2011 PPP) (% of population)"

# Keep only needed columns
econ = econ[["Country Name", "Country Code", "Year", gdp_col]].copy()
edu = edu[["Country Name", "Country Code", "Year", school_col]].copy()
infra = infra[["Country Name", "Country Code", "Year", internet_col]].copy()
poverty = poverty[["Country Name", "Country Code", "Year", poverty_col]].copy()

# Clean year
for df in [econ, edu, infra, poverty]:
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df.dropna(subset=["Year"], inplace=True)
    df["Year"] = df["Year"].astype(int)

# =========================
# 3. GLOBAL AVERAGE TRENDS
# =========================
gdp_global = econ.groupby("Year")[gdp_col].mean().reset_index()
school_global = edu.groupby("Year")[school_col].mean().reset_index()
internet_global = infra.groupby("Year")[internet_col].mean().reset_index()
poverty_global = poverty.groupby("Year")[poverty_col].mean().reset_index()

# =========================
# 4. VIZ 1: SUPPORTING SIDE
#    Global long-run trends
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Across Multiple Measures, Human Development Has Improved Worldwide", fontsize=16)

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
# 5. REGIONS FOR VIZ 2
# =========================
region_map = {
    # North America
    "USA": "North America", "CAN": "North America", "MEX": "North America",
    # Europe
    "GBR": "Europe", "FRA": "Europe", "DEU": "Europe", "ITA": "Europe", "ESP": "Europe",
    # East Asia & Pacific
    "CHN": "East Asia & Pacific", "JPN": "East Asia & Pacific", "KOR": "East Asia & Pacific",
    "IDN": "East Asia & Pacific", "THA": "East Asia & Pacific",
    # South Asia
    "IND": "South Asia", "PAK": "South Asia", "BGD": "South Asia", "LKA": "South Asia",
    # Sub-Saharan Africa
    "NGA": "Sub-Saharan Africa", "KEN": "Sub-Saharan Africa", "ETH": "Sub-Saharan Africa",
    "GHA": "Sub-Saharan Africa", "ZAF": "Sub-Saharan Africa",
    # Latin America
    "BRA": "Latin America", "ARG": "Latin America", "CHL": "Latin America",
    "COL": "Latin America", "PER": "Latin America",
    # Middle East & North Africa
    "EGY": "Middle East & North Africa", "MAR": "Middle East & North Africa",
    "SAU": "Middle East & North Africa", "IRN": "Middle East & North Africa"
}

econ["Region"] = econ["Country Code"].map(region_map)
edu["Region"] = edu["Country Code"].map(region_map)
infra["Region"] = infra["Country Code"].map(region_map)
poverty["Region"] = poverty["Country Code"].map(region_map)

econ_reg = econ.dropna(subset=["Region"]).groupby(["Year", "Region"])[gdp_col].mean().reset_index()
infra_reg = infra.dropna(subset=["Region"]).groupby(["Year", "Region"])[internet_col].mean().reset_index()

# =========================
# 6. VIZ 2: SUPPORTING SIDE
#    Regional trends
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Improvement Has Been Widespread Across Multiple Regions", fontsize=16)

for region in econ_reg["Region"].dropna().unique():
    temp = econ_reg[econ_reg["Region"] == region]
    axes[0].plot(temp["Year"], temp[gdp_col], label=region, linewidth=2)

axes[0].set_title("Regional GDP per Capita Trends")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("GDP per capita")
axes[0].legend(fontsize=8)

for region in infra_reg["Region"].dropna().unique():
    temp = infra_reg[infra_reg["Region"] == region]
    axes[1].plot(temp["Year"], temp[internet_col], label=region, linewidth=2)

axes[1].set_title("Regional Internet Access Trends")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("% of population")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()

# =========================
# VIZ 3: OPPOSING SIDE
# Latest year with actual poverty data
# =========================

year_counts = poverty.groupby("Year")[poverty_col].count().reset_index(name="count")
best_poverty_year = year_counts.sort_values(["count", "Year"], ascending=[False, False]).iloc[0]["Year"]

poverty_latest = poverty[poverty["Year"] == best_poverty_year].dropna(subset=[poverty_col]).copy()
poverty_latest = poverty_latest.sort_values(by=poverty_col, ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(poverty_latest["Country Name"], poverty_latest[poverty_col])
plt.gca().invert_yaxis()
plt.title(f"Global Progress Has Left Many Countries Behind ({int(best_poverty_year)})")
plt.xlabel("Poverty headcount ratio at $3.20/day (% of population)")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# =========================
# 8. VIZ 4: OPPOSING SIDE
#    Top vs bottom inequality gaps
# =========================
latest_gdp_year = econ["Year"].max()
latest_internet_year = infra["Year"].max()

gdp_latest = econ[econ["Year"] == latest_gdp_year].dropna(subset=[gdp_col]).copy()
internet_latest = infra[infra["Year"] == latest_internet_year].dropna(subset=[internet_col]).copy()

gdp_top10 = gdp_latest.nlargest(10, gdp_col)
gdp_bottom10 = gdp_latest.nsmallest(10, gdp_col)

internet_top10 = internet_latest.nlargest(10, internet_col)
internet_bottom10 = internet_latest.nsmallest(10, internet_col)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle("Averages Hide Deep and Persistent Inequality", fontsize=16)

axes[0].barh(gdp_top10["Country Name"], gdp_top10[gdp_col], label="Top 10")
axes[0].barh(gdp_bottom10["Country Name"], gdp_bottom10[gdp_col], label="Bottom 10")
axes[0].set_title(f"GDP per Capita Extremes ({latest_gdp_year})")
axes[0].set_xlabel("GDP per capita")
axes[0].legend()

axes[1].barh(internet_top10["Country Name"], internet_top10[internet_col], label="Top 10")
axes[1].barh(internet_bottom10["Country Name"], internet_bottom10[internet_col], label="Bottom 10")
axes[1].set_title(f"Internet Access Extremes ({latest_internet_year})")
axes[1].set_xlabel("% using the Internet")
axes[1].legend()

plt.tight_layout()
plt.show()