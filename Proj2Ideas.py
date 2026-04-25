import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
gdp_col = "average_value_GDP per capita (constant 2010 US$)"
econ_raw = pd.read_csv("data/economy-and-growth.csv")
us_gdp = econ_raw[econ_raw["Country Name"] == "United States"].copy()
us_gdp["Year"]  = pd.to_numeric(us_gdp["Year"],  errors="coerce")
us_gdp[gdp_col] = pd.to_numeric(us_gdp[gdp_col], errors="coerce")
us_gdp = us_gdp.dropna(subset=["Year", gdp_col]).sort_values("Year")
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(10, 7))
fig.subplots_adjust(top=0.84)
fig.text(0.5, 0.97, "The US Economy Has Doubled Living Standards",
         fontsize=20, fontweight="bold", ha="center", va="top",
         fontfamily="serif")
fig.text(0.12, 0.90, "GDP per capita (constant 2010 US$) from 1960 to 2020",
         fontsize=11, fontweight="normal", ha="left", va="top",
         color="#444444", fontfamily="serif")
ax.plot(us_gdp["Year"], us_gdp[gdp_col], color="#378ADD", linewidth=2.5)
ax.fill_between(us_gdp["Year"], us_gdp[gdp_col], alpha=0.1, color="#378ADD")
first = us_gdp.iloc[0]
last  = us_gdp.iloc[-1]
pct_increase = (last[gdp_col] - first[gdp_col]) / first[gdp_col] * 100
ax.annotate(
    f"${last[gdp_col]:,.0f}\n(+{pct_increase:.0f}% since {int(first['Year'])})",
    xy=(last["Year"], last[gdp_col]),
    xytext=(-80, -60),
    textcoords="offset points",
    fontsize=10, color="black", fontfamily="serif",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", linewidth=1),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2)
)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("GDP per capita (constant 2010 US$)", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.grid(axis="x", linestyle=":", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
plt.show()

# Data Transformations
# Percent increase in annotation  (last - first) / first * 100
#
#
# Design Decisions
#
# Fill shading under the line emphasizes magnitude increase for GDP +1
#
# Using constant dollars instead of nominal which is better for same purchasing power reference +2
#
# Title is highly optmisitic, talking about doubled living standards, referencing the ~200%
# increase, but it doesn't fully reflect living standards and it is an average -1
#
# End annotation highlights the impressive 206% increase, does not consider dips such as recession in
# 2008 -1
#
# Using long time span (from 1960) allows us to get impressive 206% numbers, if we were to use 
# the 1974 starting point which we have to use in the con visual due to merge limitations
# it would be 116% increase, significantly less -2

gdp_col = "average_value_GDP per capita (constant 2010 US$)"

poverty_raw = pd.read_csv("data/poverty.csv")
econ_raw    = pd.read_csv("data/economy-and-growth.csv")

top10_col = "average_value_Income share held by highest 10%"
bot10_col = "average_value_Income share held by lowest 10%"

us_ineq = poverty_raw[poverty_raw["Country Name"] == "United States"].copy()
us_ineq["Year"]    = pd.to_numeric(us_ineq["Year"],    errors="coerce")
us_ineq[top10_col] = pd.to_numeric(us_ineq[top10_col], errors="coerce")
us_ineq[bot10_col] = pd.to_numeric(us_ineq[bot10_col], errors="coerce")
us_ineq = us_ineq.dropna(subset=["Year", top10_col, bot10_col]).sort_values("Year")
us_ineq["ratio"] = us_ineq[top10_col] / us_ineq[bot10_col]

us_gdp = econ_raw[econ_raw["Country Name"] == "United States"].copy()
us_gdp["Year"]  = pd.to_numeric(us_gdp["Year"],  errors="coerce")
us_gdp[gdp_col] = pd.to_numeric(us_gdp[gdp_col], errors="coerce")
us_gdp = us_gdp.dropna(subset=["Year", gdp_col]).sort_values("Year")

merged = pd.merge(
    us_ineq[["Year", "ratio"]],
    us_gdp[["Year", gdp_col]],
    on="Year"
).sort_values("Year")

merged["ratio_pct"] = (merged["ratio"] / merged["ratio"].iloc[0] * 100) - 100
merged["gdp_pct"]   = (merged[gdp_col] / merged[gdp_col].iloc[0] * 100) - 100

plt.rcParams["font.family"] = "serif"

fig, ax = plt.subplots(figsize=(10, 7))
fig.subplots_adjust(top=0.84)

fig.text(0.5, 0.97, "GDP In The US Grows Substantially, But At What Cost?",
         fontsize=20, fontweight="bold", ha="center", va="top",
         fontfamily="serif")

fig.text(0.12, 0.90, "Inequality also significantly increases as GDP rises (starting from 1974 baseline 0%)",
         fontsize=11, fontweight="normal", ha="left", va="top",
         color="#444444", fontfamily="serif")

ax.plot(merged["Year"], merged["gdp_pct"],
        color="#378ADD", linewidth=2.5, marker="o", markersize=4,
        label="GDP per capita")

ax.plot(merged["Year"], merged["ratio_pct"],
        color="#D85A30", linewidth=2.5, marker="o", markersize=4,
        label="Income inequality (top 10% / bottom 10% ratio)")

ax.axhline(0, color="#aaa", linewidth=0.8, linestyle=":")

last = merged.iloc[-1]

ax.annotate(
    f"Inequality +{last['ratio_pct']:.0f}% since 1974",
    xy=(last["Year"], last["ratio_pct"]),
    xytext=(-140, 20),
    textcoords="offset points",
    fontsize=10,
    color="black",
    fontfamily="serif",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", linewidth=1),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2)
)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("% change since 1974", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.grid(axis="x", linestyle=":", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=10, framealpha=0.9, loc="upper left")
plt.show()

# Data Transformations
# Computed a new column, top 10% income share divided by bottom 10% income share
# 2x means the top 10% earn 2x more than the bottom 10%.
#
# joining poverty.csv and economy-and-growth.csv on year, limits to starting 
# data visualization from 1974
#
# GDP series and the inequality ratio series transformed to show % change from their
# 1974 starting values. Allows them to be directly comparable
#
# 
# DESIGN DECISIONS
#
# Both series start at 0% and use the same scale. Can make direct comparisons +2
#
# Misleading title name, suggests postive correlation between increasing GDP and 
# increasing inequality when there are many other factors in play -1
#
# Subtitle may immediatly form conclusions in a viewer's mind before they have chance to 
# observe visual fully. -1
#
# The inequality end point is annotated to show increase but not the GDP which is 
# something positive and noteworthy (only focusing on negative) -2
#
# Choice of using bottom and top 10% gives a much higher inequality increase at 49% as opposed to
# top and bottom 20% which is about 34%, purposely choosing more extreme ends. -1



