import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# 1. Create a realistic dataset
# -----------------------------
def create_sample_dataset():
    rng = np.random.default_rng(42)
    months = pd.date_range(start="2023-01-01", periods=365, freq="D")
    n = len(months)

    data = {
        "Date": months,
        "Region": rng.choice(["North", "South", "East", "West"], size=n),
        "Channel": rng.choice(["Online", "Store", "Marketplace"], size=n),
        "Product_Category": rng.choice(["Electronics", "Fashion", "Home", "Books"], size=n),
        "Customer_Age": rng.integers(18, 70, size=n),
        "Quantity": rng.integers(1, 8, size=n),
        "Unit_Price": rng.uniform(20, 180, size=n),
        "Discount_Percent": rng.uniform(0, 35, size=n),
        "Marketing_Spend": rng.uniform(50, 500, size=n),
        "Customer_Satisfaction": rng.uniform(1, 5, size=n),
    }

    # Add realistic patterns
    seasonal = np.sin(np.arange(n) / 7) * 25
    weekend_boost = np.where(months.dayofweek >= 5, 25, 0)
    region_effect = np.array([120 if r == "North" else 80 if r == "South" else 90 if r == "East" else 110 for r in data["Region"]])
    channel_effect = np.array([60 if c == "Online" else 40 if c == "Store" else 55 for c in data["Channel"]])
    category_effect = np.array([70 if c == "Electronics" else 45 if c == "Fashion" else 50 if c == "Home" else 30 for c in data["Product_Category"]])

    base_sales = (
        data["Unit_Price"] * data["Quantity"] * (1 - data["Discount_Percent"] / 100) +
        seasonal + weekend_boost + region_effect + channel_effect + category_effect +
        data["Marketing_Spend"] * 0.2
    )

    # Ensure positive sales
    data["Sales"] = np.maximum(base_sales, 10)
    df = pd.DataFrame(data)
    return df

# -----------------------------
# 2. Load / prepare data
# -----------------------------
def prepare_data(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Year"] = df["Date"].dt.year
    df["Weekend"] = df["Date"].dt.dayofweek.isin([5, 6]).astype(int)
    return df

# -----------------------------
# 3. Summary and statistics
# -----------------------------
def summary_report(df):
    print("Dataset shape:", df.shape)
    print("\nColumns:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDescriptive statistics:")
    print(df.describe(include="all").T)

# -----------------------------
# 4. Visualizations
# -----------------------------
def create_visuals(df):
    out_dir = Path("eda_outputs")
    out_dir.mkdir(exist_ok=True)

    # 1. Histogram of sales
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Sales"], bins=25, kde=True, color="steelblue")
    plt.title("Distribution of Sales")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "sales_distribution.png")
    plt.close()

    # 2. Boxplot by region
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="Region", y="Sales", palette="Set2")
    plt.title("Sales by Region")
    plt.tight_layout()
    plt.savefig(out_dir / "sales_by_region.png")
    plt.close()

    # 3. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr(numeric_cols)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png")
    plt.close()

    # 4. Sales trend by month
    monthly_sales = df.groupby("Month")["Sales"].sum().sort_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker="o", color="darkgreen")
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "monthly_sales_trend.png")
    plt.close()

    # 5. Channel comparison
    channel_sales = df.groupby("Channel")["Sales"].sum().sort_values(ascending=False)
    plt.figure(figsize=(7, 5))
    sns.barplot(x=channel_sales.index, y=channel_sales.values, palette="viridis")
    plt.title("Sales by Channel")
    plt.xlabel("Channel")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig(out_dir / "sales_by_channel.png")
    plt.close()

    # 6. Scatter: sales vs marketing spend
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Marketing_Spend", y="Sales", hue="Channel", alpha=0.7)
    plt.title("Sales vs Marketing Spend")
    plt.tight_layout()
    plt.savefig(out_dir / "sales_vs_marketing.png")
    plt.close()

    # 7. Category performance
    category_sales = df.groupby("Product_Category")["Sales"].mean()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=category_sales.index, y=category_sales.values, palette="mako")
    plt.title("Average Sales by Product Category")
    plt.ylabel("Average Sales")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / "sales_by_category.png")
    plt.close()

# -----------------------------
# 5. Key insights
# -----------------------------
def key_findings(df):
    findings = {}

    # Top influencing factors
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()["Sales"].sort_values(ascending=False)

    findings["strongest_correlations"] = corr[abs(corr) > 0.2].to_dict()

    # Sales by channel
    channel_summary = df.groupby("Channel")["Sales"].agg(["sum", "mean", "count"])
    findings["channel_summary"] = channel_summary.to_dict()

    # Region performance
    region_summary = df.groupby("Region")["Sales"].agg(["sum",*
