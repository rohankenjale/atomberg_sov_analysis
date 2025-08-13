import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results"
VISUALIZATIONS_DIR = os.path.join(RESULTS_DIR, "visualizations")

if not os.path.exists(VISUALIZATIONS_DIR):
    os.makedirs(VISUALIZATIONS_DIR)


sov_by_query = pd.read_csv(os.path.join(RESULTS_DIR, "sov_by_query.csv"))
sov_by_query_platform = pd.read_csv(os.path.join(RESULTS_DIR, "sov_by_query_platform.csv"))
sentiment_distribution = pd.read_csv(os.path.join(RESULTS_DIR, "sentiment_distribution.csv"))

plt.figure(figsize=(12, 6))
sns.barplot(x="query", y="sov_mentions_pct", hue="brand", data=sov_by_query)
plt.title("Share of Voice by Query")
plt.xlabel("Query")
plt.ylabel("Share of Voice (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, "sov_by_query.png"))
plt.close()

for query in sov_by_query_platform["query"].unique():
    plt.figure(figsize=(12, 6))
    sns.barplot(x="platform", y="sov_mentions_pct", hue="brand", data=sov_by_query_platform[sov_by_query_platform["query"] == query])
    plt.title(f"Share of Voice for '{query}' by Platform")
    plt.xlabel("Platform")
    plt.ylabel("Share of Voice (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f"sov_by_platform_{query.replace(' ', '_')}.png"))
    plt.close()

plt.figure(figsize=(8, 8))
sentiment_distribution.groupby("sentiment_label")["count"].sum().plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title("Sentiment Distribution of Brand Mentions")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, "sentiment_distribution.png"))
plt.close()

print("Visualizations saved to results/visualizations directory.")
