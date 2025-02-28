import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pickle

# Step 1: Load datasets
tracks1 = pd.read_csv("../2023_spotify_ds1.csv")
tracks2 = pd.read_csv("../2023_spotify_ds2.csv")
tracks = pd.concat([tracks1, tracks2], ignore_index=True)
songs = pd.read_csv("../2023_spotify_songs.csv")  # Additional file if needed

# Step 2: Prepare baskets grouped by playlist (using 'pid')
baskets = tracks.groupby("pid")["track_name"].apply(list).tolist()
print("Total baskets (playlists):", len(baskets))

# Step 3: Transform baskets to one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(baskets).transform(baskets)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Step 4: Mine frequent itemsets
min_support = 0.01  # Set support threshold (adjust as needed)
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_itemsets.head())

# Step 5: Generate association rules
min_confidence = 0.5  # Set confidence threshold (adjust as needed)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
print("Association Rules:")
print(rules.head())

# Step 6: Persist the generated model (rules) to disk
rules.to_csv("association_rules.csv", index=False)

with open("association_rules.pkl", "wb") as f:
    pickle.dump(rules, f)

print("Model saved successfully.")
