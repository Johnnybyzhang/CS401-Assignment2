from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list
from pyspark.ml.fpm import FPGrowth

# Step 1: Initialize Spark in local mode
spark = SparkSession.builder \
    .appName("Spotify Association Rules") \
    .master("local[*]") \
    .getOrCreate()

# Step 2: Load datasets
tracks1 = spark.read.csv("../2023_spotify_ds1.csv", header=True, inferSchema=True)
tracks2 = spark.read.csv("../2023_spotify_ds2.csv", header=True, inferSchema=True)
tracks = tracks1.union(tracks2)

# Optionally, load additional song info
songs = spark.read.csv("../2023_spotify_songs.csv", header=True, inferSchema=True)

# Step 3: Prepare baskets by grouping tracks by 'pid'
baskets = tracks.groupBy("pid").agg(collect_list("track_name").alias("tracks"))
baskets.show(5, truncate=False)

# Step 4: Mine frequent itemsets and association rules with FP-Growth
fpgrowth = FPGrowth(itemsCol="tracks", minSupport=0.01, minConfidence=0.5)
model = fpgrowth.fit(baskets)

print("Frequent Itemsets:")
model.freqItemsets.show()

print("Association Rules:")
model.associationRules.show()

# Step 5: Persist the model for future use
model.save("fp_growth_model")

# Stop Spark session if desired (when processing is complete)
spark.stop()
