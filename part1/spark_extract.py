from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowthModel
import pickle

# Initialize a SparkSession in local mode
spark = SparkSession.builder \
    .appName("Load and Export FP-Growth Model") \
    .master("local[*]") \
    .config("spark.driver.memory", "192g") \
    .config("spark.executor.memory", "192g") \
    .config("spark.driver.maxResultSize", "0") \
    .getOrCreate()

# Load the pre-computed FP-Growth model from disk
model = FPGrowthModel.load("fp_growth_model")

# Extract the association rules DataFrame from the model
rules_df = model.associationRules

# Convert the DataFrame rows to a list of dictionaries
rules_list = [row.asDict() for row in rules_df.collect()]

# Save the rules list as a pickle file for faster loading later
with open("association_rules.pkl", "wb") as f:
    pickle.dump(rules_list, f)

print("Association rules have been exported to association_rules.pkl")

spark.stop()
