# Databricks notebook source
# MAGIC %md
# MAGIC Pull and save 1 year of News Headlines from NYT

# COMMAND ----------

# =============================
# NYTimes Archive Ingestion
# =============================

API_KEY     = "9cLqd9jAufochxZTdf3XW0MVh4mvzGIO"
BASE_URL    = "https://api.nytimes.com/svc/archive/v1"
DELTA_PATH  = "/mnt/nyt/archive_yearly_delta"   # final Delta store

import requests, json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, dayofmonth, hour

# 1) Initialize Spark (already implicit in Databricks)
spark = SparkSession.builder.getOrCreate()

# 2) Define date range: April 2024 → April 2025
start_year, start_month = 2024, 4
end_year,   end_month   = 2025, 4

months = []
y, m = start_year, start_month
while (y < end_year) or (y == end_year and m <= end_month):
    months.append((y, m))
    m += 1
    if m == 13:
        m = 1
        y += 1

# 3) Loop & append each month’s headlines to Delta
for y, m in months:
    url    = f"{BASE_URL}/{y}/{m}.json"
    params = {"api-key": API_KEY}

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    docs = resp.json().get("response", {}).get("docs", [])
    if not docs:
        continue

    # parallelize & read JSON
    rdd = spark.sparkContext.parallelize([json.dumps(d) for d in docs])
    df  = spark.read.json(rdd)

    # select only the fields you need
    df_sel = df.select(
        col("headline.main").alias("headline"),
        col("pub_date").alias("pub_date"),
        col("section_name").alias("topic")
    )

    # add partition columns
    df_part = df_sel.withColumn("yr", year("pub_date")) \
                    .withColumn("mo", month("pub_date")) \
                    .withColumn("dy", dayofmonth("pub_date")) \
                    .withColumn("hr", hour("pub_date"))

    # write into Delta (append mode), partitioned
    df_part.write \
           .format("delta") \
           .mode("append") \
           .partitionBy("yr","mo","dy","hr") \
           .save(DELTA_PATH)

# 4) Register as a Hive table for easy querying
spark.sql(f"""
  CREATE TABLE IF NOT EXISTS nyt_archive
  USING DELTA
  LOCATION '{DELTA_PATH}'
""")

# 5) Verify by reading & displaying a sample
news_df = spark.table("nyt_archive")
display(news_df.limit(20))


# COMMAND ----------

# # 1) Upgrade typing_extensions to a version MLflow expects
# %pip install --upgrade typing_extensions

# %pip install alpaca_trade_api

# # 2) Install (or reinstall) MLflow
# %pip install --upgrade mlflow

# %pip install xgboost

# # 3) Restart the Python process so the new packages take effect
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Pull and save stock data

# COMMAND ----------

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST
from pyspark.sql.functions import to_timestamp, year, month, dayofmonth

# ----------------------------
# 1) Pull daily SPY bars
# ----------------------------
client = REST(
    "PKY96P1SK8M1NUKCCYSC",
    "Uf15nvt1cNm61OVbKudbnmwqhW155pr3PtdaB4gR",
    base_url="https://data.alpaca.markets",
    api_version="v2"
)

bars = client.get_bars(
    "SPY",
    tradeapi.TimeFrame(1, tradeapi.TimeFrameUnit.Day),
    start="2024-04-01",
    end="2025-04-30"
)
pdf = bars.df.reset_index().rename(columns={"index": "timestamp"})

# ----------------------------
# 2) Create Spark DataFrame
# ----------------------------
stock_df = (
    spark
      .createDataFrame(pdf)
      .withColumn("timestamp", to_timestamp("timestamp"))
)

# ----------------------------
# 3) Add partition columns
# ----------------------------
stock_part = (
    stock_df
      .withColumn("yr", year("timestamp"))
      .withColumn("mo", month("timestamp"))
      .withColumn("dy", dayofmonth("timestamp"))
)

# ----------------------------
# 4) Write to Delta & register
# ----------------------------
DELTA_STOCK_PATH = "/mnt/nyt/stock_daily"

# write out partitioned by yr/mo/dy
stock_part.write \
          .format("delta") \
          .mode("overwrite") \
          .partitionBy("yr", "mo", "dy") \
          .save(DELTA_STOCK_PATH)

# register in the metastore
spark.sql(f"""
  CREATE TABLE IF NOT EXISTS stock_daily
  USING DELTA
  LOCATION '{DELTA_STOCK_PATH}'
""")

# ----------------------------
# 5) Verify
# ----------------------------
val = spark.table("stock_daily")
display(val.orderBy("timestamp").limit(10))


# COMMAND ----------

from pyspark.sql.functions import col

# read the table
nyt_df = spark.table("nyt_archive")

# filter for month = 4 (April)
apr_df = nyt_df.filter(col("mo") == 4)
# or in Databricks:
display(apr_df)


# COMMAND ----------

# MAGIC %md
# MAGIC Create TFIDF Vecotors for each topic per day

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import to_timestamp, col, concat_ws, collect_list, count, window
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml import PipelineModel

# -------------------------------
# 1) Load the archived articles
# -------------------------------
articles = (
    spark
      .table("nyt_archive")                     # your Delta table
      .withColumn("timestamp", to_timestamp(col("pub_date")))
      .select("timestamp", "headline", "topic")
)

# ----------------------------------------------------------
# 2) Group into 24-hour tumbling windows by topic
# ----------------------------------------------------------
grouped = (
    articles
      .groupBy(
         window("timestamp", "24 hours").alias("time_window"),
         col("topic")
      )
      .agg(
         concat_ws(" ", collect_list("headline")).alias("document"),
         count("*").alias("article_count")
      )
      .select(
         col("time_window.start").alias("window_start"),
         col("time_window.end").alias("window_end"),
         "topic",
         "document",
         "article_count"
      )
      .orderBy("window_start", "topic")
)

# --------------------------------
# 3) Build the TF-IDF Pipeline
# --------------------------------
tokenizer = RegexTokenizer(inputCol="document", outputCol="words", pattern="\\W+")
stopper   = StopWordsRemover(inputCol="words",    outputCol="filtered")
hashTF    = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1<<14)
idf       = IDF(inputCol="rawFeatures", outputCol="tfidfFeatures")

tfidf_pipeline = Pipeline(stages=[tokenizer, stopper, hashTF, idf])

# ------------------------------------------------
# 4) Fit the pipeline and **save** the model
# ------------------------------------------------
tfidf_model = tfidf_pipeline.fit(grouped)
tfidf_model.write().overwrite().save("/mnt/nyt/tfidf_model")

# ------------------------------------------------
# 5) Transform to get TF-IDF features
# ------------------------------------------------
tfidf_df = tfidf_model.transform(grouped)

# ------------------------------------------------
# 6) Display the TF-IDF output
# ------------------------------------------------
display(
  tfidf_df.select(
    "window_start",
    "window_end",
    "topic",
    "article_count",
    "tfidfFeatures"
  )
)


# COMMAND ----------

import re
from pyspark.sql.functions import first

# Helper to make column‐safe names
def sanitize(name: str) -> str:
    # replace non-alphanumeric with underscore, collapse multiple, strip edges
    s = re.sub(r'\W+', '_', name)
    return re.sub(r'_+', '_', s).strip('_')

# 1) Pivot article counts
counts_pivot = (
    tfidf_df
      .groupBy("window_start", "window_end")
      .pivot("topic")
      .agg(first("article_count"))
)

# 2) Pivot TF-IDF vectors
tfidf_pivot = (
    tfidf_df
      .groupBy("window_start", "window_end")
      .pivot("topic")
      .agg(first("tfidfFeatures"))
)

# 3) Rename columns to safe names
#    and add appropriate prefixes
counts_safe = counts_pivot
for col_name in counts_pivot.columns:
    if col_name not in ("window_start", "window_end"):
        safe = sanitize(col_name)
        counts_safe = counts_safe.withColumnRenamed(col_name, f"count_{safe}")

tfidf_safe = tfidf_pivot
for col_name in tfidf_pivot.columns:
    if col_name not in ("window_start", "window_end"):
        safe = sanitize(col_name)
        tfidf_safe = tfidf_safe.withColumnRenamed(col_name, f"tfidf_{safe}")

# 4) Join the two
final_df = counts_safe.join(
    tfidf_safe,
    on=["window_start", "window_end"],
    how="inner"
)

# 5) Persist and register
OUT_PATH = "/mnt/nyt/news_tfidfs"

final_df.write \
    .format("delta") \
    .mode("overwrite") \
    .save(OUT_PATH)

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS news_tfidfs
  USING DELTA
  LOCATION '{OUT_PATH}'
""")

# 6) Show schema & sample
final_df.printSchema()
display(final_df.limit(5))