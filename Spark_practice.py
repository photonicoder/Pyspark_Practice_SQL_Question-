# Databricks notebook source
# Extended DataFrame 1: Sales Data
data1 = [
    (1, "2023-01-01", 500, 100),
    (2, "2023-01-02", 600, 120),
    (3, "2023-01-03", 700, 130),
    (4, "2023-01-04", 800, 150),
    (5, "2023-01-05", 900, 170),
    (6, "2023-01-06", 1100, 190),
    (7, "2023-01-07", 1200, 200),
    (8, "2023-01-08", 1300, 220),
    (9, "2023-01-09", 1400, 240),
    (10, "2023-01-10", 1500, 260),
    (11, "2023-01-11", 1600, 280),
    (12, "2023-01-12", 1700, 300),
    (13, "2023-01-13", 1800, 320),
    (14, "2023-01-14", 1900, 340),
    (15, "2023-01-15", 2000, 360),
]

columns1 = ["ID", "SaleDate", "Revenue", "UnitsSold"]
df1 = spark.createDataFrame(data1, columns1)

# Extended DataFrame 2: Customer Data
data2 = [
    (1, "Alice", "2023-01-01", "+1-123-456-7890"),
    (2, "Bob", "2023-01-02", "+1-234-567-8901"),
    (3, "Charlie", "2023-01-03", "+1-345-678-9012"),
    (4, "David", "2023-01-04", "+1-456-789-0123"),
    (5, "Eve", "2023-01-05", "+1-567-890-1234"),
    (6, "Frank", "2023-01-06", "+1-678-901-2345"),
    (7, "Grace", "2023-01-07", "+1-789-012-3456"),
    (8, "Hannah", "2023-01-08", "+1-890-123-4567"),
    (9, "Isaac", "2023-01-09", "+1-901-234-5678"),
    (10, "Jack", "2023-01-10", "+1-012-345-6789"),
    (11, "Kathy", "2023-01-11", "+1-123-456-7891"),
    (12, "Liam", "2023-01-12", "+1-234-567-8902"),
    (13, "Mona", "2023-01-13", "+1-345-678-9013"),
    (14, "Noah", "2023-01-14", "+1-456-789-0124"),
    (15, "Olivia", "2023-01-15", "+1-567-890-1235"),
]

columns2 = ["CustomerID", "Name", "JoinDate", "PhoneNumber"]
df2 = spark.createDataFrame(data2, columns2)

df1.show()
df2.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## How do you calculate the rolling sum of sales for the past 3 days?

# COMMAND ----------

from pyspark.sql.functions import col, sum
from pyspark.sql.window import Window

# Convert SaleDate to timestamp (or integer for sorting)
df1 = df1.withColumn("SaleDate", col("SaleDate").cast("timestamp"))

# Define window with rowsBetween for a row-based approach
windows = Window.orderBy(col("SaleDate")).rowsBetween(-2, 0)

# Apply window function
df1 = df1.withColumn("revenue_3days", sum(col("Revenue")).over(windows))

df1.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## How do you calculate the rolling average of revenue over the last 5 days?

# COMMAND ----------

from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
df1=df1.withColumn("salesDate",to_date(col("SaleDate"),"yyyy-MM-dd"))
window_spec=Window.orderBy("salesDate").rowsBetween(-5,0)
df1=df1.withColumn("average_5_days",avg("Revenue").over(window_spec))
df1.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## How can you split a phone number column into CountryCode, AreaCode, and PhoneNumber parts?

# COMMAND ----------

from pyspark.sql.functions import col, split, concat_ws

df2 = df2.withColumn("CountryCode", split(col("PhoneNumber"), "-").getItem(0)) \
         .withColumn("AreaCode", split(col("PhoneNumber"), "-").getItem(1)) \
         .withColumn("PhoneNumber", concat_ws("", split(col("PhoneNumber"), "-").getItem(2).cast("string"), split(col("PhoneNumber"), "-").getItem(3).cast("string")))

df2.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## How do you find the day-over-day percentage change in sales revenue?

# COMMAND ----------

window=Window.orderBy(col("SaleDate").desc())
df1=df1.withColumn("previous_sale_date",lag("Revenue").over(window))
df1=df1.withColumn("percenatge_change",(col("Revenue")-col("previous_sale_date"))/col("previous_sale_date")*100)
df1.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## How do you group sales data by week and calculate the total sales and units sold per week?
# MAGIC

# COMMAND ----------

df1=df1.withColumn("weeknum",weekofyear(col("SaleDate")))
df1.groupBy("Weeknum").agg(sum("Revenue").alias("TotalRevenue"),sum("UnitsSold").alias("TotalUnitsSold")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## How can you rank products based on units sold and revenue for each day?

# COMMAND ----------

window=Window.partitionBy("SaleDate").orderBy(col("unitsSold").desc(),col("Revenue").desc())
df1=df1.withColumn("rank",rank().over(window))

# COMMAND ----------

# MAGIC %md
# MAGIC ## How do you filter rows where the revenue is above a certain threshold (e.g., > 700)?

# COMMAND ----------

df1.filter("Revenue>700").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## How do you calculate the cumulative sum of units sold over time for each product?

# COMMAND ----------

df1.groupBy("id").agg(sum("unitsSold").alias("TotalUnitsSold")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## How do you calculate the moving average of units sold for a rolling window of 7 days?

# COMMAND ----------

window_spec=Window.orderBy("salesDate").rowsBetween(-7,0)
df1=df1.withColumn("average_5_days",avg("UnitsSold").over(window_spec))
df1.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## How can you add a column for the difference in revenue between consecutive days?

# COMMAND ----------

window_spec=Window.orderBy("SaleDate")
df1=df1.withColumn("prev_revenue",lag("Revenue").over(window_spec))
df1=df1.withColumn("diff_revenue",col("Revenue")-col("prev_revenue"))
df1.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## How do you create a new column that categorizes products into "High", "Medium", or "Low" based on revenue?

# COMMAND ----------

df1=df1.withColumn("categorize",when(col("Revenue")>800,"High").when((col("Revenue")<800) & (col("Revenue")>=500),"Medium").otherwise("Low"))
df1.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## How do you split a date column into individual columns for year, month, and day?

# COMMAND ----------

df1=df1.withColumn("year",year(col("SaleDate")))\
        .withColumn("month",month(col("SaleDate")))\
        .withColumn("day",dayofmonth(col("SaleDate")))
df1.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## How do you extract the month from a SaleDate column in the format "YYYY-MM"?

# COMMAND ----------

df1.withColumn("date_format",to_date(col("SaleDate"),"YYYY-MM")).show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## How can you generate the total revenue per department by performing a group-by on Department?

# COMMAND ----------


