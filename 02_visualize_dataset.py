# %% Import excel to dataframe

import pandas as pd

df = pd.read_excel("res/Online Retail.xlsx")
df.head()

# %%  Show the first 10 rows

df.head(10)

# %% Generate descriptive statistics regardless the datatypes

df.info()
df.describe()

# %% Remove all the rows with null value and generate stats again

df = df.dropna()
df.info()
df.describe()

# %% Remove rows with invalid Quantity (Quantity being less than 0)

df['Quantity'] = df['Quantity'].astype(float)
df = df[df['Quantity'] >= 0]

# %% Remove rows with invalid UnitPrice (UnitPrice being less than 0)

df = df[df['UnitPrice'] >= 0]

# %% Only Retain rows with 5-digit StockCode

df['StockCode'] = pd.to_numeric(df['StockCode'], errors='coerce')
df.dropna(subset=['StockCode'], inplace=True)
df['StockCode'] = df['StockCode'].astype(int)

# %% strip all description

df['Description'] = df['Description'].str.strip()

# %% Generate stats again and check the number of rows

df.info()
df.describe()

# %% Plot top 5 selling countries

import matplotlib.pyplot as plt
import seaborn as sns

top5_selling_countries = df["Country"].value_counts()[:5]
sns.barplot(x=top5_selling_countries.index, y=top5_selling_countries.values, palette='tab10')
plt.xlabel("Country")
plt.ylabel("Amount")
plt.title("Top 5 Selling Countries")

# %% Plot top 20 selling products, drawing the bars vertically to save room for product description

top20_selling_products = df["Description"].value_counts()[:20]
sns.barplot(x=top20_selling_products.values, y=top20_selling_products.index, palette="husl")
plt.xlabel("Amount")
plt.ylabel("Product")
plt.title("Top 20 Selling Products")

# %% Focus on sales in UK, filter data by Country

uk = df.loc[df["Country"] == "United Kingdom"]
df_uk = uk.groupby('Country').count()
sns.barplot(data=df_uk, x= 'Country', y= 'InvoiceNo')
plt.xlabel('Country')
plt.ylabel("Amount")
plt.ticklabel_format(style='plain', axis='y')
plt.title("United Kingdom Sales")

#%% Show gross revenue by year-month, by grouping data

from datetime import datetime

df["YearMonth"] = df["InvoiceDate"].apply(
    lambda dt: datetime(year=dt.year, month=dt.month, day=1)
)

df['GrossRevenue'] = df['Quantity'] * df['UnitPrice']
df_ym = df.groupby('YearMonth').sum('GrossRevenue')
sns.lineplot(data=df_ym, x= 'YearMonth', y= 'GrossRevenue')
plt.xlabel('YearMonth')
plt.ylabel("GrossRevenue")
plt.ticklabel_format(style='plain', axis='y')
plt.title("United Kingdom Sales")

# %% save df in pickle format with name "UK.pkl" for next lab activity
# we are only interested in InvoiceNo, StockCode, Description columns
# association rules

df_uk = df.loc[df["Country"] == "United Kingdom"]
df_uk_pkl = df_uk[['InvoiceNo', 'StockCode', 'Description']]
df_uk_pkl

path = 'res/'
filename = 'UK'
extension = '.pkl'
file_path = path + filename + extension
file_path

df_uk_pkl.to_pickle(file_path)
