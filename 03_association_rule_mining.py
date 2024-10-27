# %% import dataframe from pickle file
import pandas as pd

df = pd.read_pickle("res/UK.pkl")

df.head()


# %% convert dataframe to invoice-based transactional format

df_invoice = df.groupby('InvoiceNo')['Description'].apply(list).reset_index()
df_invoice.head()


# %% apply apriori algorithm to find frequent items

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit_transform(df_invoice['Description'])
df_te = pd.DataFrame(te_array, columns=te.columns_)
df_te.shape

from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(df_te, min_support=0.01, use_colnames= True)
frequent_itemsets

# %% apply apriori algorithm to find association rules

from mlxtend.frequent_patterns import association_rules

rules = association_rules(frequent_itemsets, min_threshold=0.01)
rules


# %% count of frequent itemsets that have more then 1 item

frequent_itemsets_1 = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 1)]
frequent_itemsets_1

# %% count of frequent itemsets that have more then 2 items

frequent_itemsets_2 = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 2)]
frequent_itemsets_2

# %% count of frequent itemsets that have more then 3 items

frequent_itemsets_3 = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 3)]
frequent_itemsets_3

# %% the frequent itemsets that has the most items

frequent_itemsets_most = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 4)]
frequent_itemsets_most


# %% top 10 lift association rules

rules.sort_values('lift', ascending=False).head(10)

# %% scatterplot support vs confidence
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["support"], y=rules["confidence"], alpha=0.5)
plt.xlim(0.000,0.040)
plt.ylim(0.60,0.95)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")


# %% scatterplot support vs lift
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["support"], y=rules["lift"], alpha=0.5)
plt.xlim(0.000,0.040)
plt.ylim(10,90)
plt.xlabel("Support")
plt.ylabel("lift")
plt.title("Support vs lift")