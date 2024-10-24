# %% import pandas and read the csv file 
# modify the path if needed
import pandas as pd

df = pd.read_csv("res/rectangles.csv")

# df["area"] = ? new column to dataset
df["area"] = df['width'] * df['length']
df.head()

# %%
summary = [
    ("Total Count", df["area"].shape[0]),
    ("Total Area", df["area"].sum()),
    # ("Average Area", ?),
    ("Average Area", df["area"].mean()),
    # ("Maximum Area", ?),
    ("Maximum Area", df["area"].max()),
    # ("Minimum Area", ?),
    ("Minimum Area", df["area"].min()),
]

for key, value in summary:
    print(f"{key}: {str(value)}")

# %% write to a csv
# pd.DataFrame(dict(summary), index=[0]).?
pd.DataFrame(dict(summary), index=[0]).to_csv("res/summary.csv")