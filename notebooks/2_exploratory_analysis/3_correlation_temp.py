# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"../../data/final/Load_weather_history.csv")
df.head()

df["time"] = pd.to_datetime(df["time"])

df

#&& Filter only columns names load_kw and with pattern station_
df = df.filter(regex="^(load_kw|station_.*)")
df

# %%
df_corr = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(df_corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# %% scatterplot matrix

sns.pairplot(df)
plt.show()


# %%
