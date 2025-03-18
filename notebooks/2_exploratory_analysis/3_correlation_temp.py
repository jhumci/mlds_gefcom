# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"../../data/final/Load_weather_history.csv")
df.head()

df["time"] = pd.to_datetime(df["time"])

df

#&& Filter only columns names load_kw and wit pattern station_id
df = df[["load_kw", "station_1_temp_c", "station_2_temp_c"]]
df

# %%
df_corr = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(df_corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# %% relplot

sns.relplot(data=df, x="station_1_temp_c", y="load_kw", kind="scatter")
plt.title("Load vs Station 1 Temperature")
plt.show()

# %% scatterplot matrix

sns.pairplot(df)
plt.show()


# %%
