# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ZONE = 1

df = pd.read_csv(r"../../data/final/Load_weather_history.csv")
df.head()

df["time"] = pd.to_datetime(df["time"])

df

#%%
for ZONE in df["zone_id"].unique()[0:1]:
    df_zone = df[df["zone_id"] == ZONE]
    # Filter only columns names load_kw and with pattern station_
    df_zone = df_zone.filter(regex="^(load_kw|station_.*)")
    df_zone

    df_corr = df_zone.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()


    # scatterplot matrix

    df = df[["load_kw", "station_1_temp_c", "station_5_temp_c"]]

    sns.pairplot(df)
    plt.show()


# %%
