#%%

import pandas as pd

# Load data
df_load = pd.read_csv(r"../../data/interim/Load_history.csv")
df_weather = pd.read_csv(r"../../data/interim/Station_history.csv")

# Ensure Time column is in datetime format
df_load["time"] = pd.to_datetime(df_load["time"])
df_weather["time"] = pd.to_datetime(df_weather["time"])


# Merge the two DataFrames

# Pivot weather data to have one column per station
df_weather_pivot = df_weather.pivot(index='time', columns='station_id', values='temp_c')

# Rename columns to meaningful names
df_weather_pivot.columns = [f"station_{int(col)}_temp_c" for col in df_weather_pivot.columns]

# Reset index to merge properly
df_weather_pivot = df_weather_pivot.reset_index()

# Merge with df_load
df_merged = df_load.merge(df_weather_pivot, left_on=df_load['time'], right_on='time', how='left')
df_merged
#%%
# Drop duplicate 'time' column
df_merged = df_merged.drop(columns=['time_y','time_x','datetime'])
df_merged.head()


# %% drop nan values

df_merged = df_merged.dropna()

#%%
df_merged["time"] = pd.to_datetime(df_merged["time"])
df_merged["time"]
#%%
df_merged.to_csv(r"../../data/final/Load_weather_history.csv", index=False, date_format='%Y-%m-%d %H:%M:%S')
# %%
