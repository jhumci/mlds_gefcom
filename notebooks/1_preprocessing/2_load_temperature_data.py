# %%

import pandas as pd

# Load the CSV file
df_load = pd.read_csv(r"../../data/raw/temperature_history.csv")

# Melt the DataFrame to long format
df_melted = df_load.melt(id_vars=["station_id", "year", "month", "day"], var_name="hour", value_name="temp_f")

# Convert hour column to integer (extracting the number from 'h1', 'h2', ...)
df_melted["hour"] = df_melted["hour"].str.extract("(\d+)").astype(int)

# Convert "temp_f" from string to float (if necessary, remove thousand separator)
df_melted["temp_f"] = pd.to_numeric(df_melted["temp_f"], errors='coerce')
# Convert the temperature from Fahrenheit to Celsius
df_melted["temp_c"] = (df_melted["temp_f"] - 32) * 5.0/9.0

# Create a datetime column
df_melted["time"] = pd.to_datetime(df_melted[["year", "month", "day"]]) + pd.to_timedelta(df_melted["hour"] - 1, unit="h")

# Select final columns
df_final = df_melted[["station_id", "time", "temp_c"]]
df_final["station_id"] = df_final["station_id"].astype("category")

# Display result
print(df_final.head())

# Visualization
import plotly.express as px
fig = px.scatter(data_frame=df_final, x="time", y="temp_c", color="station_id")
fig.show()

# Save the cleaned data
df_final.to_csv(r"../../data/interim/Station_history.csv", index=False)

# %%
