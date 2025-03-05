#%%
import pandas as pd

df_load  = pd.read_csv(r"../data/raw/Load_history.csv")

df_load.head()
# %%

# Melt the DataFrame to long format
df_melted = df_load.melt(id_vars=["zone_id", "year", "month", "day"], var_name="hour", value_name="load_kw")

# Convert hour column to integer (extracting the number from 'h1', 'h2', ...)
df_melted["hour"] = df_melted["hour"].str.extract("(\d+)").astype(int)

# Convert "load_kw" from string to float (removing thousand separator)
df_melted["load_kw"] = df_melted["load_kw"].str.replace(",", "").astype(float)

# Create a datetime column
df_melted["time"] = pd.to_datetime(df_melted[["year", "month", "day"]]) + pd.to_timedelta(df_melted["hour"] - 1, unit="h")

# Select final columns
df_final = df_melted[["zone_id", "time", "load_kw"]]

df_final["zone_id"] = df_final["zone_id"].astype("category")
# Display result
print(df_final)


# %%
import plotly.express as px
fig = px.scatter(data_frame=df_final, x = "time",y="load_kw", color = "zone_id")
fig.show()



# %%
df_final.to_csv(r"../data/interim/Load_history.csv")
# %%
