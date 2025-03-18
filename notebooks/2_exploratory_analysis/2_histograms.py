# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv(r"../../data/final/Load_weather_history.csv")
df.head()

df["time"] = pd.to_datetime(df["time"])


# %% Monthly Box Plots

for ZONE in df["zone_id"].unique()[0:1]:
    df_zone = df[df["zone_id"] == ZONE]

    # Histogram of the load_kw column
    fig = px.histogram(df_zone, x="load_kw", title=f"Zone {ZONE} Load Histogram")
    fig.show()
    
# %%
