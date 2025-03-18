# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv(r"../../data/final/Load_weather_history.csv")
df.head()

df["time"] = pd.to_datetime(df["time"])
df["month"] = pd.to_datetime(df["time"]).dt.month
df["day_of_week"] = pd.to_datetime(df["time"]).dt.day_name()
df["hour_of_day"] = pd.to_datetime(df["time"]).dt.hour

# %% Monthly Box Plots

for ZONE in df["zone_id"].unique()[0:1]:
    df_zone = df[df["zone_id"] == ZONE]

    # Box plot for months of the year
    fig = px.box(df_zone, x="month", y="load_kw", title=f"Zone {ZONE} Load by Month")
    fig.show()



# %% Day of the week Box Plots

# Define the correct order for days of the week
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=day_order, ordered=True)

for ZONE in df["zone_id"].unique()[0:1]:
    df_zone = df[df["zone_id"] == ZONE]

    # Box plot for days of the week
    fig = px.box(df_zone, x="day_of_week", y="load_kw", title=f"Zone {ZONE} Load by Day of the Week")
    fig.show()

# %% Hour of the day Box Plots

for ZONE in df["zone_id"].unique()[0:1]:
    df_zone = df[df["zone_id"] == ZONE]

    # Box plot for hours of the day
    fig = px.box(df_zone, x="hour_of_day", y="load_kw", title=f"Zone {ZONE} Load by Hour of the Day")
    fig.show()

# %%
