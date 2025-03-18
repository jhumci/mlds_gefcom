#%%
# Load the Holyday Data

import pandas as pd

# Load the CSV file
df_load = pd.read_csv(r"../../data/interim/Load_history.csv")

# Add the Day of the week

# Ensures that the time column is in datetime format
df_load["datetime"] = pd.to_datetime(df_load["time"])
df_load["day_of_week"] = df_load["datetime"].dt.day_name()
df_load["day_of_week"] = df_load["day_of_week"].astype("category")

df_load.head()
# %% Add the Holyday Data

# Load the CSV file
df_holidays = pd.read_csv(r"../../data/raw/Holiday_List.csv")
df_holidays

# %%
def mark_holidays(df_load, df_holidays):
    # Convert df_holiday into a more usable format
    holidays = {}
    for year in df_holidays.columns[1:]:
        for _, row in df_holidays.iterrows():
            holiday_name = row["Unnamed: 0"]
            date_str = row[year]
            try:
                date = pd.to_datetime(date_str, errors='coerce')
                if pd.notna(date):
                    holidays[date.strftime('%Y-%m-%d')] = holiday_name
            except Exception as e:
                print(f"Error parsing {date_str}: {e}")
    
    # Mark holidays in df_load
    df_load["day_of_week"] = df_load.apply(
        lambda row: "Holiday" if row["datetime"].split()[0] in holidays else row["day_of_week"], axis=1
    )
    
    return df_load

# Ensure time column is in correct format
df_load["datetime"] = pd.to_datetime(df_load["time"])
df_load["datetime"] = df_load["datetime"].dt.strftime('%Y-%m-%d')

# Apply function
df_load = mark_holidays(df_load, df_holidays)
# %%

df_load.head()
# %%
df_load["day_of_week"].value_counts()
# %%

df_load.to_csv(r"../../data/interim/Load_history.csv",index=False)
# %%
