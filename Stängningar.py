import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("Intag_stangning.xlsx")



df.rename(columns={'Intag stängt': 'Stängt', 'Intaget öppet': 'Öppet'}, inplace=True)

df["Stängt"] = pd.to_datetime(df["Stängt"], errors='coerce')
df["Öppet"] = pd.to_datetime(df["Öppet"], errors='coerce')

expanded_data = []

for _, row in df.iterrows():
    current_time = row["Stängt"]
    end_time = row["Öppet"]

    while current_time.date() < end_time.date():
        midnight = pd.Timestamp.combine(current_time.date() + pd.Timedelta(days=1), pd.Timestamp.min.time())
        hours_closed = (midnight - current_time).total_seconds() / 3600

        expanded_data.append({"Datum": current_time.date(), "Tid (timmar)": hours_closed})
        
        current_time = midnight

    hours_closed = (end_time - current_time).total_seconds() / 3600
    expanded_data.append({"Datum": current_time.date(), "Tid (timmar)": hours_closed})

df_expanded = pd.DataFrame(expanded_data)

df_result = df_expanded.groupby("Datum")["Tid (timmar)"].sum().reset_index()

all_dates = pd.date_range(start=df["Stängt"].min().date(), end=df["Öppet"].max().date())
df_full = pd.DataFrame({"Datum": all_dates})
df_full["Datum"] = df_full["Datum"].dt.date

df_final = df_full.merge(df_result, on="Datum", how="left").fillna(0)

df_final["Tid (timmar)"] = df_final["Tid (timmar)"].clip(upper=24)

df_final["Tid (timmar)"] = round(df_final["Tid (timmar)"], 0)

df_filtered = df_final[df_final["Tid (timmar)"] > 0]

df_filtered["År"] = pd.to_datetime(df_filtered["Datum"]).dt.year
df_filtered["Månad"] = pd.to_datetime(df_filtered["Datum"]).dt.month

df_yearly = df_filtered.groupby(df_filtered["Datum"].apply(lambda x: x.year)).agg(
    Stängda_dagar=("Datum", "count"),
    Stängd_tid=("Tid (timmar)", "sum")
).reset_index().rename(columns={"Datum": "År"})  # Rename the index column to 'År'




df_monthly = df_filtered.groupby(["År", "Månad"]).agg(
    Stängda_dagar=("Datum", "count"),
    Stängd_tid=("Tid (timmar)", "sum")
).reset_index()

df_monthly["Datum"] = pd.to_datetime(df_monthly["År"].astype(str) + "-" + df_monthly["Månad"].astype(str) + "-01")


df_monthly.drop(["År", "Månad"], axis=1, inplace=True)

df["År"] = pd.to_datetime(df["Stängt"]).dt.year

counts = df['År'].value_counts()
counts.sort_index(inplace=True)

df_yearly = df_yearly.merge(counts.rename("Stängningar"), left_on="År", right_index=True, how="left")

df_yearly.rename(columns={"Stängda_dagar": "Stängda dagar", "Stängd_tid": "Stängd tid"}, inplace=True)


df_final.to_csv("Stängningar.csv", index=False)

df_yearly.to_csv("Årliga_stängningar.csv", index=False)

df_monthly.to_csv("Månadsstängningar.csv", index=False)


#%%
import seaborn as sns

plt.figure(figsize=(10, 5))
for param in df_yearly.columns:
    if param != "År": 
        plt.figure(figsize=(10, 5)) 
        sns.barplot(x=df_yearly["År"], y=df_yearly[param], palette="Blues")

        plt.xlabel("År")
        plt.ylabel(param)
        plt.title(f"{param} per år")
        plt.xticks(rotation=45)
        plt.show()



#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("Stängningar.csv")



df['Datum'] = pd.to_datetime(df['Datum'])
df['YearMonth'] = df['Datum'].dt.to_period('M')
monthly_sum = df.groupby('YearMonth')['Tid (timmar)'].sum().reset_index()
monthly_sum['YearMonth'] = monthly_sum['YearMonth'].dt.to_timestamp()

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sum, x='YearMonth', y='Tid (timmar)', marker='o')

plt.xlabel('')
plt.ylabel('Timmar [h]')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()



