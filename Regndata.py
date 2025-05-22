#%%
import pandas as pd
from datetime import datetime

mf = pd.read_csv("mf.csv", delimiter=",")
df = pd.read_csv("Regndata_SMHI_Nordstan.csv", delimiter=";", skiprows=13, usecols=[0, 1, 2, 3, 4])


df['Representativt dygn'] = pd.to_datetime(df['Representativt dygn'], errors='coerce').dt.date
mf['Datum'] = pd.to_datetime(mf['Datum'], errors='coerce').dt.date



df = df[df["Representativt dygn"] >= datetime(2001, 1, 1).date()].reset_index(drop=True)
df = df.drop(df.columns[[0, 1]], axis = 1)
df = df.rename(columns={"Representativt dygn": "Datum"})
df = df.rename(columns={"Nederbördsmängd": "Nederbördsmängd [mm]"})


# for idx, kval in df["Kvalitet"].items():
#     if kval != "G":
#         print(idx, "Test") 

df = df.drop(df.columns[[2]], axis = 1)

"""
Endast ett fåtal misstänkta värden, det antas att dessa inte komma ha en påverkan på den totala trenden. 
Oavsett kommer isolation forest förhoppningsvis ta hand om det.
"""

MF = pd.merge(mf, df, on="Datum", how="outer")