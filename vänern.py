import pandas as pd
df = pd.read_csv("V�nern_vattendrag.csv", delimiter=";", skiprows=5, usecols=[0, 1])


df.rename(columns={'Datum (svensk sommartid)': 'Datum', 'Vattenst�nd': 'Vattenst�nd [cm]'}, inplace=True)




m = df['Datum'].between("2001-01-01", "2025-01-01", inclusive="left")

df.drop(m[~m].index, inplace=True)
df.reset_index(inplace=True, drop=True)



df_2 = pd.read_csv("havsniv�er.csv", delimiter=";", skiprows=5, usecols=[0, 1])

df_2.rename(columns={'Datum Tid (UTC)': 'Datum', 'Havsvattenst�nd': 'Havsvattenst�nd [cm]'}, inplace=True)

df_2['Datum'] = pd.to_datetime(df_2['Datum'], errors='coerce')

mask = df_2['Datum'].between("2001-01-01", "2025-01-01", inclusive="left")
df_2 = df_2[mask].reset_index(drop=True)

df_2['Date'] = df_2['Datum'].dt.date
df_medel = df_2.groupby('Date')['Havsvattenst�nd [cm]'].mean().reset_index()

df_medel.columns = ['Datum', 'Havsvattenst�nd [cm]']


df['Datum'] = pd.to_datetime(df['Datum']).dt.date
df_medel['Datum'] = pd.to_datetime(df_medel['Datum']).dt.date

big_df = pd.merge(df, df_medel, on="Datum", how="outer")




big_df.to_csv("havsniv�_medel.csv", index=False)

