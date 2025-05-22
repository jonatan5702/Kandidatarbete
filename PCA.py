import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("Master_DataFrame.csv")

datum_col = df[['Datum']]

metal_columns = [col for col in df.columns if "ICP-MS" in col]

numeric_data = df[["Datum"] + metal_columns]

numeric_data.dropna(inplace=True)

dates = numeric_data["Datum"]

numeric_data.drop("Datum", axis=1, inplace=True)

pt = PowerTransformer(method='yeo-johnson', standardize=True)
X_transformed = pt.fit_transform(numeric_data)

pca_full = PCA(n_components=None)
pca_full.fit(X_transformed)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plotting the cumulative variance

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Scree Plot')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Choosing the top three components

n_components_3 = 3
pca = PCA(n_components=n_components_3)
X_pca = pca.fit_transform(X_transformed)

pca_columns = [f'PC{i+1}' for i in range(n_components_3)]
df_pca = pd.DataFrame(X_pca, columns=pca_columns)

df_pca_with_datum = pd.concat([dates.reset_index(drop=True), df_pca], axis=1)



loadings_df = pd.DataFrame(pca.components_, columns=metal_columns, index=[f'PC{i+1}' for i in range(n_components_3)])



df_pca_with_datum['Datum'] = pd.to_datetime(df_pca_with_datum['Datum'])

plt.figure(figsize=(10, 6))
for i in range(n_components_3):
    plt.plot(df_pca_with_datum['Datum'], df_pca_with_datum[f'PC{i+1}'], label=f'PC{i+1}')

plt.xlabel('Date')
plt.ylabel('Principal Component Value')
plt.title('Trends of Principal Components (First 3) Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


df_pca_with_datum.to_csv("PCA.csv", index=False)