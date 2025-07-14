#%%
import pandas as pd

#%%
df_cerveja = pd.read_excel('data/dados_cerveja.xlsx')
df_nota = pd.read_excel('data/dados_cerveja_nota.xlsx')
# %%
df_nota.head()
# %%
from sklearn import linear_model

X = df_nota[['cerveja']]
y = df_nota[['nota']]

reg = linear_model.LinearRegression()
reg.fit(X, y)

# %%
a,b = reg.intercept_ , reg.coef_[0]
# %%
print(f"Intecpto: {a}, Coeficiente: {b}")

# %%
predict = reg.predict(X.drop_duplicates())

#%%

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title("Relação Cerveja x Notas")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()['cerveja'], predict)

plt.legend(['Observado', f'y = {a.item():.3f} + {b.item():.3f} x'])

# %%
b