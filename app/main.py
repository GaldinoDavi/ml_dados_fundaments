#%%
import pandas as pd

#%%
df_cerveja = pd.read_excel('data/dados_cerveja.xlsx')
df_nota = pd.read_excel('data/dados_cerveja_nota.xlsx')
# %%
df_nota.head()
# %%
from sklearn import linear_model
from sklearn import tree

X = df_nota[['cerveja']]
y = df_nota[['nota']]

reg = linear_model.LinearRegression()
reg.fit(X, y)

# %%
a,b = reg.intercept_ , reg.coef_[0]
# %%
print(f"Intecpto: {a}, Coeficiente: {b}")

# %%
predict_reg = reg.predict(X.drop_duplicates())

arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X,y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

arvore_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
arvore_d2.fit(X,y)
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())

#%%

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title("Relação Cerveja x Notas")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()['cerveja'], predict)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_d2)

plt.legend(['Observado',
            f'Regressão: y = {a.item():.3f} + {b.item():.3f} x',
            'Árvore Full',
            'Árvore d2'])

#%%
# Árvore de Decisão

# Mudar a árvore para max_deph = 2 significa que estamos modificando
# o hyper parâmetro

