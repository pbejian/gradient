#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import streamlit as st

#-------------------------------------------------------------------------------

def espace(n):
    """
    Permet de sauter n lignes dans le rendu de streamlit.
    """
    for _ in range(n):
        st.write("")

#-------------------------------------------------------------------------------

st.title("Descente de gradient")

st.write("""
    ### Un exemple de régression linéaire par descente de gradient

""")
#st.write(" ")

#-------------------------------------------------------------------------------

data = {"X" : [1.081, 1.854, 2.674, 3.753, 4.693, 5.498, 6.470, 7.386, 7.981, 9.101],
        "Y" : [3.165, 6.047, 4.831, 8.790, 9.266, 14.059, 17.403, 21.370, 21.400, 27.870]}

X = np.array([1.081, 1.854, 2.674, 3.753, 4.693, 5.498, 6.470, 7.386, 7.981, 9.101])
Y = np.array([3.165, 6.047, 4.831, 8.790, 9.266, 14.059, 17.403, 21.370, 21.400, 27.870])

#-------------------------------------------------------------------------------

msg = "On souhaite effectuer une régression linéaire avec les données d'apprentissage suivantes :"
st.write(msg)

st.latex('''X = (1.081, 1.854, 2.674, 3.753, 4.693, 5.498, 6.470, 7.386, 7.981, 9.101)''')
st.latex('''Y = (3.165, 6.047, 4.831, 8.790, 9.266, 14.059, 17.403, 21.370, 21.400, 27.870)''')

msg = "Au lieu d'utiliser un module Python dédié (en l'occurrence Scikit-Learn), "
msg = msg + "nous allons implémenter l'algorithme de descente de gradient"
msg = msg + " (voir le ficier README.md dans les sources pour plus de détails)."

st.write(msg)

#-------------------------------------------------------------------------------
# Descente de gradient :

alpha = 0.02  # learning rate
w = 0
b = 0
m = len(X)
num_iter = 10000

for _ in range(num_iter): 
    dw = 0
    db = 0
    for i in range(m):
        dw += 2*( w*X[i]+b - Y[i] )*X[i]/m 
        db += 2*( w*X[i]+b - Y[i] )/m
    w_old = w
    b_old = b
    w = w_old - alpha*dw
    b = b_old - alpha*db

#-------------------------------------------------------------------------------

st.write("""
    On cherche un modèle de la forme : 
""")

st.latex("f(x) = wx + b")

msg = "où $w$ et $b$ sont les valeurs qui minimisent la fonction de coût"
msg = msg + " (on raisonne ici suivant la logique des moindres carrés)."
st.write(msg)

#st.latex(f"w={w}\quad b={b} ")

#-------------------------------------------------------------------------------

msg = "En sortie de notre algorithme on obtient les valeurs suivantes"
msg = msg + f" (avec {num_iter} itérations et $\\alpha=${alpha}) :"
st.write(msg) 
st.latex(f"w = {w}\quad\quad b={b}")

#-------------------------------------------------------------------------------

X_train = X.reshape(-1, 1)
Y_train = Y.reshape(-1, 1)

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
w_sk = lin_reg.coef_
b_sk = lin_reg.intercept_
w_sk = w_sk[0][0]
b_sk = b_sk[0]

#-------------------------------------------------------------------------------

st.write("""
    On peut comparer aux valeurs calculées par Scikit-Learn :
""")

st.latex(f"w={w_sk}\quad\quad b={b_sk} ")

st.write("On constate que les résultats sont très proches ! 😎 ")

st.write("Pour le voir plus facilement, écrivons les différences (avec 20 décimales) :")

delta_w = w_sk - w
delta_b = b_sk - b 

st.latex(f"\Delta w = {delta_w:.20f}\quad\quad \Delta b = {delta_b:.20f}")

#-------------------------------------------------------------------------------

espace(2)

msg = "**Idée -** Rendre le nombre d'itérations et le taux d'apprentissage"
msg = msg + " modifiables par l'utilisateur avec des « sliders »."
st.write(msg)

#-------------------------------------------------------------------------------
st.markdown("""
    <hr>
""", unsafe_allow_html=True)

#-------------------------------------------------------------------------------

espace(2)

st.write("""
📝 Sources de l'application :
[https://github.com/pbejian/gradient](https://github.com/pbejian/gradient)
""")

#-------------------------------------------------------------------------------