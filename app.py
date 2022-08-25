#-------------------------------------------------------------------------------
# Importation des modules

import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import include as inc

#-------------------------------------------------------------------------------
# Application principale

st.title("Descente de gradient")
st.write("""
    ### Un exemple de régression linéaire par descente de gradient

""")

# Les données : comme il y en a pas beaucoup nous n'utilisons pas de fichiers
# externes (csv ou autre) pour les stocker.
X = np.array([1.081, 1.854, 2.674, 3.753, 4.693, 5.498, 6.470, 7.386, 7.981, 9.101])
Y = np.array([3.165, 6.047, 4.831, 8.790, 9.266, 14.059, 17.403, 21.370, 21.400, 27.870])
msg = "On souhaite effectuer une régression linéaire avec les données d'apprentissage suivantes :"
st.write(msg)
st.latex('''X = (1.081, 1.854, 2.674, 3.753, 4.693, 5.498, 6.470, 7.386, 7.981, 9.101)''')
st.latex('''Y = (3.165, 6.047, 4.831, 8.790, 9.266, 14.059, 17.403, 21.370, 21.400, 27.870)''')

# Explications à propos du coeur de l'application 
msg = "Au lieu d'utiliser un module Python dédié (en l'occurrence **Scikit-Learn**), "
msg = msg + "nous allons implémenter l'algorithme de **descente de gradient**"
msg = msg + " (voir le ficier README.md dans les sources pour plus de détails)."
st.write(msg)
st.write("""
    On cherche un modèle de la forme : 
""")
st.latex("\\widehat{y}=f(x) = wx + b")
msg = "où $w$ et $b$ sont les valeurs qui minimisent la fonction de coût"
msg = msg + " (on raisonne ici suivant la logique des moindres carrés)."
st.write(msg)

# Calcul avec notre algorithme de descente du gradient
num_iters = 10000
learning_rate = 0.02
w_grad, b_grad = inc.descente_de_gradient(X, Y, num_iters, learning_rate)
msg = "En sortie de notre algorithme on obtient les valeurs suivantes"
msg = msg + f" (avec {num_iters} itérations et" 
msg = msg + f" un learning rate de {learning_rate}) :"
st.write(msg) 
st.latex(f"w = {w_grad}\quad\quad b={b_grad}")

# Calcul avec Scikit-Learn
w_sk, b_sk = inc.regression_lineaire_via_sklearn(X, Y)
st.write("On peut comparer aux valeurs calculées par **Scikit-Learn** :")
st.latex(f"w={w_sk}\quad\quad b={b_sk} ")

# Comparaison entre les deux méthodes
st.write("On constate que les résultats sont très proches ! 😎 ")
st.write("Pour le voir plus facilement, écrivons les différences (avec 20 décimales) :")
delta_w = w_sk - w_grad
delta_b = b_sk - b_grad 
st.latex(f"\Delta w = {delta_w:.20f}\quad\quad \Delta b = {delta_b:.20f}")

#-------------------------------------------------------------------------------
# Conclusion avec le lien vers les sources sur GitHub

st.markdown("""
    <hr>
""", unsafe_allow_html=True)
inc.espace(2)
st.write("""
📝 Sources de l'application :
[https://github.com/pbejian/gradient](https://github.com/pbejian/gradient)
""")
#-------------------------------------------------------------------------------