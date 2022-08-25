import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st


def espace(n):
    """
    Cette fonction ne renvoie rien mais affiche n lignes vides
    dans une application streamlit.
    """
    for _ in range(n):
        st.write("")
    return None


def regression_lineaire_via_sklearn(X, Y):
    """
    In  : Deux tableaux Numpy composés de flottants et de taille shape=(m,) 
    Out : Le couple (w, b) calculé par sklearn de telle sorte que la droite 
          de régression ait pour équation y = wx + b 
    """
    # On commence par redimensionner les données :
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    # On crée un modèle de régression linéaire et on l'entraîne avec X et Y
    model = LinearRegression()
    model.fit(X, Y)
    # On récupère les paramètres du modèle (attention ce sont des tableaux)
    w = model.coef_
    b = model.intercept_
    w = w[0][0]
    b = b[0]
    # On renvoie les coefficients sous forme de tuple.
    return (w, b)


def descente_de_gradient(X, Y, num_iters, learning_rate):
    """
    In  : Deux tableaux Numpy composé de flottants et de taille shape=(m,) 
          ainsi que les hypermètres 'num_iters' (nombre d'itération) et 
          'learning_rate' le taux d'apprentissage (souvent noté alpha).
    Out : Le couple (w, b) calculé par la méthode de descente du gradient
          de telle sorte que la droite de régression ait pour équation y = wx + b.       
    """
    # Initialisation (la plus simple)
    w = 0
    b = 0   
    m = len(X)
    # La boucle principale (nombre d'itération dans l'algo)
    for _ in range(num_iters): 
        dw = 0
        db = 0
        for i in range(m):
            dw += 2*( w*X[i]+b - Y[i] )*X[i]/m 
            db += 2*( w*X[i]+b - Y[i] )/m
        w = w - learning_rate*dw
        b = b - learning_rate*db
   # On renvoie les coefficients sous forme de tuple.
    return (w, b)