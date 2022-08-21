# Descente de gradient pour une régression linéaire (univariée)

On travaille sur une régression linéaire dont les valeurs d'entrainement sont les suivantes :

$$
X = (1.081, 1.854, 2.674, 3.753, 4.693, 5.498, 6.470, 7.386, 7.981, 9.101)
$$

$$
Y = (3.165, 6.047, 4.831, 8.790, 9.266, 14.059, 17.403, 21.370, 21.400, 27.870)
$$


La ***fonction de coût*** que l'on doit minimiser est la suivante :

$$
J(w,b) = \dfrac{1}{m} \sum_{k=1}^m  \big((wx_i + b) - y_i\big) ^2
$$


On cherche un modèle de la forme :

$$
f(x) = wx + b
$$

où les coefficients $w$ et $b$ sont les valeurs qui minimisent la fonction de coût. Pour calculer ces deux coefficients on utilise l'algorithme de ***descente de gradient*** qui consiste à répéter les calculs suivants (boucle $\texttt{for}$ ou $\texttt{while}$) :

$$
w := w -\alpha \dfrac{\partial J}{\partial w}(w,b) \\\\
$$

$$
b := b -\alpha \dfrac{\partial J}{\partial b}(w,b)
$$

où $\alpha$ est un paramètre appelé ***learning rate***. Le nombre d'itération ainsi que $\alpha$ sont des « hyperparamètres » qu'il convient de choisir judicieusement (c'est à dire ?). Il faut aussi donner une valeurs initiales à $w$ et $b$. Sauf raison particulière, on peut prendre $w =0$ et $b=0$.  

Voici la dérivée partielle de $J$ par rapport à $w$ :

$$
\dfrac{\partial J}{\partial w}(w,b) = \frac{1}{m} \sum_{k=1}^m  2\big((wx_i + b) - y_i\big) \times x_i
$$

Puis celle par rapport à $b$ :

$$
\dfrac{\partial J}{\partial w}(w,b) = \frac{1}{m} \sum_{k=1}^m  2\big((wx_i + b) - y_i\big)
$$


**Remarque --** Dans cette version de l'algorithme de descente de gradient on utilise plusieurs boucles $\texttt{for}$. Pour être efficace si l'on a beaucoup de données, il serait judicieux de « vectoriser » les calculs.