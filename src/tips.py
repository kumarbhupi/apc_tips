import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import scipy.stats

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset


# Carreguem dataset d'exemple
dataset = load_dataset('../tips.csv')
data = dataset.values

x = data[:, np.arange(data.shape[1]) != 1]
x[:, 1] = np.unique(data[:, 2], return_inverse=True)[1]
x[:, 2] = np.unique(data[:, 3], return_inverse=True)[1]
x[:, 3] = np.unique(data[:, 4], return_inverse=True)[1]
x[:, 4] = np.unique(data[:, 5], return_inverse=True)[1]
x = x.astype('float64')

y = data[:, 1]

print("Dimensionalitat de la BBDD:", dataset.shape)
print("Dimensionalitat de les entrades X", x.shape)
print("Dimensionalitat de l'atribut Y", y.shape)

fig, ax = plt.subplots()

tips_x = data[:, 1]
total_bill_y = data[:, 0:1]


# ax.scatter(total_bill_y, tips_x, s=10)
# plt.xlabel('Total a pagar')
# plt.ylabel('Propina')
# fig.show()

# Apartat B
def mse(v1, v2):
    return ((v1 - v2) ** 2).mean()


def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()

    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)

    # Retornem el model entrenat
    return regr


def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t


"""plt.figure()
plt.title("Histograma de l'atribut 0")
plt.xlabel("Attribute Value")
plt.ylabel("Count")

hist = plt.hist(x[:,0], bins=11, range=[np.min(x[:,0]), np.max(x[:,0])], histtype="bar", rwidth=0.8)
plt.show()"""

x_t = standarize(x)

"""hist = plt.hist(x_t[:,0], bins=11, range=[np.min(x_t[:,0]), np.max(x_t[:,0])], histtype="bar", rwidth=0.8)
plt.show()"""

from sklearn.metrics import r2_score

# Extraiem el primer atribut de x i canviem la mida a #exemples, #dimensions de l'atribut.
# En el vostre cas, haureu de triar un atribut com a y, i utilitzar la resta com a x.
total_bill = x[:, 0].reshape(x.shape[0], 1)
total_bill_stand = standarize(total_bill)

regr = regression(total_bill_stand, y)
predicted = regr.predict(total_bill_stand)

# Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
"""plt.figure()
ax = plt.scatter(total_bill_stand, y, s=10)
plt.plot(total_bill_stand[:, 0], predicted, 'r')
plt.show()"""

# Mostrem l'error (MSE i R2)
MSE = mse(y, predicted)
r2 = r2_score(y, predicted)

print("Mean squeared error: ", MSE)
print("R2 score: ", r2)

""" Per a assegurar-nos que el model s'ajusta be a dades noves, no vistes, 
cal evaluar-lo en un conjunt de validacio (i un altre de test en situacions reals).
Com que en aquest cas no en tenim, el generarem separant les dades en 
un 80% d'entrenament i un 20% de validació.
"""


def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0] * train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val


# Dividim dades d'entrenament
x_train, y_train, x_val, y_val = split_data(x, y)

"""plt.figure()
ax = plt.scatter(total_bill_stand, y, s=10)
plt.plot(total_bill_stand[:, 0], predicted, 'r')
plt.show()"""
for i in range(x_train.shape[1]):
    x_t = x_train[:, i]  # seleccionem atribut i en conjunt de train
    x_v = x_val[:, i]  # seleccionem atribut i en conjunt de val.
    x_t = np.reshape(x_t, (x_t.shape[0], 1))
    x_v = np.reshape(x_v, (x_v.shape[0], 1))

    regr = regression(x_t, y_train)

    ax = plt.scatter(x_v, y_val, s=10)
    plt.plot(x_v, regr.predict(x_v), 'r')
    plt.ylabel("Tip")
    plt.xlabel(dataset.columns[np.arange(data.shape[1]) != 1][i])
    plt.show()

    error = mse(y_val, regr.predict(x_v))  # calculem error
    r2 = r2_score(y_val, regr.predict(x_v))



    print("Error en atribut %d: %f" % (i, error))
    print("R2 score en atribut %d: %f" % (i, r2))

