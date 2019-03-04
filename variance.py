import numpy as np



var = []
for i in range(211):
    variances = np.var(X_train[i],axis=0)
    var.append(variances)
var = np.array(var)
print(var.shape)
