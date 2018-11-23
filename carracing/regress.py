
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import json
import os

np.set_printoptions(threshold=np.nan)
# *Load necessary libraries*

def load_model(filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    model_params = np.array(data[0]) # assuming other stuff is in data
    b = np.array(model_params[:3])
    weight = np.array(model_params[3:]).reshape(288, 3)
    return weight, b

def load_train(folder):
    allfolders = os.listdir(folder)
    for each in allfolders:

        if each == allfolders[0]:
            data_x = np.load(folder+each)['h']
            data_y = np.load(folder+each)['origin']
            data_action = np.load(folder+each)['action']
        else:
            data_x = np.concatenate((data_x, np.load(folder+each)['h']))
            data_y = np.concatenate((data_y, np.load(folder+each)['origin']))
            data_action = np.concatenate((data_action, np.load(folder+each)['action']))

    return data_x, data_y
#data_x = data_x[:4,:2]
#data_y = data_y[:4,:1]
# *Generate our data*

# In[3]:


# *Add intercept data and normalize*

# In[4]:

data_x, data_y = load_train('record/')
#order = np.random.permutation(len(data_x))
train_x = data_x
train_y = data_y


print(train_x.shape)

# *Shuffle data and produce train and test sets*

# In[5]:



#test_y = np.put(train_y.flatten(), human, test_y.flatten())

#print(train_x.shape)
#print(test_x.shape)

#freevars = np.array([np.random.randint(0, 288, size=int((len(test_x)*train_x.shape[1])/1000)), np.random.randint(288, 576, size=int((len(test_x)*train_x.shape[1])/1000)), np.random.randint(576, 864, size=int((len(test_x)*train_x.shape[1])/1000))])
#freevars = np.array([np.random.randint(0, 576, size=int((len(test_x)*train_x.shape[1])/1000)*2)])
#print(freevars)
def get_gradient(w, b, x, y):
    y_estimate = np.dot(x,w)
    error = (y - y_estimate)
    #print(error)
    mse = (np.square(error)).mean(axis=None)
    gradient = -(1.0/len(x)) * np.dot(x.T,error)
    db = -np.sum(error, axis=0, keepdims=True)
    return gradient, db, mse


# *Create gradient function*

# In[6]:

w = np.random.randn(288,3)
b = np.random.randn(1,3)


actualw,actualb = load_model('log/carracing.cma.16.64.best.json')
#b = b.reshape(1,3)

#ya = w[:4].dot(train_x[:4].T)+b
#print(ya.T.flatten() - train_y[:4].flatten())

alpha = .01
tolerance = 1e-5
#freevars = freevars.reshape(180)

# Perform Gradient Descent
iterations = 1
while True:
    w,b = load_model('log/fitness-19.json')
    gradient, db, error = get_gradient(w, b, train_x, train_y)
    
    #learnrate = np.divide(actualw - w, gradient)
    #print(learnrate)
    #print(np.sum(learnrate, axis=0))
    

    #print(gradient.shape)
    #print(freevars)

    #if(iterations > 50000):
        #break
    #print(gradient)
    #db = np.zeros((1,3))
    new_w = w - alpha * gradient
    new_b = b - alpha * db
    
    #print("x", train_x)
    #print("y", train_y)
    #print("w", new_w)
    #print("b", new_b)
    # Stopping Condition
    if np.sum(abs(new_w - w)) < tolerance:
        print "Converged."
        break

    # Print error every 50 iterations
    if iterations % 100 == 0:
        print "Iteration: %d - Error: %.4f" %(iterations, error)
        savejson(w, filename)
    
    iterations += 1
    w = new_w
    b = new_b

#print "w =",w
#print "Test Cost =", get_gradient(w, b, train_x, test_y)[2]


# *Perform gradient descent to learn model*

# In[9]:

'''
plt.plot(data_x[:,1], data_x.dot(w), c='g', label='Model')
plt.scatter(train_x[:,1], train_y, c='b', label='Train Set')
plt.scatter(test_x[:,1], test_y, c='r', label='Test Set')
plt.grid()
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# *Plot the model obtained*

# In[10]:


w1 = np.linspace(-w[1]*3, w[1]*3, 300)
w0 = np.linspace(-w[0]*3, w[0]*3, 300)
J_vals = np.zeros(shape=(w1.size, w0.size))

for t1, element in enumerate(w1):
    for t2, element2 in enumerate(w0):
        wT = [0, 0]
        wT[1] = element
        wT[0] = element2
        J_vals[t1, t2] = get_gradient(wT, train_x, train_y)[1]

plt.scatter(w[0], w[1], marker='*', color='r', s=40, label='Solution Found')
CS = plt.contour(w0, w1, J_vals, np.logspace(-10,10,50), label='Cost Function')
plt.clabel(CS, inline=1, fontsize=10)
plt.title("Contour Plot of Cost Function")
plt.xlabel("w0")
plt.ylabel("w1")
plt.legend(loc='best')
plt.show()

'''
# *Generate contour plot of the cost function*
