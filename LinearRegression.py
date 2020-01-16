import numpy as np
from tensorflow.keras import datasets

#######################Optional Dataset
(X_train, Y_train), (X_test, Y_test) = datasets.boston_housing.load_data()

X_train = X_train.T #(nx, m)
X_test = X_test.T

Y_train = Y_train.reshape(1, len(Y_train)) #(1, m)
Y_test = Y_test.reshape(1, len(Y_test))

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

######################

class LinearRegression():
    '''
    A class for the implementation of the linear regression model of machine learning.
    
    '''
    def __init__(self):
        '''
        Initialises the intercept and coefficients of the linear model 
        '''
        self.intercept_ = 0
        self.coef_ = 0
        self.costs = []
        self.learning_rate = 0
        print('Linear Regression model based on Gradient Descent.\n X_train shape: (nx, m)\n Y_train shape: (1, m) m: total training examples')
    
    def _set_learning_rate(self, X, Y):
        '''
        Returns suitable learning rate for the dataset
        '''
        self.intercept_ = 0
        self.coef_ = np.zeros((X.shape[0], 1))
        m = X.shape[1]
        optimum_found = False
        self.learning_rate = .1
        print('Searching Optimum learning Rate !')
        
        while(not optimum_found):
            Ypred = np.dot(self.coef_.T, X) + self.intercept_
            
            i_cost = (1/m) * np.sum((Ypred - Y)**2) #cost when w and b are 0
            
            f_cost = self._propagate(X, Y)
            
            if f_cost - i_cost <= 0:
                optimum_found = True
            else:
                self.intercept_ = 0
                self.coef_ = np.zeros((X.shape[0], 1))
                self.learning_rate /= 5
        print('Optimum Found training with Learning rate: ', self.learning_rate)
    
    
    def _propagate(self, X, Y):
        '''
        Fuction forword progates the neural network and finds the gradients and cost
        
        '''
        m = X_train.shape[1]
        
        Ypred = np.dot(self.coef_.T, X) + self.intercept_
        
        d_intercept_ = (1/m) * np.sum(Ypred - Y) 
        d_coef_ = (1/m) * np.dot(X, (Ypred - Y).T)
        
        assert (self.coef_.shape == d_coef_.shape)
        
        self.intercept_ -= self.learning_rate * d_intercept_
        self.coef_ -= self.learning_rate * d_coef_
        
        Ypred = np.dot(self.coef_.T, X) + self.intercept_
        cost = (1/m) * np.sum( (Ypred - Y)**2 )

        return cost
    
    
    def fit(self, X_train, Y_train, iterations=100, verbose=False):
        '''
        Fuction optimises the the model
        '''
        #Initialising parameters shape of w: (nx, 1) nx: number of features
        self._set_learning_rate(X_train, Y_train)
        
        self.coef_, intercept_ = np.zeros((X_train.shape[0], 1)), 0
        self.costs = []
        print('Training model.... we thank you for your patience!')
        for step in range(iterations):
            #iterate to reduce the cost
            
            cost = self._propagate(X_train, Y_train)
            
            #Keep Record of the cost at every 5 step
            if step % 5 == 0:
                self.costs.append(cost)
            
            #Print cost at every 10 step
            if step % 10 == 0 and verbose:
                print('Cost at {0} step: {1}'.format(step, cost))          
            
        print('\nTraining completed Try predict(X), score(X, Y), cost_plot() methods to inspect')
            
    def predict(self, X_test):
        '''
        Predicts the output of the testing dataset based on the coefficients and intercept
        '''
        return np.dot(self.coef_.T, X_test) + self.intercept_
    
    
    def cost_plot(self):
        '''
        plots the descent of the costs
        '''
        import matplotlib.pyplot as plt
        plt.legend('Learning Rate: '.format(self.learning_rate))
        plt.xlabel('iteratons')
        plt.ylabel('cost')
        plt.title('Cost vs Iteration plot')
        plt.plot(self.costs)
        
    def score(self, X, Y):
        '''
        Calculate the R squared value to show how well the model is fitted
        '''
        Ypred = self.predict(X)
        Ymean = np.mean(Y)
        
        return np.sum((Ypred - Ymean)**2) / np.sum((Y - Ymean)**2)

model = LinearRegression()
model.fit(X_train, Y_train, 1_000)
model.cost_plot()
Yp = model.predict(X_test)
plt.plot(Yp[0], color='red')
plt.plot(Y_test[0], color='green')
model.score(X_test, Y_test)