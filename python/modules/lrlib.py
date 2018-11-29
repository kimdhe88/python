#linearregression library
import numpy as np
from matplotlib import pyplot as plt
from threading import Timer

class LinearRegression:
    def __init__(self, X, Y, theta, t0_alpha=0.01, t1_alpha=0.01):
        self.x = np.column_stack((X,np.ones(len(X))))
        self.y = Y
        self.theta = theta
        self.t0_alpha = t0_alpha
        self.t1_alpha = t1_alpha
        self.m = len(self.y)
        self.runcount = 0

    def get_theta(self):
        return self.theta

    def compute_cost(self, x, y, theta):
        return np.sum(np.square(np.matmul(x, theta) - y)) / (2 * len(y))

    def gradient_descent(self):
        t0 = self.theta[0] - (self.t1_alpha / self.m) * np.sum((np.dot(self.x, self.theta) - self.y) * self.x[:,0])
        t1 = self.theta[1] - (self.t0_alpha / self.m) * np.sum(np.dot(self.x, self.theta) - self.y)
        self.theta = np.array([t0, t1])
        self.runcount += 1

    def showData(self):
        print('step : %-8s' % self.runcount, 'cost : %-12.8f' % self.compute_cost(self.x, self.y, self.theta), 't0 : %-12.8f' % self.theta[0],'t1 : %-12.8f' % self.theta[1])


class drawGraph:
    def __init__(self, X, Y):
        self.x = np.column_stack((X,np.ones(len(X))))
        self.y = Y
        self.theta = np.zeros(2)

        self.fig, self.axs = plt.subplots(2, 2, figsize=(13, 13))
        self.axs01_xlim, self.axs01_ylim = 15, 10

        self.Xs, self.Ys = np.meshgrid(np.linspace(-self.axs01_xlim, self.axs01_xlim, 100), np.linspace(-self.axs01_ylim, self.axs01_ylim, 100))
        self.Zs = np.array([self.compute_cost(self.x, self.y, [t0, t1]) for t0, t1 in zip(np.ravel(self.Xs), np.ravel(self.Ys))])
        self.Zs = np.reshape(self.Zs, self.Xs.shape)

    def compute_cost(self, x, y, theta):
        return np.sum(np.square(np.matmul(x, theta) - y)) / (2 * len(y))

    def update_data(self, theta):
        self.reset_plot()
        self.theta = theta

        self.weightrange = np.linspace(self.theta[0]-2.0, self.theta[0]+2.0, 100)
        self.biasrange = np.linspace(self.theta[1]-2.0,self.theta[1]+2.0, 100)
        cost = self.compute_cost(self.x, self.y, self.theta)
        hypothesis = np.dot(self.x, self.theta)

        costforWeight = []
        costforBias = []

        for i in self.weightrange:
            t_theta = np.array([ i, self.theta[1] ])
            costforWeight.append(self.compute_cost(self.x, self.y, t_theta))
        for i in self.biasrange:
            t_theta = np.array([ self.theta[0], i ])
            costforBias.append(self.compute_cost(self.x, self.y, t_theta))

        self.axs[0, 0].plot(self.x[:,0], self.y, 'bo', self.x[:,0], hypothesis, 'r-')
        self.axs[0, 1].plot(self.theta[0], self.theta[1], 'r*')
        self.axs[0, 1].contour(self.Xs, self.Ys, self.Zs, np.logspace(-10, 10, 50))
        self.axs[1, 0].plot(self.theta[0], cost, 'ro', self.weightrange, costforWeight, 'b-')
        self.axs[1, 1].plot(self.theta[1], cost, 'ro', self.biasrange, costforBias, 'b-')


    def reset_plot(self):
        #self.axs.clear()
        self.axs[0, 0].clear()
        self.axs[0, 1].clear()
        self.axs[1, 0].clear()
        self.axs[1, 1].clear()

        self.axs[0, 0].set_xlabel('x')
        self.axs[0, 0].set_ylabel('y')
        self.axs[0, 0].set_xlim(min(self.x[:,0]) - 0.1, max(self.x[:,0]) + 0.1)
        self.axs[0, 0].set_ylim(min(self.y)-1, max(self.y)+1)

        self.axs[0, 1].set_xlabel('W')
        self.axs[0, 1].set_ylabel('b')
        self.axs[0, 1].set_xlim(-self.axs01_xlim,self.axs01_xlim)
        self.axs[0, 1].set_ylim(-self.axs01_ylim,self.axs01_ylim)

        self.axs[1, 0].set_xlabel('W')
        self.axs[1, 0].set_ylabel('cost(W,b)')

        self.axs[1, 1].set_xlabel('bias')
        self.axs[1, 1].set_ylabel('cost(W,b)')

    def drawplot(self, interval=0.1):
        plt.draw(), plt.pause(interval)

class functionTimer:
   def __init__(self,t,func):
      self.t=t
      self.func = func
      self.thread = Timer(self.t,self.handle_function)

   def handle_function(self):
      self.func()
      self.thread = Timer(self.t,self.handle_function)
      self.thread.start()

   def start(self):
      self.thread.start()

   def cancel(self):
      self.thread.cancel()
