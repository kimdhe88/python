import numpy as np
import lrlib as lr
import time
import pandas as pd

#initialize
data = pd.read_csv("/home/hun/lab/python3.6.5/data/data.csv")
temps = data['atemp'].values
rentals = data['cnt'].values / 1000
t0 = np.random.uniform(-200.0,200.0)
t1 = np.random.uniform(-200.0,200.0)
theta = np.array([t0, t1])

learner = lr.LinearRegression(X=temps, Y=rentals, theta=theta, t0_alpha=0.005, t1_alpha=0.005)
t = lr.functionTimer(0.00000001,learner.gradient_descent)
t.start()

painter = lr.drawGraph(temps,rentals)
while True:
    #learner.showData()
    theta = learner.get_theta()
    painter.update_data(theta)
    painter.drawplot(0.001)
