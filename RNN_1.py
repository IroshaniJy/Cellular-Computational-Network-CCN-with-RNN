#reated on Thu Feb  8 09:54:17 2018

#@author: Iroshani

# Implementation of a CCN with four RNNs. Learning is performed synchronously.


import tensorflow as tf
import numpy as np
import pandas as pd
import time
from tensorflow.contrib import rnn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from threading import Thread, Barrier


#defining a global variable
pred_train = np.empty(shape=[0,5],dtype=float)
pred_test = np.empty(shape=[0,5],dtype=float)

#random number generation
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

#thread synchronization
num_cells = 4
barrier1 = Barrier(4)
barrier2 = Barrier(4)

def init_weights(shape):
    #Weight initialization
    weights = tf.random_normal(shape, stddev=0.0001)
    return tf.Variable(weights)

def calcout(out, w):
    yhat = tf.matmul(out[-1], w)  # output
    return yhat

def init_rnn(thread_id, x_size, h_size, y_size, time_steps):
    print("---Cell %d is initialized---\n"%thread_id)
    # Create symbols to represent input and  target
    X = tf.placeholder(dtype = tf.float32, shape=[None, time_steps, x_size])
    y = tf.placeholder(dtype = tf.float32, shape=[None, y_size])

    # Weight initializations
    w = init_weights((h_size, y_size))
    
    # Forward propagation
    input_ = tf.unstack(X, time_steps,1)
    lstmlayer = rnn.BasicLSTMCell(h_size, reuse=False)
    output_ = rnn.static_rnn(lstmlayer, input_, dtype = tf.float32)
    yhat = calcout(output_[-1], w)
   
    # Backward propagation
    cost = tf.losses.mean_squared_error(y,yhat)
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    # Initialized and run the model 
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    return X, y, yhat, cost, updates, sess 
    

def train_rnn(thread_id, X, y, timesteps, yhat, cost, updates, sess, train_X, test_X, train_y, test_y):
    print("---Cell %d started training---\n"%thread_id)
    
    cost_history = np.empty(shape=[1],dtype=float)
    pred_train = np.zeros((train_X.shape[0],5))
    train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
    for epoch in range(200):
        #Train with each example
        for i in range(train_X.shape[0]):
            pred_train[i,thread_id] = sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
            #set the neighbor cell connectivity 
            # cell 1 - connected to 2 and 4
            # cell 2 - connected to 1 and 4
            # cell 3 - connected to 2 and 4
            # cell 4 - connected to 2 and 3
            if i>10:
                if thread_id !=2:
                    train_X[i+1:i+2,-1,5] = pred_train[i,2]
                else:
                    train_X[i+1:i+2,-1,5] = pred_train[i,1]
                if thread_id !=4:
                    train_X[i+1:i+2,-1,6] = pred_train[i,4]
                else:
                    train_X[i+1:i+2,-1,6] = pred_train[i,3]
            # synchronization
            barrier1.wait()
        cost_history = np.append(cost_history, sess.run(cost, feed_dict={X: train_X, y: train_y}))
            
        pred = sess.run(yhat,feed_dict={X: train_X})
        train_mse = mean_squared_error(train_y, pred)

        print("Cell = %d, Epoch = %d, train mse = %f \n" % (thread_id, epoch + 1, train_mse))
     
    print("---Cell %d finished training---\n"%thread_id)
    return sess, yhat, pred, train_mse, cost_history

def test_rnn(thread_id, X, y, yhat, sess, test_X, test_y):
    print("---Cell %d started testing---\n"%thread_id)
    
    pred_test = np.zeros((test_X.shape[0],5))
    test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
    for i in range(len(test_X)):
        pred_test[i,thread_id] = sess.run(yhat, feed_dict={X: test_X[i:i+1]})
        #set the neighbor cell connectivity 
        # cell 1 - connected to 2 and 4
        # cell 2 - connected to 1 and 4
        # cell 3 - connected to 2 and 4
        # cell 4 - connected to 2 and 3
        if i>10:
            if thread_id !=2:
                test_X[i+1:i+2,-1,5] = pred_test[i,2]
            else:
                test_X[i+1:i+2,-1,5] = pred_test[i,1]
            if thread_id !=4:
                test_X[i+1:i+2,-1,6] = pred_test[i,4]
            else:
                test_X[i+1:i+2,-1,6] = pred_test[i,3]
        # synchronization
        barrier2.wait()

    test_mse  = mean_squared_error(test_y, pred_test[:,thread_id])
    print("Cell = %d, test mse = %f \n" % (thread_id,test_mse))
    
    print("---Cell %d finished testing---\n"%thread_id)
    return sess, pred_test, test_mse

def load_data():
        dw = loadmat('dw.mat')
        dv = loadmat('dv.mat')
        dw_2 = loadmat('test_dw.mat')
        dv_2 = loadmat('test_dv.mat')

        train_dw = dw['dw']
        train_dv = dv['dv']
        test_dw = dw_2['test_dw']
        test_dv = dv_2['test_dv']

        scaler = MinMaxScaler()
        train_dw = scaler.fit_transform(train_dw)
        train_dv = scaler.fit_transform(train_dv)
        test_dw = scaler.fit_transform(test_dw)
        test_dv = scaler.fit_transform(test_dv)

        return train_dw, train_dv, test_dw, test_dv

# process the train and test data specific to each cell ( A cell is represented in thread_id)
def process_data(thread_id, train_dw, train_dv, test_dw, test_dv):
    print("---Started processing data for Cell %d ---\n"%thread_id)
    rows = len(train_dw)
    # train_input = dv(t),dw(t),dw(-1),dw(t-2),dw_pred_neighbor1(t),dw_pred_neighbor2(t)
    train_input = np.column_stack((np.ones((rows-3)),train_dv[2:(rows-1),thread_id],train_dw[2:(rows-1),thread_id],train_dw[1:(rows-2),thread_id],train_dw[0:(rows-3),thread_id]))
    # train target = dw(t+1)
    train_target = train_dw[3:rows,thread_id].reshape(rows-3,1)
        
    test_rows = len(test_dw)
    # test_input = dv(t),dw(t),dw(-1),dw(t-2),dw_pred_neighbor1(t),dw_pred_neighbor2(t)
    test_input = np.column_stack((np.ones((test_rows-3)),test_dv[2:(test_rows-1),thread_id],test_dw[2:(test_rows-1),thread_id],test_dw[1:(test_rows-2),thread_id],test_dw[0:(test_rows-3),thread_id]))
    # test target = dw(t+1)
    test_target =  test_dw[3:test_rows,thread_id].reshape(test_rows-3,1)
        
    #set the neighbor cell connectivity 
    # cell 1 - connected to 2 and 4
    # cell 2 - connected to 1 and 4
    # cell 3 - connected to 2 and 4
    # cell 4 - connected to 2 and 3
    if thread_id != 2:
        train_input = np.column_stack((train_input, train_dw[2:(rows-1),2]))
        test_input = np.column_stack((test_input, test_dw[2:(test_rows-1),2]))
    else:
        train_input = np.column_stack((train_input, train_dw[2:(rows-1),1]))
        test_input = np.column_stack((test_input, test_dw[2:(test_rows-1),1]))
    if thread_id != 4:
        train_input = np.column_stack((train_input, train_dw[2:(rows-1),4]))
        test_input = np.column_stack((test_input, test_dw[2:(test_rows-1),4]))
    else:
        train_input = np.column_stack((train_input, train_dw[2:(rows-1),3]))
        test_input = np.column_stack((test_input, test_dw[2:(test_rows-1),3]))
        
    print("---Finished processing data for Cell %d ---\n"%thread_id)
    return train_input, test_input, train_target, test_target
    
def worker_thread(thread_id, train_dw, train_dv, test_dw, test_dv):
    
    #process data
    train_X, test_X, train_y, test_y= process_data(thread_id, train_dw, train_dv, test_dw, test_dv)

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes
    h_size = 10                 # Number of hidden nodes
    y_size = 1   		            # Number of outputs
    time_steps = 1
    # initialize rnn structure
    X, y, yhat, cost, updates, sess = init_rnn(thread_id, x_size, h_size, y_size, time_steps)
    
    # train rnn
    sess, yhat, pred, train_mse, cost_history = train_rnn(thread_id, X, y, time_steps,yhat, cost, updates, sess,  train_X, test_X, train_y, test_y)
   
    # reset thread synch
    barrier1.reset()

    # test rnn
    sess, pred_test, test_mse= test_rnn(thread_id, X, y, yhat, sess, test_X, test_y)
 
    # reset thread synch
    barrier2.reset()

   
    # Create an object to save the model and save
    # This can be restored later
    model_path = "data/model_"+str(thread_id)+".ckpt"
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

    #save training and testing outputs
    df = pd.DataFrame(data=np.column_stack((train_y,pred)),columns=['train_target','train_predicted'])
    df.to_csv("data/train_data_cell_"+str(thread_id)+".csv",index=True)
    print("Cell"+str(thread_id)+" data saved as:train_data_cell_"+str(thread_id)+".csv")

    df = pd.DataFrame(data=np.column_stack((range(len(cost_history)),cost_history)),columns=['epoch','cost'])
    df.to_csv("data/cost_cell_"+str(thread_id)+".csv",index=True)
    print("Cell"+str(thread_id)+" data saved as:cost_cell_"+str(thread_id)+".csv")


    df = pd.DataFrame(data=np.column_stack((test_y,pred_test[:,thread_id])),columns=['test_target','test_predicted'])
    df.to_csv("data/test_data_cell_"+str(thread_id)+".csv",index=True)
    print("Cell"+str(thread_id)+" data saved as:test_data_cell_"+str(thread_id)+".csv")


    sess.close()
    

def main():

    start = time.time()

    # Load data
    train_dw, train_dv, test_dw, test_dv = load_data()
    threads = []
    
    for i in range(1,num_cells+1):
    	t = Thread(target=worker_thread, args=(i,train_dw, train_dv, test_dw, test_dv))
    	threads.append(t)

    for t in threads:
    	t.start()

    for t in threads:
    	t.join()

    end = time.time()
    print(time.strftime("%H:%M:%S",time.gmtime(end-start)))
  
    # View trained and tested data
    for i in range(1,num_cells+1):
        file_name = "data/train_data_cell_"+str(i)+".csv"
        df= pd.read_csv(file_name)
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        line1, = ax1.plot(df.train_target)
        line2, = ax1.plot(df.train_predicted)
        plt.title('Training Target Vs Predicted for Cell ' +str(i))
        file_name = 'img/Training_TargetVsPredicted for Cell ' +str(i)+ '.pdf'
        plt.savefig(file_name)
    
        file_name = "data/test_data_cell_"+str(i)+".csv"
        df= pd.read_csv(file_name)
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        line1, = ax1.plot(df.test_target)
        line2, = ax1.plot(df.test_predicted)
        plt.title('Test Target Vs Predicted for Cell ' +str(i))
        file_name = 'img/Testing_TargetVsPredicted for Cell ' +str(i)+ '.pdf'
        plt.savefig(file_name)
    
        file_name = "data/cost_cell_"+str(i)+".csv"
        df = pd.read_csv(file_name)
        plt.ion()
        plt.plot(df.epoch,df.cost)
        plt.axis([0,np.max(df.epoch),0,np.max(df.cost)])
        plt.title('Training Error Vs Epoch for Cell ' +str(i))
        file_name = 'img/Training Error Vs Epoch for Cell ' +str(i)+ '.pdf'
        plt.savefig(file_name)
 


if __name__ == '__main__':
    main()
