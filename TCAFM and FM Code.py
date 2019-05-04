# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:49:04 2018

@author: FZL
"""

#this code is inspired from http://nowave.it/factorization-machines-with-tensorflow.html
# if you use this code, please mention the paper:
# Lahlou, Fatima Zahra, Houda Benbrahim, and Ismail Kassou. "Textual Context Aware Factorization Machines: Improving Recommendation by Leveraging Users' Reviews." Proceedings of the 2nd International Conference on Smart Digital Environment. ACM, 2018.

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

##################################### Example of how data should be
#x_data = np.matrix([
##    Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
##   A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
#    [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
#    [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
#    [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
#    [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
#    [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
#    [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
#    [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ]
#]).astype(np.float32)
#
## ratings
#y_data = np.array([5, 3, 1, 4, 5, 1, 5])


####################################################### Loading dataset

## importing data in the form of (user, item, rating)
# one hot encoding data, 
# returning data in the form of X,Y
def import_data_uir():
    
    ratingFile = open('Madison/yelp_MadisonRestaurantRatingUIR.csv', 'r')  
    ratings_data = pd.read_csv(ratingFile, header=None, names=['user','item','rating'],
                                  dtype={'user': 'str','item': 'str', 'rating': 'int32',
                                         })
    # One-hot encode rating data 
    transformed_ratings_data = pd.get_dummies(ratings_data) 
    transformed_ratings_data_without_rating = transformed_ratings_data.drop(['rating'], axis='columns')
    
	x_data = np.array(transformed_ratings_data_without_rating)
    x_data = np.nan_to_num(x_data)
    y_data = np.array(transformed_ratings_data['rating'].as_matrix())
    return x_data, y_data
	
## importing data in the form of (user, item, rating, contexts)
# one hot encoding data, 
# returning data in the form of X,Y
def import_data_uirc():
    ratingFile = open('Madison/MadisonRestaurantRatingsBag_of_centroidsK7Minwords31UIRC.csv', 'r')    
    ratings_data = pd.read_csv(ratingFile, header=None)
    
    # One-hot encode rating data 
    transformed_ratings_data = pd.get_dummies(ratings_data)
    transformed_ratings_data_without_rating = transformed_ratings_data.drop([2], axis='columns')
    
    x_data = np.array(transformed_ratings_data_without_rating)
    x_data = np.nan_to_num(x_data)
    y_data = np.array(transformed_ratings_data[2].as_matrix())
    return x_data, y_data

################################

x_data, y_data = import_data_uir()
n_usersAndItem = 17773 

##########################################

# Let's add an axis to make tensoflow happy.
y_data.shape += (1, )

def create_training_test_sets(x_data,y_data,test_size = 0.2):
   testing_size = len(x_data) - int(test_size*len(x_data))
   #we  shuffle the data first
   shuffled_x_data, shuffled_y_data = shuffle(x_data, y_data)
   train_x = shuffled_x_data[:testing_size,:] 
   train_y = shuffled_y_data[:testing_size,:] 
   test_x = shuffled_x_data[testing_size:,:]
   test_y = shuffled_y_data[testing_size:,:]
   return train_x,train_y,test_x,test_y

train_x,train_y,test_x,test_y = create_training_test_sets(x_data,y_data)

# We use Placeholders for the inputs and targets. 
# The actual data will be assigned at run time in the Session. 
# X and y won't be further modified by the backend;
# we use Variables to hold bias, weights and factor layers. These are the parameters that will be updated when fitting the model.
n, p = x_data.shape
# design matrix
X = tf.placeholder(shape=[None, p], dtype=tf.float32) #fzl: il est plus logique que X soit [1,p] ou [none, p]: Placeholders do not need to be statically sized. Now, when we define the values of x in the feed_dict we can have any number of values.
# target vector
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# bias and weights
w0 = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros([p]))
# number of latent factors
k = 200 
learningRate=0.1   
# interaction factors, randomly initialized 
V = tf.Variable(tf.random_normal([k, p], stddev=0.1))
    
#FZL: for TCAFM model     
#FZL: we apply fm_model on a matrix X' = X*Beta, where Beta(i) = 1 for users and items, and Beta(i) = a value < 1 for the others 
#FZL: beta(i) indicate the weights of contextual features
#FZL: Beta is a diagonal matrix of shape [p, p] (there are p of xi)
#FZL: beta est a parameter to learn when values are not 1

#FZL: we initialize beta(i) to  0
dim_beta_1 = n_usersAndItem
dim_beta_weight =p - dim_beta_1
#diago_beta_weight = tf.Variable(tf.zeros([dim_beta_weight]))
diago_beta_weight = tf.random_normal([dim_beta_weight], stddev=0.1)
diago_beta1 = tf.ones([dim_beta_1])
diago_Beta = tf.concat([diago_beta_weight,diago_beta1],0 ) #FZL: User and items are at the end
#Beta = tf.diag(diago_Beta)
Beta = tf.Variable(tf.diag(diago_Beta))

def fm_model(data):
    
    #In the following code we compute WX 
    #and use reduce_sum() to add together the row elements of the resulting Tensor (axis 1). 
    #keep_dims is set to True to ensure that input/output dimensions are respected.
    
    linear_terms = tf.add(w0,
                          tf.reduce_sum(
                                  tf.multiply(W, X), 1, keepdims=True))    
    
    # We do the same for the interaction terms.
    interactions = (tf.multiply(0.5,
                                tf.reduce_sum(
                                        tf.subtract(
                                                tf.pow( tf.matmul(X, tf.transpose(V)), 2), 
                                                tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))), 
                                        1, keepdims=True)))
                                        
    #fzl: tf.reduce_sum -> Computes the sum of elements across dimensions of a tensor.
    #fzl: tf.pow -> Computes the power of one value to another
    #fzl: matmul -> multipliction de matrices
    #fzl: tf.subtract(x,y)  -> Returns x - y element-wise
    #FZL: I have verified computings, they are all corrects 
    
    # And add everything together to obtain the target estimate.EP
    prediction = tf.add(linear_terms, interactions)
    
    return prediction

def tcafm_model(data):    
    X_prime = tf.matmul(X , Beta, a_is_sparse=True, b_is_sparse=True)    
    prediction = fm_model(X_prime)     
    return prediction

def train_fm_model(x):
    with open("ResultsFileName.txt", 'w') as f:
        prediction = fm_model(x)
        # L2 regularized sum of squares loss function over W and V
        lambda_w = tf.constant(0.001, name='lambda_w')
        lambda_v = tf.constant(0.001, name='lambda_v')
        
        l2_norm = (tf.reduce_sum(
                    tf.add(
                        tf.multiply(lambda_w, tf.pow(W, 2)),
                        tf.multiply(lambda_v, tf.pow(V, 2)))))
        error = tf.reduce_mean(tf.square(tf.subtract(y, prediction)))
        error_for_epoch_mse_computing = tf.reduce_sum(tf.square(tf.subtract(y, prediction)))
        loss = tf.add(error, l2_norm)
        # To train the model we instantiate an Optimizer object and minimize the loss function.
        optimizer = tf.train.AdagradOptimizer(learning_rate=learningRate).minimize(loss)
        
        # We are ready to compile the graph, and launch it on the Tensorflow backend.
        N_EPOCHS = 1000 #150    
        BATCH_SIZE =1000
        
        # Launch the graph.
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(N_EPOCHS):
                # We shuffle data to avoid biasing the gradient.
                shuffled_train_x, shuffled_train_y = shuffle(train_x, train_y)
                epoch_error = 0
                i=0
                while i < len(shuffled_train_x):
                   start = i
                   end = i + BATCH_SIZE
                   batch_x = np.array(shuffled_train_x[start:end])
                   batch_y = np.array(shuffled_train_y[start:end])
                   _, c = sess.run([optimizer, error_for_epoch_mse_computing], feed_dict={x: batch_x, y: batch_y})
                   epoch_error += c
                   i+=BATCH_SIZE
    
                epoch_error = epoch_error / train_x.shape[0]  #FZL: for computing the total MSE
                print('Epoch', epoch+1, 'completed out of',N_EPOCHS,'loss:',epoch_error,file=f)
                test_mse = sess.run(error, feed_dict={X: test_x, y: test_y})
                print('Test MSE: ', test_mse,file=f)
               
            
            print('MSE: ', sess.run(error, feed_dict={X: test_x, y: test_y}),file=f)            
            
            print('Loss (regularized error):', sess.run(loss, feed_dict={X: test_x, y: test_y}),file=f)
            
        print("End of FM with Hyperparametres: k: 7 learning rate: 0.1 lambda_w: 0.001 lambda_V: 0.001",file=f)   
        print('al hamdou li LLAH rabbi al 3alamin',file=f)
        print("-------------------------------------------------------",file=f)
    
def train_tcafm_model(x):
    with open("ResultsFileName.txt", 'w') as f:
        prediction = tcafm_model(x)
        # L2 regularized sum of squares loss function over W and V and beta
        lambda_w = tf.constant(0.001, name='lambda_w')
        lambda_v = tf.constant(0.001, name='lambda_v')
        lambda_beta = tf.constant(0.3, name='lambda_beta')
        l2_norm = (tf.add(
                       tf.reduce_sum(
                               tf.add(
                                   tf.multiply(lambda_w, tf.pow(W, 2)),
                                   tf.multiply(lambda_v, tf.pow(V, 2)),)),
                       tf.reduce_sum(tf.multiply(lambda_beta, tf.pow(Beta, 2)))))
        error = tf.reduce_mean(tf.square(tf.subtract(y, prediction)))
        error_for_epoch_mse_computing = tf.reduce_sum(tf.square(tf.subtract(y, prediction)))
        loss = tf.add(error, l2_norm)
        # To train the model we instantiate an Optimizer object and minimize the loss function.
        optimizer = tf.train.AdagradOptimizer(learning_rate=learningRate).minimize(loss)
        
        # We are ready to compile the graph, and launch it on the Tensorflow backend.
        N_EPOCHS = 1000    
        BATCH_SIZE =1000
        
        # Launch the graph.
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(N_EPOCHS):
                # We shuffle data to avoid biasing the gradient.
                shuffled_train_x, shuffled_train_y = shuffle(train_x, train_y)
                epoch_error = 0
                i=0
                while i < len(shuffled_train_x):
                   start = i
                   end = i + BATCH_SIZE
                   batch_x = np.array(shuffled_train_x[start:end])
                   batch_y = np.array(shuffled_train_y[start:end])
                   _, c = sess.run([optimizer, error_for_epoch_mse_computing], feed_dict={x: batch_x,
                                                                      y: batch_y})
                   epoch_error += c
                   i+=BATCH_SIZE
                epoch_error = epoch_error / train_x.shape[0]  #FZL: for computing the total MSE
                print('Epoch', epoch+1, 'completed out of',N_EPOCHS,'loss:',epoch_error,file=f)
                print('Test MSE: ', sess.run(error, feed_dict={X: test_x, y: test_y}),file=f)
              
            print('MSE: ', sess.run(error, feed_dict={X: test_x, y: test_y}),file=f)
            print('Loss (regularized error):', sess.run(loss, feed_dict={X: test_x, y: test_y}),file=f)
            print("Fin des tests TCAFM avec comme parametres: k: 7 learning rate: 0.1 lambda_w: 0.001 lambda_V: 0.001 lambda_beta:0.3",file=f)   
            print('al hamdou li LLAH rabbi al 3alamin',file=f)
            print("-------------------------------------------------------",file=f)
            print("-------------------------------------------------------",file=f)            
            print('Predictions:', sess.run(prediction, feed_dict={X: x_data, y: y_data}),file=f)
            print("-------------------------------------------------------",file=f)            
            print('Learnt weights:', sess.run(W, feed_dict={X: test_x, y: test_y}),file=f)
            print("-------------------------------------------------------",file=f)            
            print('Learnt factors:', sess.run(V, feed_dict={X: test_x, y: test_y}),file=f)
            print("-------------------------------------------------------",file=f)            
            print('Learnt Beta:', sess.run(Beta, feed_dict={X: test_x, y: test_y}),file=f)
            print("Fin des tests TCAFM avec comme parametres: k: 7 learning rate: 0.1 lambda_w: 0.001 lambda_V: 0.001 lambda_beta:0.3",file=f)   
            print('al hamdou li LLAH rabbi al 3alamin',file=f)
            print("-------------------------------------------------------",file=f)
         
train_fm_model(X)

