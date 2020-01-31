import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from  sklearn.utils import shuffle
from sklearn.preprocessing  import LabelEncoder
from sklearn.model_selection import train_test_split
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#from __future__ import absolute_import, division, print_function, unicode_literals
# try:
#     import tensorflow.compat.v1 as tf
#     print(tf.__version__)
# except Exception:
#   pass
#   tf.disable_v2_behavior()


def read_dataset():
    df = pd.read_csv("sonar.csv")
    print("Dataset loaded successfully")
    #print("Number of coloumns:",len(df.columns))

    X = df[df.columns[0:5]].values
    #print(X)

    y = df[df.columns[60]]
    #print(y)

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    print("Ans:",y)
    Y = one_hot_encode(y)
    #print(Y)
    print("Value of X.shape",X.shape)
    return (X,Y)
    
def one_hot_encode(labels):
    n_labels = len(labels)
    print(n_labels)
    n_unique_labels=len(np.unique(labels))
    print(n_unique_labels)
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    #print(one_hot_encode)
    one_hot_encode[np.arange(n_labels),labels] = 1
    #print(one_hot_encode)
    return one_hot_encode
        
   
def main():
   X, Y= read_dataset()
   #print(X,Y)
   X, Y = shuffle(X,Y,random_state=1)
   print(X,Y)

   train_x ,test_x, train_y, test_y = train_test_split(X,Y,test_size=0.30,random_state=415)
   print("train_x",train_x.shape)
   print("test_x",test_x.shape)
   print("train_y",train_y.shape)
   print("test_y",test_y.shape)

   learning_rate=0.3
   training_epochs=1000
   cost_history=np.empty(shape=[1],dtype=float)
   n_dim = X.shape[1]
   print("Number of columns are n_dim",n_dim)

   n_class = 2
   model_path="Marvellous"
   
   n_hidden_1 = 60
   n_hidden_2 = 60
   n_hidden_3 = 60
   n_hidden_4 = 60


   x  = tf.compat.v1.placeholder(tf.float32,[None,n_dim])
   y_ = tf.compat.v1.placeholder(tf.float32,[None,n_class])

   W = tf.Variable(tf.zeros([n_dim,n_class]))
   b= tf.Variable(tf.zeros([n_class]))

   weights = { 
       'h1':tf.Variable(tf.random.truncated_normal([n_dim,n_hidden_1])),
       'h2':tf.Variable(tf.random.truncated_normal([n_hidden_1,n_hidden_2])),
       'h3':tf.Variable(tf.random.truncated_normal([n_hidden_2,n_hidden_3])),
       'h4':tf.Variable(tf.random.truncated_normal([n_hidden_3,n_hidden_4])),
       'out':tf.Variable(tf.random.truncated_normal([n_hidden_4,n_class]))
   }

   biases = {
       'b1':tf.Variable(tf.random.truncated_normal([n_hidden_1])),
       'b2':tf.Variable(tf.random.truncated_normal([n_hidden_2])),
       'b3':tf.Variable(tf.random.truncated_normal([n_hidden_3])),
       'b4':tf.Variable(tf.random.truncated_normal([n_hidden_4])),
       'out':tf.Variable(tf.random.truncated_normal([n_class])),
   }

   cost = sess.run(cost_function,feed_dict={x:train_x,y_:train_y})
   cost_history = np.append(cost_history,cost)
   correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
   pred_y = sess.run(y,feed_dict={x:test})
   mse = tf.reduce_mean(tf.square(pred_y-test_y))
   mse_=sess.run(mse)
   accuracy=(sess.run(accuracy,feed_dict={x:train_x,y_:train_y}))
   accuracy_history.append(accuracy)
   print('epoch:','epoch'-',cost',cost,"-MSE:",mse_,"-Train Accuracy",accuracy)

   save_path=saver.save(sess,model_path)
   print("model saved in file:%s",save_path)

   plt.plot(accuracy_history)
   plt.title("Accuracy history")
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.show()

   plt.plot(range(len(cost_history)),cost_history)
   plt.title("Loss calculation")
   plt.axis([0,training_epochs,0,np,max(cost_history)/100])
   plt.xlabel(['Epochs'])
   plt.ylabel(['Loss'])
   plt.show()

   correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
   accuracy=tf.reduce_mean(tf.square(pred_y-test_y))
   print("Test Accuracy:",(sess.run(y,feed_dict={x:test_x,y_:test_y})))

   pred_y = sess.run(y,feed_dict={x:test_x})
   mse = tf.reduce_mean(tf.square(pred_y-test_y))
   print("Mean Sqaure error:%.4f" %sess.run(mse))
if __name__ == "__main__":
   main()
