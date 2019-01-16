'''
Created on 2019年1月14日

@author: adp
'''
import  tensorflow  as tf #A
sess  = tf.InteractiveSession() #A
raw_data  = [ 1. ,  2. ,  8. ,  -1. ,  0. ,  5.5 ,  6. ,  13 ] #B
spikes  = tf.Variable([False]  * len (raw_data), name='spikes') #C
print([123] * len(raw_data))
print([False]  * len (raw_data))
spikes.initializer.run() #D
saver  = tf.train.Saver() #E
for  i  in range ( 1 ,  len (raw_data)): #F
    if  raw_data[i]  - raw_data[i- 1 ]  > 5 : #F
            spikes_val  = spikes.eval() #G
            spikes_val[i]  = True #G
            print(spikes_val)
            updater  = tf.assign(spikes, spikes_val) #G
            updater.eval() #H
save_path  = saver.save(sess, "./spikes.ckpt") #I
print (" spikes data saved in file: %s "  % save_path)