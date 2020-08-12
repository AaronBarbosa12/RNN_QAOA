import cirq
import networkx as nx
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
from MaxCut_QAOA_Graphs import MaxCut_QAOA_Graphs
from QRNN_Model import QRNN_Model
import datetime
import os 

################################
# GENERATE INPUT DATA
################################

# Create a bunch of QAOA graph circuits at a fixed level of p
training_size = 2000
batch_size = 500
num_neurons = 20
epochs = 100

qaoa = MaxCut_QAOA_Graphs(data_path = '/home/aaron/QAOA/RNN/')
circuits, symbols, cost_hams, training_set_ids = qaoa.generate_ER_graphs(batch_size = training_size,
                                                                        size_range = [8,12],
                                                                        p = 1)
qaoa.visualize_graphs()

################################
# Randomly initialize parameters
#######################################

initial_hidden_state = np.zeros(shape = (training_size,num_neurons)).astype(np.float32)
initial_params = np.random.normal(size=(training_size,2)).astype(np.float32)

expectation_op = tfq.get_expectation_op() 

initial_exp = expectation_op(circuits, [str(element) for element in symbols] , initial_params, cost_hams)

data_raw = [circuits, cost_hams, initial_hidden_state,initial_params,initial_exp]

#########################
# Create MODEL
#########################

model = QRNN_Model(symbols, num_neurons = num_neurons)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Log progress with Tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = qaoa.data_path + 'logs/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

######################
# TRAIN MODEL
#########################

import math 

def chunk(lst,batch_size):
    circuits = lst[0]
    cost_hams = lst[1]
    initial_hidden_state = lst[2]
    initial_params = lst[3]
    initial_exp = lst[4]

    if len(circuits) < batch_size:
        print("Error: Batch size larger than number of samples!")
        return

    n_chunks = math.floor(len(circuits)/batch_size)
    left_over = len(circuits) % batch_size

    """Yield successive n-sized chunks from lst."""
    batches = []
    
    for i in range(0,n_chunks):
        #print(batch_size*i,batch_size*(i+1))
        batches.append([circuits[batch_size*i:batch_size*(i+1)], 
                        cost_hams[batch_size*i:batch_size*(i+1)], 
                        initial_hidden_state[batch_size*i:batch_size*(i+1)],
                        initial_params[batch_size*i:batch_size*(i+1)],
                        initial_exp[batch_size*i:batch_size*(i+1)]])
        
    if left_over != 0:
        #print(batch_size*(i+1),batch_size*(i+1)+left_over)
        batches.append([circuits[batch_size*i:batch_size*i+left_over], 
                        cost_hams[batch_size*i:batch_size*i+left_over], 
                        initial_hidden_state[batch_size*i:batch_size*i+left_over],
                        initial_params[batch_size*i:batch_size*i+left_over],
                        initial_exp[batch_size*i:batch_size*i+left_over]])
    return batches

x = chunk(data_raw,batch_size) # Separate data into training batches

# Iterate over epochs.
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for batch_num, training_batch in enumerate(x):
        with tf.GradientTape() as tape:
            out = model(training_batch)
            
            # Compute loss
            loss = tf.reduce_mean(out[5])  

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss,step=epoch)
            '''
            tf.summary.histogram('Kernel - shared state cell 0', model.rnn_0.shared.kernel ,step=epoch)
            tf.summary.histogram('Kernel - hidden state cell 0', model.rnn_0.state.kernel ,step=epoch)

            tf.summary.histogram('Kernel - shared state cell 1', model.rnn_0.shared.kernel ,step=epoch)
            tf.summary.histogram('Kernel - hidden state cell 1', model.rnn_0.state.kernel ,step=epoch)

            tf.summary.histogram('Kernel - shared state cell 2', model.rnn_2.shared.kernel ,step=epoch)
            tf.summary.histogram('Kernel - hidden state cell 2', model.rnn_2.state.kernel ,step=epoch)
    
            tf.summary.histogram('Kernel - shared state cell 3', model.rnn_3.shared.kernel ,step=epoch)
            tf.summary.histogram('Kernel - hidden state cell 3', model.rnn_3.state.kernel ,step=epoch)

            tf.summary.histogram('Kernel - shared state cell 4', model.rnn_4.shared.kernel ,step=epoch)
            tf.summary.histogram('Kernel - hidden state cell 4', model.rnn_4.state.kernel ,step=epoch)
            '''

    print('epoch %s: mean loss = %s' % (epoch, loss))

####################
## SAVE MODEL 
####################

model_path = qaoa.data_path +"models/"
model_file_path = model_path+"QRNN.h5"

if not os.path.exists(model_path):
    os.mkdir(model_path)

model.save_weights(model_file_path)
print("Model saved to: " + model_file_path)