# Handling Networks
import networkx as nx

# Math
import numpy as np
import numpy.linalg
from random import sample 
import random
import math 

# Simulations
import cirq
from cirq.circuits import InsertStrategy
import tensorflow_quantum as tfq
import tensorflow as tf

import sympy

import pylab
import picos as pic
from picos.tools import diag_vect
import cvxopt as cvx
import cvxopt.lapack

#Visualization
import matplotlib.pyplot as plt

#Logging
from datetime import datetime
import time
import os
import pandas as pd 
import uuid 

class MaxCut_QAOA_Graphs:

    def __init__(self, data_path):
        time  = datetime.now() #time that this object was created
        self.current_time = time.strftime("%d_%m_%Y___%H_%M_%S")

        self.data_path = data_path + str(self.current_time)+ '/' #path to where all data is stored

        self.graph_desc_path = self.data_path + 'graph_desc' + '.csv' #path to where graph descriptions are stored

        self.graph_file_path = self.data_path + 'graphs/' #path to where graph themselves are stored

        self.image_path = self.data_path + 'images/'

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        if not os.path.exists(self.graph_file_path):
            os.mkdir(self.graph_file_path)

        if not os.path.exists(self.image_path):
            os.mkdir(self.image_path)
    
    def set_data_folders(self, data_path):
        self.data_path = data_path

        self.graph_desc_path = self.data_path + 'graph_desc' + '.csv' #path to where graph descriptions are stored

        self.graph_file_path = self.data_path + 'graphs/' #path to where graph themselves are stored

        self.image_path = self.data_path + 'images/'

    def load_graphs(self):
        all_graph_files = os.listdir(self.graph_file_path)
        all_graph_ids = [os.path.splitext(element)[0] for element in all_graph_files]
        return fetch_graphs_from_ID(all_graph_ids)

    def generate_ER_graphs(self,batch_size,size_range,p):
        min_num_nodes = size_range[0]
        max_num_nodes = size_range[1]

        all_graphs = []

        print("Generating dataset...")
    
        for i in range(batch_size):
            if i % 50 == 0:
                print("Generating graph " + str(i))

            n = random.randint(min_num_nodes,max_num_nodes)
            
            # Generate Erdos-Renyi Graphs
            current_graph = nx.erdos_renyi_graph(n, 0.5, seed=None, directed=False)

            while nx.is_connected(current_graph) == False:
                current_graph = nx.erdos_renyi_graph(n, 0.5, seed=None, directed=False)
            
            all_graphs.append(current_graph)

        print("Creating graph database...")
        
        graph_ids = self.save_graphs_to_csv(all_graphs)

        self.graph_ids = graph_ids

        print("Finished generating dataset")

        return self.generate_circuit_batch(graph_ids,p)
    
    def save_graphs_to_csv(self,graphs):
        ########################################################################
        # RANDOMLY GENERATES BATCH OF ERDOS-RENYI GRAPHS WITH 10-20 NODES
        '''
        INPUTS:
            batch_size - number of graphs to generate

        OUTPUTS:
            graph_ids - list of uniquely generated IDs for each graph
                        ** graphs are stored as dataframes in: .../*SELF.DATA_PATH*'/graphs/*GRAPH ID*.csv **
                        ** graph descriptions are stored as a dataframe in: .../*SELF.DATA_PATH*'/graphs/graph_desc.csv **

                        "GRAPH_ID": Uniquely generated ID for graph
                        "GW_CUT": Maximim cut predicted by the Geomanns-Williamson algorithm 
                        "NUM_NODES": Number of nodes 
                        "NUM_EDGES": Number of edges 
                        "LOGRATIO_EDGETONODES": logarithm of the ratio of number of edges to the number of nodes 
                        "DENSITY": Density of graph
                        "LOGRATIO_GWTOEDGES": logarithm of the ratio of Geomanns-Williamson cut to the number of edges
                        "LOGRATIO_GWTONODES": logarithm of the ratio of Geomanns-Williamson cut to the number of nodes
                        "SPECTRAL_GAP": difference between first and second largest eigenvalue
                        "LOG_LARGESTEIGVAL": logarithm of the ratio of the largest eigenvalue
                        "LOG_SECONDLARGESTEIGVAL": logarithm of the ratio of the second largest eigenvalue
                        "LOG_THIRDLARGESTEIGVAL": logarithm of the ratio of the third largest eigenvalue
                        "LOG_FOURTHLARGESTEIGVAL": logarithm of the ratio of the fourth largest eigenvalue
                        "LOG_FIFTHLARGESTEIGVAL": logarithm of the ratio of the fifth largest eigenvalue
                        "LOG_SIXTHLARGESTEIGVAL": logarithm of the ratio of the sixth largest eigenvalue
                        "LOG_SMALLESTEIGVAL": logarithm of the ratio of the smallest eigenvalue
                        "MIN_ECC": smallest eccentricity of graph
                        "MAX_ECC": largest eccentricity of graph
                        "LOGRATIO_MAXMINECC: logarithm of the ratio of maximum eccentricity of graph to minimum eccentricity"

        '''
        ########################################################################

        graph_ids = [0]*len(graphs)

        graph_description = pd.DataFrame(columns = ["GRAPH_ID","GW_CUT","NUM_NODES","NUM_EDGES","LOGRATIO_EDGETONODES","DENSITY",
                                                    "LOGRATIO_GWTOEDGES","LOGRATIO_GWTONODES","SPECTRAL_GAP","LOG_LARGESTEIGVAL",
                                                    "LOG_SECONDLARGESTEIGVAL","LOG_SMALLESTEIGVAL","MIN_ECC","MAX_ECC","LOGRATIO_MAXMINECC"])

        for i,current_graph in enumerate(graphs):

           current_cut = self.get_GW_cut(current_graph)
           current_graph_num_nodes = current_graph.number_of_nodes()
           current_graph_num_edges = current_graph.number_of_edges()
           current_graph_density = nx.density(current_graph) 

           current_graph_logratio_edgestonodes = np.log(current_graph_num_edges/current_graph_num_nodes)
           current_graph_ratio_GWtoedges = np.log(current_cut/current_graph_num_edges)
           current_graph_ratio_GWtonodes = np.log(current_cut/current_graph_num_nodes)

           L = nx.normalized_laplacian_matrix(current_graph)
           e = numpy.linalg.eigvals(L.A)
           e.sort()

           current_graph_spectral_gap = abs(max(e) - e[len(e)-2])

           current_graph_loglargesteigenval = np.log(e[len(e)-1])
           current_graph_logsecondlargesteigenval = np.log(e[len(e)-2])
           current_graph_logthirdlargesteigenval = np.log(e[len(e)-3])
           current_graph_logfourthlargesteigenval = np.log(e[len(e)-4])
           current_graph_logfifthlargesteigenval = np.log(e[len(e)-5])
           current_graph_logsixthlargesteigenval = np.log(e[len(e)-6])
           current_graph_logsmallesteigenval = np.log(e[1]) #Smallest NONTRIVIAL eigenvalue
            
           eccs = list(nx.eccentricity(current_graph).values())
           eccs.sort()
           current_graph_min_eccentricity = eccs[1]
           current_graph_max_eccentricity = eccs[len(e)-1]
           current_graph_ratio_maxmin_eccentricity = np.log(current_graph_max_eccentricity/current_graph_min_eccentricity)
    

           graph_ids[i] = str(uuid.uuid1())
           
           graph_df = nx.to_pandas_edgelist(current_graph, dtype=int)
           graph_description = graph_description.append({"GRAPH_ID":graph_ids[i],
                                                         "GW_CUT":current_cut,
                                                         "NUM_NODES":current_graph_num_nodes,
                                                         "NUM_EDGES":current_graph_num_edges,
                                                         "LOGRATIO_EDGETONODES":current_graph_logratio_edgestonodes,
                                                         "DENSITY":current_graph_density,
                                                         "LOGRATIO_GWTOEDGES":current_graph_ratio_GWtoedges,
                                                         "LOGRATIO_GWTONODES":current_graph_ratio_GWtonodes,
                                                         "SPECTRAL_GAP":current_graph_spectral_gap,
                                                         "LOG_LARGESTEIGVAL":current_graph_loglargesteigenval,
                                                         "LOG_SECONDLARGESTEIGVAL":current_graph_logsecondlargesteigenval,
                                                         "LOG_THIRDLARGESTEIGVAL":current_graph_logthirdlargesteigenval,
                                                         "LOG_FOURTHLARGESTEIGVAL":current_graph_logfourthlargesteigenval,
                                                         "LOG_FIFTHLARGESTEIGVAL":current_graph_logfifthlargesteigenval,
                                                         "LOG_SIXTHLARGESTEIGVAL":current_graph_logsixthlargesteigenval,
                                                         "LOG_SMALLESTEIGVAL":current_graph_logsmallesteigenval,
                                                         "MIN_ECC":current_graph_min_eccentricity,
                                                         "MAX_ECC":current_graph_max_eccentricity,
                                                         "LOGRATIO_MAXMINECC":current_graph_ratio_maxmin_eccentricity}, ignore_index = True) 
                                             
           current_graph_file_path = self.graph_file_path + str(graph_ids[i]) + '.csv'
           graph_df.to_csv(current_graph_file_path, index=True)   
        
        #graph_description.to_csv(self.graph_desc_path, mode= 'a', index=False, header=False)
        with open(self.graph_desc_path, 'a') as f:
            graph_description.to_csv(f, mode='a', index=False, header=f.tell()==0)

        return graph_ids        

    def fetch_graphs_from_ID(self,ids):
        ########################################################################
        # Returns graphs given IDs
        '''
        INPUTS:
            ids - list containing IDs of graphs to be fetched

        OUTPUTS:
            all_graphs - list of graphs specified by ids
        '''
        ########################################################################
        all_graphs = []
        gw_maxcuts = [0]*len(all_graphs)

        for index,current_id in enumerate(ids):
           current_graph_file_path = self.graph_file_path + str(current_id) + '.csv'
           current_graph_df = pd.read_csv(filepath_or_buffer = current_graph_file_path)           
           current_graph = nx.from_pandas_edgelist(current_graph_df, 'source', 'target', ['weight'])
           all_graphs.append(current_graph)

        return all_graphs

    def get_GW_cut(self,graph):
        ########################################################################
        # RETURNS AVERAGE GEOMANNS WILLIAMSON CUT FOR A GIVEN GRAPH
        ########################################################################

        G = graph
        N = len(G.nodes())

        # Allocate weights to the edges.
        for (i,j) in G.edges():
            G[i][j]['weight']=1.0

        maxcut = pic.Problem()

        # Add the symmetric matrix variable.
        X=maxcut.add_variable('X',(N,N),'symmetric')

        # Retrieve the Laplacian of the graph.
        LL = 1/4.*nx.laplacian_matrix(G).todense()
        L=pic.new_param('L',LL)

        # Constrain X to have ones on the diagonal.
        maxcut.add_constraint(pic.tools.diag_vect(X)==1)

        # Constrain X to be positive semidefinite.
        maxcut.add_constraint(X>>0)

        # Set the objective.
        maxcut.set_objective('max',L|X)

        #print(maxcut)

        # Solve the problem.
        maxcut.solve(verbose = 0,solver='cvxopt')

        # Use a fixed RNG seed so the result is reproducable.
        cvx.setseed(1)

        # Perform a Cholesky factorization.
        V=X.value
        cvxopt.lapack.potrf(V)
        for i in range(N):
            for j in range(i+1,N):
                V[i,j]=0

        # Do up to 100 projections. Stop if we are within a factor 0.878 of the SDP
        # optimal value.
        count=0
        obj_sdp=maxcut.obj_value()
        obj=0
        while (count < 100 or obj < 0.878*obj_sdp):
            r=cvx.normal(N,1)
            x=cvx.matrix(np.sign(V*r))
            o=(x.T*L*x).value
            if o > obj:
                x_cut=x
                obj=o
            count+=1
        x=x_cut

        # Extract the cut and the seperated node sets.
        S1=[n for n in range(N) if x[n]<0]
        S2=[n for n in range(N) if x[n]>0]
        cut = [(i,j) for (i,j) in G.edges() if x[i]*x[j]<0]
        leave = [e for e in G.edges if e not in cut]

        # Show the relaxation optimum value and the cut capacity.
        rval = maxcut.obj_value()
        sval = sum(G[e[0]][e[1]]['weight'] for e in cut)

        return sval 

    def qaoa_circuit_from_graph(self,graph,p):
        cirq_qubits = cirq.GridQubit.rect(1,graph.number_of_nodes())
        qaoa_circuit = cirq.Circuit()
        # Create Mixer ground state
        for qubit_index,qubit in enumerate(cirq_qubits):
            qaoa_circuit.append([cirq.H(qubit)], strategy=InsertStrategy.EARLIEST)

        qaoa_parameters = []
        for step in range(1,p+1):
            gamma_i = sympy.Symbol("gamma{}_p={}".format(step,p))
            beta_i = sympy.Symbol("beta{}_p={}".format(step,p))
            qaoa_parameters.append(gamma_i)
            qaoa_parameters.append(beta_i)

            #Apply ising hamiltonian
            for current_edge in graph.edges():
                qubit1 = cirq_qubits[current_edge[0]]
                qubit2 = cirq_qubits[current_edge[1]]

                qaoa_circuit.append([cirq.CNOT(qubit1,qubit2)], strategy=InsertStrategy.EARLIEST)
                qaoa_circuit.append([cirq.Rz(-1*gamma_i)(qubit2)], strategy=InsertStrategy.EARLIEST)
                qaoa_circuit.append([cirq.CNOT(qubit1,qubit2)], strategy=InsertStrategy.EARLIEST)
                
            #Apply Driver Hamiltonian 
            for current_node in graph.nodes():
                qubit = cirq_qubits[current_node]
                qaoa_circuit.append([cirq.Rx(beta_i)(qubit)], strategy=InsertStrategy.EARLIEST)

            #Generate Cost Hamiltionian
            cost_ham = None
            for current_edge in graph.edges():
                qubit1 = cirq_qubits[current_edge[0]]
                qubit2 = cirq_qubits[current_edge[1]]

                if cost_ham is None:
                    cost_ham = -1/2*cirq.Z(qubit1)*cirq.Z(qubit2) + 1/2
                else:
                    cost_ham += -1/2*cirq.Z(qubit1)*cirq.Z(qubit2) +1/2
                
        return qaoa_circuit, qaoa_parameters, cost_ham

    def generate_circuit_batch(self,graph_ids,p):
        '''
        #randomly generate graphs and their descriptions
        '''
        graphs = self.fetch_graphs_from_ID(graph_ids) 

        qaoa_circuits = []
        cost_hams = []

        for graph_index,graph in enumerate(graphs):
            qaoa_circuit, qaoa_parameters, cost_ham = self.qaoa_circuit_from_graph(graph,p)
            qaoa_circuits.append(qaoa_circuit)
            cost_hams.append(cost_ham)

        qaoa_circuits = tfq.convert_to_tensor(qaoa_circuits)

        cost_hams = tfq.convert_to_tensor([cost_hams])
        cost_hams = tf.transpose(cost_hams)

        return qaoa_circuits, qaoa_parameters, cost_hams, graph_ids

    def visualize_graphs(self):
        ########################################################################
        # Generates images describing the dataset
        '''
        Output is saved as a .png file in images folder
        Function for debugging purposes
        '''
        ########################################################################

        # Randomly sample dataset
        random_sample_ids = sample(self.graph_ids,20)
        random_sample_graphs = self.fetch_graphs_from_ID(random_sample_ids) 

        # Visualize randomly sampled graphs
        fig, ax = plt.subplots(nrows=5, ncols=4)
        ax = ax.flatten()

        fig.suptitle('Random Sample - '+str(self.current_time), fontsize=16)

        for i in range(20):
            nx.draw_networkx(random_sample_graphs[i], ax=ax[i],with_labels=False,node_size=20)
            ax[i].set_axis_off()

        image_path = self.data_path + 'images/sample_'+ str(self.current_time) + '.png'
        plt.savefig(image_path,dpi=1000)

        #Create histogram describing the entire dataset
        fig, ax = plt.subplots(nrows=1, ncols=1)

        graph_desc = pd.read_csv(filepath_or_buffer = self.graph_desc_path)           
        
        counts =  graph_desc["NUM_NODES"].value_counts(ascending=True)

        fig.suptitle('Statistics of Entire Dataset '+str(self.current_time), fontsize=16)
        plt.bar(counts.index.tolist(),counts.values)

        image_path = self.image_path + '/desc_'+ str(self.current_time) + '.png'
        plt.savefig(image_path,dpi=1000)

        plt.close