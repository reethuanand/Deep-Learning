import numpy as np

class NeuralNetwork(object):
    def __init__(self, nodes_input , nodes_hidden, nodes_output, learningrate):
        # Setting the number of input nodes, hidden ,output nodes of layer
        self.nodes_input = nodes_input
        self.nodes_hidden = nodes_hidden
        self.nodes_output = nodes_output

        # Initializing weights of hidden and output layers
        self.input_to_hidden_weights = np.random.normal(0.0, self.nodes_input**-0.5, 
                                       (self.nodes_input, self.nodes_hidden))
        self.weights_hidden_to_output = np.random.normal(0.0, self.nodes_hidden**-0.5, 
                                       (self.nodes_hidden, self.nodes_output))
        self.lr = learningrate
        
        ### TODO: Setting up self.activation_function to implement sigmoid type function ###
        # Note: defining the function in python with a lambda expression
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replacing zero with your sigmoid calculation.
        
        #
        #def sigmoid(x):
        #    return 0  # Replacing zero with your sigmoid value calculation 
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Training the network batch with features and targets. 
            Parameters
            ---------
            features define : 2 Dimensional array, every row is one data record, every column is a one feature
            targets define  : target values with a 1 Dimensional array structure
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.input_to_hidden_weights.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implementing the forwardpass function 
            # Implementing the backproagation function 
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implementing forwardpass here 
            parameters
            ---------
            X defines : features batch
        '''
        ### Implementing the forwardpass here ###
        ### Forwardpass ###
        # TODO: Hidden layer - Replacing these values with our calculations.
        hidden_inputs = np.dot(X, self.input_to_hidden_weights) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replacing these values with our calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer - output is continuous - f(x) = x
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implementing backpropagation
            Parameters
            ---------
            final_outputs: output of forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: changing in weights from input to hidden layers
            delta_weights_h_o: changing in weights from hidden to output layers
        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replacing this value with your calculations.
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        
        # TODO: Calculating the hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output, error)
        
        # TODO: Backpropagated error terms - Replacing these values with your calculations.
        output_error_term = error
        
        hidden_error_term = hidden_error * (hidden_outputs * (1 - hidden_outputs))
        # step of weight(input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None] 
        # step of weight (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None] 

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Updating weights on gradient descent steps
            Arguments
            ---------
            delta_weights_i_h: changing weights from input to hidden layers
            delta_weights_h_o: changing weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.input_to_hidden_weights += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Running a forward pass through the network with feature inputs
            Arguments
            ---------
            features: one dimensional array of feature values
        '''
        #### Implementing the forward pass :
        # TODO: replacing these values with the appropriate calculations - Hidden layer
        hidden_inputs = np.dot(features, self.input_to_hidden_weights) #  hidden layer into signals
        hidden_outputs = self.activation_function(hidden_inputs) 
        
        # TODO: replacing these values with the appropriate calculations - Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # final output layer into signals
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs

# Setting up hyperparameters for iterations
iterations = 2000
learningrate = 1
nodes_hidden = 10
nodes_output = 1
