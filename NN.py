# -*- coding: utf-8 -*-
"""
Created on Mon Jun 06 11:36:31 2016

@author: Br3A
"""
import numpy, math

class Neuron:
    def __init__(self, net = 0, out = 0):
        self.net = net     # total net input to the neuron
        self.out = out     # output of neuron after passing input through activation function
    
    def activate(self):
        self.out = ( 1 / ( 1 + math.exp(-(self.net)) ) )
        

class Layer:
    def __init__(self, weightMatrixShape, numOfNeurons):
        self.parameters = numpy.random.random_sample(weightMatrixShape)
        self.newParameters = numpy.zeros(self.parameters.shape)
        self.numOfNeurons = numOfNeurons
        self.delta = numpy.zeros(numOfNeurons)
        self.neurons = []
        for i in xrange(self.numOfNeurons):
            self.neurons.append(Neuron())
    
    # To make a vector of outputs of all neurons in the layer    
    def compute_output(self):
        self.out = []
        for neuron in self.neurons:
            self.out.append(neuron.out)
        
        self.out = numpy.array(self.out)
            
        
        
        
        
class NeuralNetwork:
    def __init__(self, numOfLayers, numOfNeuronsPerLayer):
        self.numOfLayers = numOfLayers
        self.numOfNeuronsPerLayer = numOfNeuronsPerLayer
        self.error = 1
        self.etta = 0.5
        self.layers = []
        
        a, b = 0, 2
        if type(self.numOfNeuronsPerLayer) == tuple:    # In case when every layer has different number of neurons
            for i in xrange(self.numOfLayers - 1):      # excluding last layer because it doesn't have a weight matrix
                self.layers.append(Layer(self.numOfNeuronsPerLayer[a:b], self.numOfNeuronsPerLayer[i]))
                a += 1
                b += 1
            self.layers.append(Layer((1,1), self.numOfNeuronsPerLayer[-1]))     # giving last layer a dummy weight matrix of 1x1
        elif type(self.numOfNeuronsPerLayer) == int:    # When all layers have same number of neurons
            for i in xrange(self.numOfLayers - 1):
                weightMatrixShape = (self.numOfNeuronsPerLayer, self.numOfNeuronsPerLayer)
                self.layers.append(Layer(weightMatrixShape, self.numOfNeuronsPerLayer))
            self.layers.append(Layer((1,1), self.numOfNeuronsPerLayer))
        
    
    def feed_forward(self, X_train):
        i = 0
        
        # Setting first layer's neurons output same as input
        for neuron in self.layers[0].neurons:
            neuron.out = X_train[i]
            i += 1
        
        self.layers[0].compute_output()
        
                
        # multiplying weight vectors and outputs vectors of previous layers in order to calculate each layers output
        for layer in xrange(1, len(self.layers)):
            for neuron in xrange(len(self.layers[layer].neurons)):
                self.layers[layer].neurons[neuron].net = numpy.dot(self.layers[layer - 1].out, self.layers[layer - 1].parameters[:, neuron:neuron+1])                                       
                self.layers[layer].neurons[neuron].activate()
            self.layers[layer].compute_output()
            
    
    def compute_error(self, Y_train):
        return (numpy.sum((Y_train - self.layers[self.numOfLayers - 1].out) ** 2) / 2.0)
        
    
    def back_propagate(self, Y_train):
        self.error = self.compute_error(Y_train)
        
        # computing deltas for last layer's neurons
        lastLayer = self.layers[self.numOfLayers - 1]
        for i in xrange(lastLayer.numOfNeurons):
            lastLayer.delta[i] = ( -(Y_train[i] - lastLayer.neurons[i].out) * (lastLayer.neurons[i].out * (1 - lastLayer.neurons[i].out)) )
            
        secondLastLayer = self.layers[self.numOfLayers - 2]
        
        for i in xrange(secondLastLayer.parameters.shape[0]):
            for j in xrange(secondLastLayer.parameters.shape[1]):
                secondLastLayer.newParameters[i, j] = (self.etta * lastLayer.delta[j] * secondLastLayer.neurons[i].out)
        
        secondLastLayer.newParameters = numpy.subtract(secondLastLayer.parameters, secondLastLayer.newParameters)
                
        # computing deltas and new weight matrices for remaining layers
        for i in xrange(self.numOfLayers - 2, 0, -1):   # -1 for reverse loop
            layer = self.layers[i]
            for j in xrange(layer.numOfNeurons):
                layer.delta[j] = numpy.dot(self.layers[i+1].delta, layer.parameters[j, :]) * (layer.neurons[j].out * (1 - layer.neurons[j].out))
            
            previousLayer = self.layers[i - 1]
            
            for k in xrange(previousLayer.parameters.shape[0]):
                for l in xrange(previousLayer.parameters.shape[1]):
                    #previousLayer.newParameters[i, j] = (self.etta * layer.delta[l] * previousLayer.neurons[k].out)
                    previousLayer.newParameters[k, l] = (self.etta * layer.delta[l] * previousLayer.neurons[k].out)
            previousLayer.newParameters = previousLayer.parameters - previousLayer.newParameters
        
    def update_weights(self):
        
        for layer in self.layers:
            layer.parameters = layer.newParameters.copy()
        
        
    def fit(self, X_train, Y_train, epochs = 1000):
        
        for epoch in xrange(epochs):
            for i in xrange(X_train.shape[0]):
                self.feed_forward(X_train[i, :])
                self.back_propagate(Y_train[i, :])
                self.update_weights()
    
                        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        