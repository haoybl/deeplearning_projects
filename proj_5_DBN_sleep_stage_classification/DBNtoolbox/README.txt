DBNToolbox v1.0 (2010.11.24)

This toolbox contains matlab code for working with Deep Belief Nets. Most of
the code is written in object-oriented matlab classes. The core algorithms are
based off code by Ruslan Salakhutdinov and Geoff Hinton, as well as some of the
Theano Deep Learning tutorials (http://deeplearning.net/tutorial/).

This code is very un-optimized. If you find ways of improving it, please let me
know. I have not got around to implementing any of it in C for speed and
possible GPU use but am interested in the code if others do.

Feel free to use and modify this code as you wish, as long as it is for research
purposes. I make no guarnatees about it except that there are almost certainly
bugs. Please email me if you find them or have other suggestions.

Author:
Drausin Wulsin
wulsin@seas.upenn.edu
Department of Bioengineering, University of Pennsylvania



Files (in 'lib' directory)
--------------------------------------------------------------------------------

NNLayer.m	: the base (abstract) class for neural-network layers like RBMs

RBM.m	:			a class implementation of the Restricted Boltzmann Machine; a
						subclass of	NNLayer

DeepNN.m :	a class implementation of the Deep Belief Net

minimize.m :	a function for conjugate gradient descent

GetError.m :	used to calculate value and derivatives of the objective function
							in the DBN backprop minimization (minimize.m)

GreedyLayerTrain.m :	a function that greedily trains NNLayer subclasses
											(currently just RBMs,  but more to come later) in a
											sequential manner

TrainDeepNN.m :		a function that handles greedy pretraining and then constructs
									a DBN for either reconstruction or classification

--------------------------------------------------------------------------------

See the 'examples' directory for two simple (trivial) examples of 
classification and reconstruction DBNs.
