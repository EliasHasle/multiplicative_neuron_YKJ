# multiplicative_neuron_YKJ

A simple implementation of the multiplicative neuron described in:
R.N. Yadav, P.K. Kalra & J. John (2006) On the use of Multiplicative Neuron
in Feedforward Neural Networks, International Journal of Modelling and Simulation, 26:4, 331-336,
DOI: 10.1080/02286203.2006.11442385
https://doi.org/10.1080/02286203.2006.11442385

The outputs are products of wij*xj+bij expressions, where wij,bij are taken from trainable matrices W,B, respectively. Stacked layers of these neurons can represent multivariate polynomials.

Basic example:

    from multiplicative_neuron_YKJ import mul_YKJ
    #x is a tensor expression
	y = mul_YKJ(x,2) #2 is the number of outputs

Apart from this, the code is the documentation. There are additional parameters for initializers, activation function and regularizers. Regularization losses (if any) are added to tf.GraphKeys.REGULARIZATION_LOSSES.

There is an example with regularization and training in the source, which activates when the file is run as main script.

The implementation is released under an MIT license. Use freely, but please attribute the author properly and also cite the original paper.

If any problems are encountered, I will be glad to know (glad for the knowledge, not for the problem). You are welcome to submit an issue or a pull request.

Elias Hasle, @EliasHasle