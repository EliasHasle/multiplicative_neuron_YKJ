import tensorflow as tf

#f(PI(wij*xj+bij)), not sure this is correctly implemented yet!
def multiplicative_neuron_YKJ(
		inputs,
		output_size,
		activation=None,
		weight_initializer=tf.keras.initializers.TruncatedNormal(dtype=tf.float32),
		bias_initializer=tf.keras.initializers.TruncatedNormal(dtype=tf.float32),
		weight_regularizer=tf.contrib.layers.l1_regularizer(0.001),
		bias_regularizer=None,
		activity_regularizer=None
	):
	
	input_size = inputs.shape.as_list()[-1]
	#print("INPUT SIZE: %d" % input_size)
	
	#Weights
	W = tf.Variable(weight_initializer((output_size, input_size)))
	if weight_regularizer != None:
		weight_reg = weight_regularizer(W)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_reg)
	
	#Biases
	B = tf.Variable(bias_initializer((output_size, input_size)))
	if bias_regularizer != None:
		bias_reg = bias_regularizer(B)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, bias_reg)
	
	rows = W*inputs + B
	#print("ROWS SIZE: %s" % str(rows.shape.as_list()))
	h = tf.reduce_prod(rows, axis=(-1), keepdims=True)
	#print("H SIZE: %s" % str(h.shape.as_list()))
	if activation!=None:
		h = activation(h)
	if activity_regularizer != None:
		act_reg = activity_regularizer(h)
		tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, act_reg)
	return h
	
#Alias
mul_YKJ = multiplicative_neuron_YKJ

if __name__ == "__main__":
	import numpy as np
	dataset_size = 1000000
	xx = np.array([np.random.randint(-999,1000, (2,)) for e in range(dataset_size)], dtype=np.float32)
	yy = np.array([[np.prod(x)] for x in xx], dtype=np.float32)
	
	#print(xx)
	#print(yy)
	
	ins = tf.placeholder(tf.float32, (None,2))
	targets  = tf.placeholder(tf.float32, (None,1))
	outs = mul_YKJ(ins,1)#, bias_regularizer=tf.contrib.layers.l2_regularizer(0.01))
	
	deviation = outs-targets
	print("DEVIATION SHAPE: %s" % str(deviation.shape.as_list()))
	loss_comps = deviation/(tf.abs(targets)+0.5)
	print("LOSS_COMPS SHAPE: %s" % str(loss_comps.shape.as_list()))
	error = tf.nn.l2_loss(loss_comps)/dataset_size
	reg_loss = tf.losses.get_regularization_loss()
	#print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	#reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
	train_op = optimizer.minimize(error+reg_loss)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		
		#training
		for epoch in range(200):
			_,err,errr = sess.run([train_op, error, reg_loss], feed_dict={ins: xx, targets: yy})
			print("Error: %g, regularization loss: %g" % (err, errr))