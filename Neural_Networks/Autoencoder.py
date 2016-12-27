import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

epochs = 200
batch_size = 100

num_input = 784
num_hidden_1 = 300
num_hidden_2 = 150

num_show = 10

input_feature = tf.placeholder(tf.float32,[None,num_input])

weights = {
	'encode_hidden_1' : tf.Variable(tf.random_normal([num_input,num_hidden_1])),
	'encode_hidden_2' : tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2])),
	'decode_hidden_1' : tf.Variable(tf.random_normal([num_hidden_2,num_hidden_1])),
	'decode_hidden_2' : tf.Variable(tf.random_normal([num_hidden_1,num_input]))
}
biases = {
	'encode_hidden_1' : tf.Variable(tf.random_normal([num_hidden_1])),
	'encode_hidden_2' : tf.Variable(tf.random_normal([num_hidden_2])),
	'decode_hidden_1' : tf.Variable(tf.random_normal([num_hidden_1])),
	'decode_hidden_2' : tf.Variable(tf.random_normal([num_input]))
}

def encoder_model(input_raw, w, b):
	output_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_raw,w['encode_hidden_1']), b['encode_hidden_1']))
	output_2 = tf.nn.sigmoid(tf.add(tf.matmul(output_1,w['encode_hidden_2']), b['encode_hidden_2']))
	return output_2

def decoder_model(output_raw, w, b):
	revert_1 = tf.nn.sigmoid(tf.add(tf.matmul(output_raw,w['decode_hidden_1']), b['decode_hidden_1']))
	revert_2 = tf.nn.sigmoid(tf.add(tf.matmul(revert_1,w['decode_hidden_2']), b['decode_hidden_2']))
	return revert_2

encode_feature = encoder_model(input_feature, weights, biases)
revert_feature = decoder_model(encode_feature, weights, biases)

cost = tf.reduce_mean(tf.pow(revert_feature - input_feature, 2))
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
batch_total = mnist.train.num_examples//batch_size
for epoch in range(epochs):
	cost_sum = 0.0
	for batch in range(batch_total):
		batch_feature, batch_label = mnist.train.next_batch(batch_size)
		_ , cost_receive = sess.run([optimizer, cost], feed_dict={input_feature:batch_feature})
		cost_sum += cost_receive
	cost_avg = cost_sum/batch_total
	print("Epoch ", epoch , " Cost : ", cost_avg)
print("Autoencoder Building Done !!!")

test_revert = sess.run(revert_feature, feed_dict={input_feature: mnist.test.images[:num_show]})
f, a = plt.subplots(2, num_show)
for i in range(num_show):
	a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
	a[1][i].imshow(np.reshape(test_revert[i], (28, 28)))
f.show()
plt.draw()
plt.waitforbuttonpress()

sess.close()
print('Test Show Over')
#结果保存在战‘Autoencoder.png’
'''
Epoch  0  Cost :  0.259587637972
Epoch  1  Cost :  0.162556138174
Epoch  2  Cost :  0.145353405801
Epoch  3  Cost :  0.135164313858
Epoch  4  Cost :  0.126551522206
Epoch  5  Cost :  0.121471781798
Epoch  6  Cost :  0.118880662187
Epoch  7  Cost :  0.116534240422
Epoch  8  Cost :  0.112446515425
Epoch  9  Cost :  0.109391464428
Epoch  10  Cost :  0.108252271929
Epoch  11  Cost :  0.105574365407
Epoch  12  Cost :  0.10310472506
Epoch  13  Cost :  0.102590527385
Epoch  14  Cost :  0.101425595934
Epoch  15  Cost :  0.100492113138
Epoch  16  Cost :  0.0988104778935
Epoch  17  Cost :  0.0961448389292
Epoch  18  Cost :  0.0943582539667
Epoch  19  Cost :  0.0928582459011
Epoch  20  Cost :  0.0919606232372
Epoch  21  Cost :  0.0904809739373
Epoch  22  Cost :  0.0903096014532
Epoch  23  Cost :  0.0900590482761
Epoch  24  Cost :  0.0898591443084
Epoch  25  Cost :  0.0896481411836
Epoch  26  Cost :  0.0894288760695
Epoch  27  Cost :  0.0892112809961
Epoch  28  Cost :  0.0890101494166
Epoch  29  Cost :  0.0885129881582
Epoch  30  Cost :  0.0882559005645
Epoch  31  Cost :  0.0878804701431
Epoch  32  Cost :  0.0876312197745
Epoch  33  Cost :  0.08729038057
Epoch  34  Cost :  0.0869944601303
Epoch  35  Cost :  0.085925106623
Epoch  36  Cost :  0.0854257397625
Epoch  37  Cost :  0.0851852478087
Epoch  38  Cost :  0.0849117267538
Epoch  39  Cost :  0.0846029523557
Epoch  40  Cost :  0.0843833194809
Epoch  41  Cost :  0.0839610887522
Epoch  42  Cost :  0.0837727562812
Epoch  43  Cost :  0.0818550784615
Epoch  44  Cost :  0.0814572540874
Epoch  45  Cost :  0.0813636116006
Epoch  46  Cost :  0.0812157314609
Epoch  47  Cost :  0.0805009860884
Epoch  48  Cost :  0.0798344136097
Epoch  49  Cost :  0.0796295726028
Epoch  50  Cost :  0.0795016306639
Epoch  51  Cost :  0.0792201292379
Epoch  52  Cost :  0.0790215729854
Epoch  53  Cost :  0.0789156909693
Epoch  54  Cost :  0.0782274431803
Epoch  55  Cost :  0.0775258439915
Epoch  56  Cost :  0.0770959964395
Epoch  57  Cost :  0.0769478762421
Epoch  58  Cost :  0.0768949017741
Epoch  59  Cost :  0.0767293149504
Epoch  60  Cost :  0.0766169623895
Epoch  61  Cost :  0.0762614174187
Epoch  62  Cost :  0.0741570095853
Epoch  63  Cost :  0.0738090357455
Epoch  64  Cost :  0.0737176126106
Epoch  65  Cost :  0.0736233179678
Epoch  66  Cost :  0.0735246578807
Epoch  67  Cost :  0.0734141018309
Epoch  68  Cost :  0.073319281868
Epoch  69  Cost :  0.073232846531
Epoch  70  Cost :  0.073127714233
Epoch  71  Cost :  0.0730034248666
Epoch  72  Cost :  0.072942737517
Epoch  73  Cost :  0.0728811041604
Epoch  74  Cost :  0.0727632126619
Epoch  75  Cost :  0.0726844623685
Epoch  76  Cost :  0.0725862965665
Epoch  77  Cost :  0.0725168116662
Epoch  78  Cost :  0.0724524813484
Epoch  79  Cost :  0.07178539268
Epoch  80  Cost :  0.0712668908049
Epoch  81  Cost :  0.0711342159049
Epoch  82  Cost :  0.0710371786762
Epoch  83  Cost :  0.0702500576729
Epoch  84  Cost :  0.0694232504612
Epoch  85  Cost :  0.0693236357245
Epoch  86  Cost :  0.0692625038597
Epoch  87  Cost :  0.0691571690819
Epoch  88  Cost :  0.0686599025401
Epoch  89  Cost :  0.0680940536342
Epoch  90  Cost :  0.0672960155931
Epoch  91  Cost :  0.0669290369072
Epoch  92  Cost :  0.0650467702205
Epoch  93  Cost :  0.0646482206271
Epoch  94  Cost :  0.0645429221337
Epoch  95  Cost :  0.0644679244269
Epoch  96  Cost :  0.0635266994685
Epoch  97  Cost :  0.0618886190924
Epoch  98  Cost :  0.061824896952
Epoch  99  Cost :  0.0610317883573
Epoch  100  Cost :  0.0604139828546
Epoch  101  Cost :  0.0602467680993
Epoch  102  Cost :  0.0601661374217
Epoch  103  Cost :  0.0601247934117
Epoch  104  Cost :  0.0596212242815
Epoch  105  Cost :  0.0586923423477
Epoch  106  Cost :  0.0584887686643
Epoch  107  Cost :  0.0572902670299
Epoch  108  Cost :  0.0572436174005
Epoch  109  Cost :  0.0559862557663
Epoch  110  Cost :  0.0558665195514
Epoch  111  Cost :  0.0545373334126
Epoch  112  Cost :  0.0541049211608
Epoch  113  Cost :  0.0532278992507
Epoch  114  Cost :  0.0523548385433
Epoch  115  Cost :  0.0511120720614
Epoch  116  Cost :  0.0504398909482
Epoch  117  Cost :  0.0503872564638
Epoch  118  Cost :  0.0503559213335
Epoch  119  Cost :  0.0502614230595
Epoch  120  Cost :  0.0502259644189
Epoch  121  Cost :  0.0502367500758
Epoch  122  Cost :  0.0501696679538
Epoch  123  Cost :  0.0501075282151
Epoch  124  Cost :  0.050097976923
Epoch  125  Cost :  0.0500239241665
Epoch  126  Cost :  0.0499898160249
Epoch  127  Cost :  0.0499648847092
Epoch  128  Cost :  0.048291490735
Epoch  129  Cost :  0.0475470102511
Epoch  130  Cost :  0.0474949120798
Epoch  131  Cost :  0.046436529356
Epoch  132  Cost :  0.0449345546351
Epoch  133  Cost :  0.0448276842656
Epoch  134  Cost :  0.0447779544307
Epoch  135  Cost :  0.0447534301538
Epoch  136  Cost :  0.0447177326341
Epoch  137  Cost :  0.0446910231222
Epoch  138  Cost :  0.0442943542722
Epoch  139  Cost :  0.0433353492685
Epoch  140  Cost :  0.0432751607895
Epoch  141  Cost :  0.0427474084226
Epoch  142  Cost :  0.0415395243805
Epoch  143  Cost :  0.0409495862844
Epoch  144  Cost :  0.0395026334714
Epoch  145  Cost :  0.039385244677
Epoch  146  Cost :  0.039367156916
Epoch  147  Cost :  0.0393049473112
Epoch  148  Cost :  0.0391948895427
Epoch  149  Cost :  0.0387962654436
Epoch  150  Cost :  0.0385489583354
Epoch  151  Cost :  0.0380083373663
Epoch  152  Cost :  0.0378711248121
Epoch  153  Cost :  0.0378064774315
Epoch  154  Cost :  0.0377889479494
Epoch  155  Cost :  0.0377233364162
Epoch  156  Cost :  0.0376898357882
Epoch  157  Cost :  0.0376691519266
Epoch  158  Cost :  0.0369727073271
Epoch  159  Cost :  0.0367518726059
Epoch  160  Cost :  0.0367066366496
Epoch  161  Cost :  0.0366304200346
Epoch  162  Cost :  0.0366118367694
Epoch  163  Cost :  0.0365651990338
Epoch  164  Cost :  0.0365369765799
Epoch  165  Cost :  0.0365090182424
Epoch  166  Cost :  0.0364604148066
Epoch  167  Cost :  0.0364323453199
Epoch  168  Cost :  0.0364056124403
Epoch  169  Cost :  0.0363947022909
Epoch  170  Cost :  0.0363067797097
Epoch  171  Cost :  0.0363170930879
Epoch  172  Cost :  0.0362804844366
Epoch  173  Cost :  0.0362237312306
Epoch  174  Cost :  0.0362239721824
Epoch  175  Cost :  0.036185995381
Epoch  176  Cost :  0.0361593886736
Epoch  177  Cost :  0.036138049845
Epoch  178  Cost :  0.0360569947281
Epoch  179  Cost :  0.0359651234543
Epoch  180  Cost :  0.0359171813794
Epoch  181  Cost :  0.0358721747385
Epoch  182  Cost :  0.035830762664
Epoch  183  Cost :  0.0357761263305
Epoch  184  Cost :  0.0356516988846
Epoch  185  Cost :  0.0355847170136
Epoch  186  Cost :  0.0354380835593
Epoch  187  Cost :  0.0353867508539
Epoch  188  Cost :  0.0353269614415
Epoch  189  Cost :  0.0348537637767
Epoch  190  Cost :  0.0345198963041
Epoch  191  Cost :  0.0344652953812
Epoch  192  Cost :  0.0344421602853
Epoch  193  Cost :  0.0344206559929
Epoch  194  Cost :  0.0343837092952
Epoch  195  Cost :  0.0343631312725
Epoch  196  Cost :  0.0343373144621
Epoch  197  Cost :  0.0343216292763
Epoch  198  Cost :  0.0342855004289
Epoch  199  Cost :  0.0342540984804
Autoencoder Building Done !!!
'''
