#library
library(mxnet)

#Setpth
input.path = "data/usable.RData"
output.path = "model/LeNet"
output.path2 = "prediction/LeNet.result.RData"

#load 
load(input.path)

##LeNet
# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel="(5,5)", num_filter="20", name = "conv1")
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh", name = "tanh1")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel="(2,2)", stride="(2,2)", name = "pool1")
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel="(5,5)", num_filter="50", name = "conv2")
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh", name = "tanh2")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel="(2,2)", stride="(2,2)", name = "pool2")
# first fullc
flatten <- mx.symbol.Flatten(data=pool2, name = "flatten")
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden="500", name = "fc1")
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh", name = "tanh3")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden="10", name = "fc2")
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)


##on cpu
mx.set.seed(0)
tic <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=mx.cpu(), num.round=5, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100),
                                     batch.end.callback = mx.callback.log.speedometer(100))

print(proc.time() - tic)

##model save
mx.model.save(model, output.path, 5)


##predict output

PREDICT = predict(model, train.array)

save(PREDICT, file = output.path2)
