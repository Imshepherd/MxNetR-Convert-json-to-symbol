#Library
library(mxnet)

#Setpth
input.path = "data/usable.RData"
input.path2 = "prediction/LeNet.result.RData"

#load data
load(input.path); load(input.path2)


#OPEN "1. LeNet/Created Code.R" and post up.
data = mx.symbol.Variable(name = 'data')
conv1 <- mx.symbol.Convolution(data = data, kernel = "(5,5)", num_filter = "20", name = 'conv1')
tanh1 <- mx.symbol.Activation(data = conv1, act_type = "tanh", name = 'tanh1')
pool1 <- mx.symbol.Pooling(data = tanh1, kernel = "(2,2)", pool_type = "max", stride = "(2,2)", name = 'pool1')
conv2 <- mx.symbol.Convolution(data = pool1, kernel = "(5,5)", num_filter = "50", name = 'conv2')
tanh2 <- mx.symbol.Activation(data = conv2, act_type = "tanh", name = 'tanh2')
pool2 <- mx.symbol.Pooling(data = tanh2, kernel = "(2,2)", pool_type = "max", stride = "(2,2)", name = 'pool2')
flatten <- mx.symbol.Flatten(data = pool2, name = 'flatten')
fc1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = "500", name = 'fc1')
tanh3 <- mx.symbol.Activation(data = fc1, act_type = "tanh", name = 'tanh3')
fc2 <- mx.symbol.FullyConnected(data = tanh3, num_hidden = "10", name = 'fc2')
softmaxoutput5 <- mx.symbol.SoftmaxOutput(data = fc2, name = 'softmaxoutput5')

#OPEN "1. LeNet/Created Code.R" and post up.


##on cpu
mx.set.seed(0)
tic <- proc.time()
model <- mx.model.FeedForward.create(softmaxoutput5, X=train.array, y=train.y,
                                     ctx=mx.cpu(), num.round=5, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100),
                                     batch.end.callback = mx.callback.log.speedometer(100))

PREDICT2 = predict(model, train.array)

all.equal(PREDICT, PREDICT2)
