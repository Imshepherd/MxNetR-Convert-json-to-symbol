# MxNet-Conver-json-to-symbol
## Convert network written in json format to mxnet symbol code
 If you have a pretrained model in mxnet format. Doing the Fine-tune with Pretrained Models works. 
 
 You can use the example code below directly in the R console.
 
 For example for the DesNet.
 
```r
  Dense_model = mx.model.load('model/densenet-imagenet-169-0', 125)

  all_layers = Dense_model$symbol$get.internals()
  relu1_output = which(all_layers$outputs == 'relu1_output') %>% all_layers$get.output()
  softmax_output = which(all_layers$outputs == 'softmax_output') %>% all_layers$get.output()
  
  out = mx.symbol.Group(c(relu1_output, softmax_output))
  executor = mx.simple.bind(symbol = out, data = c(224, 224, 3, 1), ctx = mx.cpu())
  
  mx.exec.update.arg.arrays(executor, Dense_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(executor, Dense_model$aux.params, match.name = TRUE)
```
 And run the executor by other mxnet function
 
 Using symbol.get_internals to get the internal parts, can only get symbol from start.
 
 ## When you want to use specific layers or 
 ## Changing some specific layers architecture in pre-train model.
 
 You may rewrite the hole Net.
 
 Or you may use function in ['1. LeNet/3. convert json to symbol.R'](https://github.com/Imshepherd/MxNet-Conver-json-to-symbol/blob/master/1.%20LeNet/3.%20convert%20json%20to%20symbol.R), converting json files to R code for the example of LeNet.
 
 Also, there is an example of DesNet. 
 
 The example code in here['1. LeNet/3. convert json to symbol.R'](https://github.com/Imshepherd/MxNet-Conver-json-to-symbol/blob/master/2.%20DesNet/1.%20convert%20json%20to%20symbol.R)
 
 Pre-Train densenet model is downloading from
 - [densenet](https://github.com/bruinxiong/densenet.mxnet)
 

