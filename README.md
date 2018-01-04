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
  
  executor = mx.simple.bind(symbol = out, data = c(224, 224, 3, 1), ctx = mx.cpu())
  
  mx.exec.update.arg.arrays(executor, Dense_model$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(executor, Dense_model$aux.params, match.name = TRUE)
```
 And run the executor by other mxnet function
 Using symbol.get_internals to get the internal parts, only can get symbol from start.
 
 ## when you want to use specific layers or change some specific layers architecture in pre-train model.
 
 
