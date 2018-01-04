#Library
library(rjson)
library(mxnet)
library(magrittr)

#Setpath
input.path = "model/LeNet"
output.path = "1. LeNet/Created Code.R"

#load model
model = mx.model.load(input.path,5)

#convet to json format
JSON = model$symbol$as.json()

#read json format into list
model.list = fromJSON(JSON)

#get all information from list
Symbol.List = list(
  Names = sapply( 1:length(model.list$nodes), function(x){model.list$nodes[[x]]$name} ),
  op = sapply( 1:length(model.list$nodes), function(x){model.list$nodes[[x]]$op} )
)


##################
#Creating fucntion

Create.Symbol.data = function(){
  return(paste0("data = mx.symbol.Variable(name = 'data')", "\n"))
}

Create.Symbol.Convolution = function(LIST){
  
  parameter = paste0(names(LIST$attr), " = ", paste0('"', LIST$attr, '"'))
  if (length(parameter) > 5){
    parameter[3] = paste0(parameter[3], "\n")
  }
  parameter = paste0(paste0(parameter, ", "), collapse = "")
  
  INPUT = LIST$inputs 
  INPUT.num = sapply(1:length(INPUT), function(x){INPUT[[x]][1]+1})
  
  if (sum(INPUT.num %in% 1) == 1) {  #1 represent data
    INPUT.arg = "data"
  } else {
    which.is.op = which(Symbol.List$op[INPUT.num] != "null")
    INPUT.arg = Symbol.List$Names[INPUT.num[which.is.op]]
  }
  
  if ( !is.null(LIST$attr) ){
    TEXT = paste0(
      LIST$name, " <- ",
      "mx.symbol.Convolution(",
      "data = ", INPUT.arg, ", ",     
      parameter,
      "name = '", LIST$name, "'",
      ")", "\n")
    
  } else {
    TEXT = paste0(
      LIST$name, " <- ",
      "mx.symbol.Convolution(",
      "data = ", INPUT.arg, ", ",     
      "name = '", LIST$name, "'",
      ")", "\n")
    
  }
  
  return(TEXT)
}

Create.Symbol.Activation = function(LIST){
  
  parameter = paste0(names(LIST$attr), " = ", paste0('"', LIST$attr, '"'))
  if (length(parameter) > 5){
    parameter[3] = paste0(parameter[3], "\n")
  }
  parameter = paste0(paste0(parameter, ", "), collapse = "")
  
  INPUT = LIST$inputs 
  INPUT.num = sapply(1:length(INPUT), function(x){INPUT[[x]][1]+1})
  
  if (sum(INPUT.num %in% 1) == 1) {  #1 represent data
    INPUT.arg = "data"
  } else {
    which.is.op = which(Symbol.List$op[INPUT.num] != "null")
    INPUT.arg = Symbol.List$Names[INPUT.num[which.is.op]]
  }
  
  if ( !is.null(LIST$attr) ){
    TEXT = paste0(
      LIST$name, " <- ",
      "mx.symbol.Activation(",
      "data = ", INPUT.arg, ", ",     
      parameter,
      "name = '", LIST$name, "'",
      ")", "\n")
    
  } else {
    TEXT = paste0(
      LIST$name, " <- ",
      "mx.symbol.Activation(",
      "data = ", INPUT.arg, ", ",     
      "name = '", LIST$name, "'",
      ")", "\n")
    
  }
  
  return(TEXT)
}

Create.Symbol.Pooling = function(LIST){
  
  parameter = paste0(names(LIST$attr), " = ", paste0('"', LIST$attr, '"'))
  if (length(parameter) > 5){
    parameter[3] = paste0(parameter[3], "\n")
  }
  parameter = paste0(paste0(parameter, ", "), collapse = "")
  
  INPUT = LIST$inputs 
  INPUT.num = sapply(1:length(INPUT), function(x){INPUT[[x]][1]+1})
  
  if (sum(INPUT.num %in% 1) == 1) {  #1 represent data
    INPUT.arg = "data"
  } else {
    which.is.op = which(Symbol.List$op[INPUT.num] != "null")
    INPUT.arg = Symbol.List$Names[INPUT.num[which.is.op]]
  }
  
  if ( !is.null(LIST$attr) ){
    TEXT = paste0(
      LIST$name, " <- ",
      "mx.symbol.Pooling(",
      "data = ", INPUT.arg, ", ",     
      parameter,
      "name = '", LIST$name, "'",
      ")", "\n")
    
  } else {
    TEXT = paste0(
      LIST$name, " <- ",
      "mx.symbol.Pooling(",
      "data = ", INPUT.arg, ", ",     
      "name = '", LIST$name, "'",
      ")", "\n")
    
  }
  
  return(TEXT)
}

Create.Symbol.Flatten = function(LIST){
  
  parameter = paste0(names(LIST$attr), " = ", paste0('"', LIST$attr, '"')) #############171230
  if (length(parameter) > 5){
    parameter[3] = paste0(parameter[3], "\n")
  }
  parameter = paste0(paste0(parameter, ", "), collapse = "")
  
  INPUT = LIST$inputs 
  INPUT.num = sapply(1:length(INPUT), function(x){INPUT[[x]][1]+1})
  
  if (sum(INPUT.num %in% 1) == 1) {  #1 represent data
    INPUT.arg = "data"
  } else {
    which.is.op = which(Symbol.List$op[INPUT.num] != "null")
    INPUT.arg = Symbol.List$Names[INPUT.num[which.is.op]]
  }
  
  if ( !is.null(LIST$attr) ){
    TEXT = paste0(
      LIST$name, " <- ",
      "mx.symbol.Flatten(",
      "data = ", INPUT.arg, ", ",     
      parameter,
      "name = '", LIST$name, "'",
      ")", "\n")
    
  } else {
    TEXT = paste0(
      LIST$name, " <- ",
      "mx.symbol.Flatten(",
      "data = ", INPUT.arg, ", ",     
      "name = '", LIST$name, "'",
      ")", "\n")
    
  }
  
  
  return(TEXT)
}

Create.Symbol.FullyConnected = function(LIST){
  
  parameter = paste0(names(LIST$attr), " = ", paste0('"', LIST$attr, '"')) #############171230
  if (length(parameter) > 5){
    parameter[3] = paste0(parameter[3], "\n")
  }
  parameter = paste0(paste0(parameter, ", "), collapse = "")
  
  INPUT = LIST$inputs 
  INPUT.num = sapply(1:length(INPUT), function(x){INPUT[[x]][1]+1})
  
  if (sum(INPUT.num %in% 1) == 1) {  #1 represent data
    INPUT.arg = "data"
  } else {
    which.is.op = which(Symbol.List$op[INPUT.num] != "null")
    INPUT.arg = Symbol.List$Names[INPUT.num[which.is.op]]
  }
  
  if ( !is.null(LIST$attr) ){
    TEXT = paste0(
      LIST$name, " <- ",
      "mx.symbol.FullyConnected(",
      "data = ", INPUT.arg, ", ",     
      parameter,
      "name = '", LIST$name, "'",
      ")", "\n")
    
  } else {
    TEXT = paste0(
      LIST$name, " <- ",
      "mx.symbol.FullyConnected(",
      "data = ", INPUT.arg, ", ",     
      "name = '", LIST$name, "'",
      ")", "\n")
    
  }
  
  
  return(TEXT)
}

Create.Symbol.SoftmaxOutput = function(LIST){
  parameter = paste0(names(LIST$attr), " = ", paste0('"', LIST$attr, '"')) #############171230
  if (length(parameter) > 5){
    parameter[3] = paste0(parameter[3], "\n")
  }
  parameter = paste0(paste0(parameter, ", "), collapse = "")
  
  INPUT = LIST$inputs 
  INPUT.num = sapply(1:length(INPUT), function(x){INPUT[[x]][1]+1})
  
  if (sum(INPUT.num %in% 1) == 1) {  #1 represent data
    INPUT.arg = "data"
  } else {
    which.is.op = which(Symbol.List$op[INPUT.num] != "null")
    INPUT.arg = Symbol.List$Names[INPUT.num[which.is.op]]
  }
  
  if ( !is.null(LIST$attr) ){
    TEXT = paste0(
      LIST$name, " <- ",
      "mx.symbol.SoftmaxOutput(",
      "data = ", INPUT.arg, ", ",     
      parameter,
      "name = '", LIST$name, "'",
      ")", "\n")
    
  } else {
    TEXT = paste0(
      LIST$name, " <- ",
      "mx.symbol.SoftmaxOutput(",
      "data = ", INPUT.arg, ", ",     
      "name = '", LIST$name, "'",
      ")", "\n")
    
  }
  
  
  return(TEXT)
}


###Loop for creat code

OUTPUT.file = NULL

for (i in 1:length(Symbol.List$Names)){
  LIST = model.list$nodes[[i]]
  
  if (i == 1){
    OUTPUT.file[i] = Create.Symbol.data()
  }
  
  if ( LIST$op == "Convolution"){
    OUTPUT.file[i] = Create.Symbol.Convolution(LIST)
  }
  
  if ( LIST$op == "Activation"){
    OUTPUT.file[i] = Create.Symbol.Activation(LIST)
  }
  
  if ( LIST$op == "Pooling"){
    OUTPUT.file[i] = Create.Symbol.Pooling(LIST)
  }
  
  if ( LIST$op == "Flatten"){
    OUTPUT.file[i] = Create.Symbol.Flatten(LIST)
  }
  
  if ( LIST$op == "FullyConnected"){
    OUTPUT.file[i] = Create.Symbol.FullyConnected(LIST)
  }
  
  if ( LIST$op == "SoftmaxOutput"){
    OUTPUT.file[i] = Create.Symbol.SoftmaxOutput(LIST)
  }
  
}

###write out code
OUTPUT.file = OUTPUT.file[!is.na(OUTPUT.file)]

cat(OUTPUT.file, file = output.path)
