#library
library(mxnet)

#Setpth
input.path = "data/"
output.path = "data/usable.RData"

##Data with kaggle web
train <- read.csv(paste0(input.path, 'train.csv'))
test <- read.csv(paste0(input.path, 'test.csv'))

train <- data.matrix(train)
test <- data.matrix(test)

train.x <- train[1:2000,-1]
train.y <- train[1:2000,1]

test <- test[1:2000,]

train.x <- t(train.x/255)
test <- t(test/255)

table(train.y)

##Data into array
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test
dim(test.array) <- c(28, 28, 1, ncol(test))



##
save(train.array, train.y, test.array, file = output.path)