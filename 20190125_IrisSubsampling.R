# Load libraries
library(keras)

# Load data
data <- iris
x_val <- data[,1:4]
y_val <- data[,5]

# Identity function
identity_lambda <- function(arg) {
  arg + 0
}

# Begin model
model <- keras_model_sequential()

# Add layers
layer_lambda(model,identity_lambda)
layer_dense(model,units = 3, activation = 'linear')
layer_dense(model,units = 3, activation = 'softmax')

# Compile model
compile(model, optimizer = "nadam", 
        loss = 'categorical_crossentropy',
        metrics = 'categorical_accuracy')

# Train the model
# fit(model,x = as.matrix(x_val), 
#     y = to_categorical(as.numeric(y_val)-1),
#     batch_size = length(y_val),
#     epochs = 1000,
#     view_metrics = FALSE)

# A bootstrap fitting algorithm
# Set number of epochs
outer_epochs <- 1000
# Initialize metrics_history
metrics_history <- NULL
for (ii in 1:outer_epochs) {
  cat(ii,"\n")
  # Obtain a bootstrap sample (size n)
  sample_list <- sample(1:length(y_val), size = 50, replace = T)
  x_sample <- x_val[sample_list,]
  y_sample <- y_val[sample_list]
  # Run one internal epoch on boostrap sample
  history <- fit(model,x = as.matrix(x_sample), 
                  y = to_categorical(as.numeric(y_sample)-1),
                  batch_size = length(y_sample),
                  epochs = 1,
                  view_metrics = FALSE)
  metrics_history <- rbind(metrics_history,
                           unlist(history$metrics))
}

# Make predictions
predictions <- predict(model,as.matrix(x_val))
# Get predicted classes
pred_classes <- apply(predictions,1,which.max)
# Get accuracy
mean(pred_classes==as.numeric(y_val))
