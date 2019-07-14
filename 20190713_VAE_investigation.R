#################################################################
##                   MLE Normal Distribution                   ##
#################################################################

# Load Library
library(tensorflow)

# Read data
data <- rnorm(10000,10,10)

# Optimization parameters
mu <- tf$Variable(tf$zeros(shape(1L)))
sigma <- tf$Variable(tf$ones(shape(1L)))
data_minus_mu <- data - mu

loss <- length(data)/2*tf$log(sigma^2)+tf$reduce_sum((data_minus_mu)^2)/2/(sigma^2)

optimizer <- tf$train$NadamOptimizer()
train <- optimizer$minimize(loss)

# Launch the graph and initialize the variables.
sess = tf$Session()
sess$run(tf$global_variables_initializer())

# Fit the line (Learns best fit is W: 0.1, b: 0.3)
for (step in 1:100001) {
  sess$run(train)
  if (step %% 20 == 0)
    cat(step, "-", sess$run(mu), sess$run(sigma), "\n")
}
