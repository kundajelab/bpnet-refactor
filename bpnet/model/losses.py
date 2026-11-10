import tensorflow as tf


def mse_loss_function(y_log_true, y_log_pred):
    # logcounts mse loss without sample weights
    mse_loss = tf.keras.losses.mean_squared_error(
        y_log_true, y_log_pred)
    return mse_loss

def poisson_loss_function(y_log_true, y_log_pred):
    # we can use the Possion PMF from TensorFlow as well
    # dist = tf.contrib.distributions
    # return -tf.reduce_mean(dist.Poisson(y_pred).log_pmf(y_true))

    # last term can be avoided since it doesn't depend on y_pred
    # however keeping it gives a nice lower bound to zero
    
    y_true = tf.math.exp(y_log_true) # -1? 
    y_pred = tf.math.exp(y_log_pred)
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)
    loss = y_pred - y_true*tf.math.log(y_pred+1e-8) + tf.math.lgamma(y_true+1.0)

    return loss

def multinomial_log_prob(true_counts, logits):
    """
        Log probability of the observed counts under a multinomial 
        distribution over the last axis whose probabilities are 
        softmax(logits) and whose total is the sum of the counts
        
        log(n!) - sum_i log(x_i!) + sum_i x_i * log_softmax(logits)_i
        
        This is the log_prob of tensorflow-probability's Multinomial
        (which this package used until it stopped depending on that
        library), written with the same operations, in the same 
        order and under the same name scopes. Values and gradients 
        are then the same to the last bit, eagerly and in a compiled 
        function. (The name scopes matter: in a compiled function 
        TensorFlow adds up the contributions to the gradient with 
        respect to the counts in an order that depends on the names 
        of the operations.)
        
        Args:
            true_counts (tensorflow.Tensor): observed counts, 
                (..., positions)
            logits (tensorflow.Tensor): predicted logits of the same 
                shape and type
                
        Returns:
            tensorflow.Tensor: log probabilities, (...)
    """
    true_counts = tf.convert_to_tensor(true_counts)
    logits = tf.convert_to_tensor(logits)
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)

    with tf.name_scope('Multinomial'), tf.name_scope('log_prob'):
        # positions without counts contribute nothing, whatever their 
        # log probability (also when it is -inf)
        log_p = tf.math.log_softmax(logits)
        log_unnormalized = tf.reduce_sum(
            tf.math.multiply_no_nan(log_p, true_counts), axis=-1)

        # log of the number of ways the counts can be arranged
        with tf.name_scope('log_combinations'):
            total_permutations = tf.math.lgamma(counts_per_example + 1)
            counts_factorial = tf.math.lgamma(true_counts + 1)
            redundant_permutations = tf.reduce_sum(
                counts_factorial, axis=-1)
            log_combinations = total_permutations - redundant_permutations

        return log_unnormalized + log_combinations


def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood
    Args:
        true_counts: observed count values
        logits: predicted logits values
    """
    return (-tf.reduce_sum(multinomial_log_prob(true_counts, logits)) / 
            tf.cast(tf.shape(true_counts)[0], dtype=tf.float32))
