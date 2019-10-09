import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras import Model

class ElasticWeightConsolidation(Model):
    def __init__(self, num_inputs, num_hidden, num_outputs, weight=10e4):
        super(ElasticWeightConsolidation, self).__init__()
        self.f1 = Flatten(name='f1')
        self.lin1 = Dense(num_hidden, input_shape=(num_inputs,), activation='relu', name='lin1')
        self.lin1bn = BatchNormalization(name='lin1bn')
        self.lin2 = Dense(num_hidden, activation='relu', name='lin2')
        self.lin2bn = BatchNormalization(name='lin2bn')
        self.lin3 = Dense(num_outputs, activation='linear', name='lin3')
        
        self.estimated_mean = dict()
        self.estimated_fisher = dict()
        self.weight = weight

    def call(self, input_tensor, training=False):
        x = self.f1(input_tensor)
        x = self.lin1(x)
        x = self.lin1bn(x, training=training)
        x = self.lin2(x)
        x = self.lin2bn(x, training=training)
        return self.lin3(x)
        
    def update_mean_params(self):
        for variable in self.trainable_variables:
            self.estimated_mean[variable.name] = tf.identity(variable)

    def update_fisher_params(self, dataset, num_batch):
        batch_size = 100
        dl = dataset.batch(batch_size).shuffle(100000)
        log_likelihoods = None
        
        with tf.GradientTape() as tape:
            for i, (input, target) in enumerate(dl):
                indices = tf.expand_dims(target, axis=1)
                if i > num_batch:
                    break
                output = tf.nn.log_softmax(self(input), axis=1) 
                if log_likelihoods is None:
                    log_likelihoods = tf.gather_nd(output, indices)
                else:
                    log_likelihoods = tf.concat([log_likelihoods, tf.gather_nd(output, indices)], axis=0)
            log_likelihood = tf.math.reduce_mean(log_likelihoods)
        grad_log_liklihood = tape.gradient(log_likelihood, self.trainable_variables)
        for variable, weight in zip(self.trainable_variables, grad_log_liklihood):
             self.estimated_fisher[variable.name] = tf.math.square(tf.identity(weight))
        
#         for i, (input, target) in enumerate(dl):
#             if i > num_batch:
#                 break
#             with tf.GradientTape() as tape:
#                 output = tf.nn.log_softmax(self(input), axis=1) 
#                 log_likelihood = output[0][target.numpy()[0]]
#             grad_log_liklihood = tape.gradient(log_likelihood, self.trainable_variables)
#             for variable, weight in zip(self.trainable_variables, grad_log_liklihood):
#                 if self.estimated_fisher[variable.name] is None:
#                     self.estimated_fisher[variable.name] = tf.math.square(tf.identity(weight))
#                 else:
#                     self.estimated_fisher[variable.name] += tf.math.square(tf.identity(weight))
#         for variable, weight in zip(self.trainable_variables, grad_log_liklihood):
#             self.estimated_fisher[variable.name] /= num_batch
                        
                        
    def register_ewc_params(self, dataset, num_batches):
        self.update_fisher_params(dataset, num_batches)
        self.update_mean_params()
    
    def loss(self, y_true, y_pred):
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        cons_loss = self.compute_consolidation_loss()
        loss = cons_loss + cross_entropy(y_true, y_pred)
        return loss
    
    def compute_consolidation_loss(self):
        if not self.estimated_mean or not self.estimated_fisher:
            return 0
        else:
            losses = []
            for param in self.trainable_variables:
                estimated_mean = self.estimated_mean[param.name]
                estimated_fisher = self.estimated_fisher[param.name]
                losses.append(tf.math.reduce_sum(tf.math.multiply(estimated_fisher, tf.math.square(param - estimated_mean))))
            return (self.weight / 2) * tf.math.reduce_sum(losses)