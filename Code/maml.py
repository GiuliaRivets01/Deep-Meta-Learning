import torch
import torch.nn as nn

from networks import Conv4


class MAML(nn.Module):

    def __init__(self, num_ways, input_size, T=10, second_order=False, inner_lr=0.4, **kwargs):
        super().__init__()
        self.num_ways = num_ways
        self.input_size = input_size
        self.num_updates = T
        self.second_order = second_order
        self.inner_loss = nn.CrossEntropyLoss()
        self.inner_lr = inner_lr

        self.network = Conv4(self.num_ways, img_size=int(input_size ** 0.5))

        # controller input = image + label_previous

    def apply(self, x_supp, y_supp, x_query, y_query, training=False):
        """
        Pefrosmt the inner-level learning procedure of MAML: adapt to the given task
        using the support set. It returns the predictions on the query set, as well as the loss
        on the query set (cross-entropy).
        You may want to set the gradients manually for the base-learner parameters

        :param x_supp (torch.Tensor): the support input iamges of shape (num_support_examples, num channels, img width, img height)
        :param y_supp (torch.Tensor): the support ground-truth labels
        :param x_query (torch.Tensor): the query inputs images of shape (num_query_inputs, num channels, img width, img height)
        :param y_query (torch.Tensor): the query ground-truth labels

        :returns:
          - query_preds (torch.Tensor): the predictions of the query inputs
          - query_loss (torch.Tensor): the cross-entropy loss on the query inputs
        """

        # TODO: implement this function

        # Note: to make predictions and to allow for second-order gradients to flow if we want,
        # we use a custom forward function for our network. You can make predictions using
        # preds = self.network(input_data, weights=<the weights you want to use>)
        # Create a copy of the initial parameters
        fast_weights = [param.clone() for param in self.network.parameters()]

        ''' INNER TRAINING (support set)
        Use support set from sampled task to update the model's parameters '''
        for _ in range(self.num_updates):
            ''' 1. Compute loss on support set '''
            # Loss on support set = inner loss
            supp_preds = self.network(x_supp, weights=fast_weights)
            supp_loss = self.inner_loss(supp_preds, y_supp)

            ''' 2. Perform a few gradient update steps and gradient descent steps to 
                update the parameters in the direction tha minimizes the support set loss '''
            # Gradient Update steps
            grads = torch.autograd.grad(supp_loss, fast_weights, create_graph=self.second_order)
            # Compute adapted paramters with gradient descent
            fast_weights = [weight - self.inner_lr * grad for weight, grad in zip(fast_weights, grads)]

        ''' META-UPDATE '''
        ''' 1. After the inner training on the support set, compute the gradient of the loss 
            on the query set with respect to the initial parameters '''
        query_preds = self.network(x_query, weights=fast_weights)
        query_loss = self.inner_loss(query_preds, y_query)

        ''' 2. Update the initial parameters based on these gradients '''

        if training:
            query_loss.backward()

        return query_preds, query_loss

        raise NotImplementedError()




