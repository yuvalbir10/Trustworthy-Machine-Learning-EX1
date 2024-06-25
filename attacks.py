import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
        performs random initialization and early stopping, depending on the 
        self.rand_init and self.early_stop flags.
        """
        # Initialize the adversarial samples
        adv_samples = x.clone().detach()
        
        # Randomly initialize if rand_init is True
        if self.rand_init:
            adv_samples += torch.empty_like(adv_samples).uniform_(-self.eps, self.eps)
            adv_samples = torch.clamp(adv_samples, 0, 1)
        
        # Iterate for n attack iterations
        for iteration in range(self.n):
            adv_samples.requires_grad = True

            # Forward pass to get the model predictions
            outputs = self.model(adv_samples)
            
            
            # Calculate the loss
            if targeted:
                # maximize the loss for targeted attacks so it will be closer to the target (to zero loss)
                loss = -self.loss_func(outputs, y) 
            else:
                loss = self.loss_func(outputs, y)
            
            sum_loss = torch.sum(loss)

            # Calculate the gradients
            grad = torch.autograd.grad(
                sum_loss, adv_samples, retain_graph=False, create_graph=False
            )[0]

            # Update the adversarial samples using the gradients
            with torch.no_grad():
                adv_samples += self.alpha * grad.sign()
                perturbations = torch.clamp(adv_samples - x, min=-self.eps, max=self.eps)
                adv_samples = torch.clamp(x + perturbations, 0, 1)
            self.model.zero_grad()
            
            # Check if early stopping is enabled and the attack goal is met

            if self.early_stop:
                should_stop = False
                if not targeted and torch.all(torch.ne(torch.argmax(outputs, dim=1), y)):
                    should_stop = True
                elif targeted and torch.all(torch.eq(torch.argmax(outputs, dim=1), y)):
                    should_stop = True
                
                if should_stop:
                    print(f"Early stopping targeted={targeted}, iteration={iteration}") # TODO: remove this print
                    break
            elif iteration == self.n - 1: # TODO: todel this 'if' block
                print(f"Max iterations reached, no early stopping")
        
        # TODO: use assertions to ensure the adversarial images are valid images and lie within the -ball centered at their benign counterparts

        return adv_samples


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255., momentum=0.,
                 k=200, sigma=1 / 255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma = sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        
    def nes_gradient(self, x, y, targeted):
        """Estimate the gradient using NES with antithetic sampling."""
        # Initialize the gradient
        gradient = torch.zeros_like(x)
        
        # Perform antithetic sampling
        for i in range(self.k):
            # Sample delta_i from N(0, I)
            # TODO: check if this really yields delta_i from N(0, I)
            delta = torch.randn_like(x)
            theta_pos = x + self.sigma * delta
            theta_neg = x - self.sigma * delta
            
            with torch.no_grad():
                # Calculate the loss at theta_pos and theta_neg
                loss_pos = self.loss_func(self.model(theta_pos), y)
                loss_neg = self.loss_func(self.model(theta_neg), y)
                
                # Calculate the gradient estimate
                if targeted:
                    loss_diff = loss_neg - loss_pos
                else:
                    loss_diff = loss_pos - loss_neg
                
                gradient += loss_diff.view(-1, *[1] * (x.dim() - 1)) * delta
        
        # Return the averaged gradient
        gradient /= (2 * self.k * self.sigma)
        
        return gradient

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
            and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
            each sample in x.
        """
        x_adv = x.clone().detach()
        if self.rand_init:
            x_adv += torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)
        
        momentum = torch.zeros_like(x)
        queries = torch.zeros(x.size(0), dtype=torch.int32)

        for _ in range(self.n):
            grad = self.nes_gradient(x_adv, y, targeted)

            if self.momentum > 0: # TODO: check that the momentum handling is correct
                momentum = self.momentum * momentum + grad
                grad = momentum

            x_adv = x_adv + self.alpha * torch.sign(grad) # TODO: check if that is the correct way to update using step size (alpha)
            x_adv = torch.clamp(x_adv, x - self.eps, x + self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)

            queries += 2 * self.k

            with torch.no_grad():
                logits = self.model(x_adv)
                preds = logits.argmax(dim=1)
                if targeted:
                    result_success_status = torch.eq(preds, y)
                else:
                    result_success_status = torch.ne(preds, y)

            if self.early_stop and all(result_success_status):
                break
            

        return x_adv, queries


class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """

    def __init__(self, models, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        # Initialize the adversarial samples
        adv_samples = x.clone().detach()
        
        # Randomly initialize if rand_init is True
        if self.rand_init:
            adv_samples += torch.empty_like(adv_samples).uniform_(-self.eps, self.eps)
            adv_samples = torch.clamp(adv_samples, 0, 1)
        
        # Iterate for n attack iterations
        for iteration in range(self.n):
            adv_samples.requires_grad = True

            # Forward pass to get the model predictions
            models_outputs = [model(adv_samples) for model in self.models]
            
            # Calculate the loss
            if targeted:
                # maximize the loss for targeted attacks so it will be closer to the target (to zero loss)
                loss = (-1) * sum([self.loss_func(outputs, y) for outputs in models_outputs])
            else:
                loss = sum([self.loss_func(outputs, y) for outputs in models_outputs])
            
            sum_loss = torch.sum(loss)

            # Calculate the gradients
            grad = torch.autograd.grad(
                sum_loss, adv_samples, retain_graph=False, create_graph=False
            )[0]

            # Update the adversarial samples using the gradients
            with torch.no_grad():
                adv_samples += self.alpha * grad.sign()
                perturbations = torch.clamp(adv_samples - x, min=-self.eps, max=self.eps)
                adv_samples = torch.clamp(x + perturbations, 0, 1)
            
            for model in self.models:
                model.zero_grad()
            
            # This is not even running in main_b.py, but I implemented it anyway in the way I think it should be
            # Check if early stopping is enabled and the attack goal is met.
            if self.early_stop:
                should_stop = False
                if not targeted and all([torch.all(torch.ne(torch.argmax(outputs, dim=1), y)) for outputs in models_outputs]):
                    should_stop = True
                elif targeted and all([torch.all(torch.eq(torch.argmax(outputs, dim=1), y)) for outputs in models_outputs]):
                    should_stop = True
                
                if should_stop:
                    break
        
        # TODO: use assertions to ensure the adversarial images are valid images and lie within the -ball centered at their benign counterparts

        return adv_samples
