---
published: true
---
In here, we have a PyTorch short training model doodle:

![train](https://raw.githubusercontent.com/dejanbatanjac/dejanbatanjac.github.io/master/images/train.png)

Most notable is the `forward` pass where we step into the model and predict the `y_hat`.
We can write this more formal:

`y_hat = model(X)`

Read: Compute predicted `y_hat` by passing X to the model.
We define a model for instance:

~~~
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          # more layers in here ...
          torch.nn.Linear(H, D_out),
        )
~~~        
The model can be either:
* [torch.nn.Sequential](https://pytorch.org/docs/stable/nn.html#torch-nn-sequential)
* [torch.nn.Functional](https://pytorch.org/docs/stable/nn.html#torch-nn-functional)

What we get from the model forward phase is the `y_hat`, or the prediction.

We use that prediction to create the `loss`, using the `loss_fn` (the loss function)

`loss = loss_fn(y_hat, y)`

Sum of all loss values we call the error but we may also say the loss again.
One of the most common used loss function is the Euclid distance loss:

`loss_fn = torch.nn.MSELoss()`

Once we found the `loss` we exete the `loss.backward()`. This is an automatic gradient calculation phase. PyTorch has the automatic gradient calculus right out of the box. 

The `loss.backward()` will calculate the gradients. Gradients are needed in the next phase, when using the `optimizer.step()` function we fine tune our model parameters.

Just to add, we can anytime get all our model parameters as : `model.parameters()` method.
Once we updated the parameters, we repeat the forward phase again.

We used the `torch.nn.Lineear()` layer in our example, which assumes the linear transformation of our input. Usually some non-linear transformation follows the `Linear` PyTorch layer since we know neural networks learn best when we apply non-linear function to the input.

Note: Each PyTorch forward phase creates the calculation graph; thanks to that graph we compute the gradients...


