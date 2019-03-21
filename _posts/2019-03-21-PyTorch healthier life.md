---
published: true
---
**Create things on GPU**
Consider these two lines:

    torch.zeros(100, device="gpu")
    torch.zeros(100).to("cuda")

They should effect the same, but first one is faster as it assumes creating GPU tensor directly without copy from CPU, while the second one uses the copy from CPU trick.

**Two different loss functions**
If you have two different loss functions, finish the forwards for both of them separately, and then finally you can do `(loss1 + loss2).backward()`. 
It's a bit more efficient, skips quite some computation.


**Sum the loss**
In your code you want to do: 

    loss_sum += loss.item()

to make sure you do not keep track of the history of all your losses. 

`item()` will break the graph and thus allow it to be freed from one iteration of the loop to the next.
Also you could use `detach()` for the same.

**Loss backward and DataParallel**
When you do `loss.backward()`, it is a shortcut for `loss.backward(torch.Tensor([1]))`. 
This in only valid if loss is a tensor containing a single element.

DataParallel returns to you the partial loss that was computed on each GPU, so you usually want to do `loss.backward(torch.Tensor([1, 1]))` or `loss.sum().backward()`. 
Both will have the exact same behaviour.


**Disable the autograd**
If you want to disable the autograd, you should wrap you function in a with `torch.no_grad()` block.
Based on the PyTorch tutorial, during prediction (after training and evaluation phase), we are supposed to do something like

    model.eval()
    with torch.no_grad():

**Vector and matrix multiplication**
In PyTorch we use tensors. What if we need to do element vise multiplication?

    x=torch.rand(2,3,2,2)
    y=torch.rand(3)
    print(x)
    print(y)
    y = y.view(1,3,1,1)
    out = x*y
    print(out)

We use `view()`. Note view is in place operation:

    x*y.view(1, 3, 1, 1)
		
Would do the same as:

    y = y.view(1,3,1,1)
    out = x*y

**Whats the difference between view() and expand()?**
Expand allows you to repeat a tensor along a dimension of size 1.
View changes the size of the Tensor without changing the number of elements in it.

**Use model on GPU**
You should create your model as usual then call `model.cuda()` to send all parameters of this model (and other stuff in the model) to the GPU. 
Then you need to make sure that your inputs are on the gpu as well:

    input = input.cuda()

Then you can forward on the GPU by doing: 

    model(input)
		
Note that `model.cuda()` will change the model inplace while `input.cuda()` will not change input inplace and you need to do `input = input.cuda()`.


**Replace `outupt.data`**
Replace `output.data` with `output.detach()` that is the new way to do it.



**CUDA_LAUNCH_BLOCKING=1 to have clear stack trace**
When running on GPU, use `CUDA_LAUNCH_BLOCKING=1` otherwise the stack trace is going to be wrong.
