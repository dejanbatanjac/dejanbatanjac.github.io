---
published: true
---
## Terminology first

Time series, as the name imply, it is a distribution of a value in time.

In ML we can identify the following things:
* the sequence
* the window
* the samples (observations)
* number of samples
* the batch of input observations
* the input features
* the hidden features 
* iterations
* time steps

Oh my, so many of these!

## No concept of input and output

In time series there is no concept of input and output. At least, not before we declare the model and specify them.

Let's check on this example:

If we have a time series: 0.1, 0.2, 0.3, 0.4, 0.5, we may say:

**(i)5 examples, one time step each
~~~
t=1: 0.1
t=2: 0.2
t=3: 0.3
t=4: 0.4
t=5: 0.5
~~~

**(ii)4 examples, two time step each**
~~~
t=1: 0.1 0.2
t=2: 0.2 0.3
t=3: 0.3 0.4
t=4: 0.4 0.5
~~~

**(iii)3 examples, three time step each**
~~~
t1: 0.1 0.2 0.3
t2: 0.2 0.3 0.4
t3: 0.3 0.4 0.5
~~~

**(iv)2 examples, 4 time step each**

~~~
t1: 0.1 0.2 0.3 0.4
t2: 0.2 0.3 0.4 0.5
~~~

**(v)One example, 5 time steps**

~~~
t1: 0.1 0.2 0.3 0.4 0.5
~~~


>> Here you  can check our freedom, to declare the time seria as we like.

Similar, there are no constraint to define what is the input and what is the output.
Input variables are also called features. Or input features.

Let's say for the example (v):

    t1: 0.1 0.2 0.3 0.4 0.5
    
Input is the whole sequence: `0.1 0.2 0.3 0.4 0.5`
Output is the last value `0.5`.
But output may be also the value `0.4`. Or the value just before the last. (lag output).
Output may also be the first value `0.1`.
Output may also be the first two values `0.1 0.2` or last three `0.3 0.4 0.5`

It is up to our choice.

Notice how we can based on the sequence (v)`t1: 0.1 0.2 0.3 0.4 0.5` define the:
* input as `0.1 0.2 0.3 0.4` 
* and the output as `0.5`

Hopefully this provides some clue that the freedom is on your end, and that you can interpret the sequence (time series) as you like.

---

Let me try to provide some feedback on the terminology:

The sequence would be a sequence of some values, based on the original time series. But generally, it may be applied to any data we can sort.

The window is a sequential sequence subset. For the sequence `0.1 0.2 0.3 0.4 0.5` the window may be `0.1 0.2 0.3`

The samples or observations, are particular elements of the sequence.

The number of samples, is the total number of samples we created based on sequence. This is up to use, as it defines our freedom of choosing the samples.
The max total number of samples is equal to the number of elements in a time series. 
The min total number of samples is 1, like in example (v).


The batch of input observations. As we said the input observations or samples is something we define freely. We can batch these samples, to be computationally efficient. 

The input features. Input variables are also called the input features. We define what is the input and what is the output. The whole window (as we just defined a window) can be a feature.

The hidden features. In our model we may have so called hidden features. We typically get these hidden features `h` once we multiply the input features with the tensor of weights. `h` = `i`* `W`.

The iterations = How many batches to complete the one epoch.
One epoch is when we compute all the training examples, or all the examples.
