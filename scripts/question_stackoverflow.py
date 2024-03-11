Hi,

I was playing with numerical simulations when I observed a surprising result. So surprising indeed that I suspect I made a mistake, but I cannot put my finger on where.

I simulated picking numbers from a random uniform distribution between 0 and 1, and computing $m$ different quantiles. As expected, as the number of simulations gets bigger, the value of the $i$th $m$-quantile approaches $i/m$. What surprised me was the convergence rate.

Specifically, here is the code I used to compute the error between the estimator and the exact value:

```python
def uniform_quantiles_error(n,m):
    
    vec = np.random.random((n+1))
    vec.sort()
    
    return np.array([abs(vec[(n // m)*i] - i / m) for i in range(m+1)])
```

Here is a plot of the generated values:
[![enter image description here][1]][1]

I estimated the convergence rates, and these were the results:


  [1]: https://i.stack.imgur.com/9Z4X5.png