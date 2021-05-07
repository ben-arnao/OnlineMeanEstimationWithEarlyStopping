# OnlineMeanEstimationWithEarlyStopping

Example usage

```python
e = Estimator(1, min_param_change_perc=1, param_change_delta=0.1, param_change_patience=200, momentum=0.995)
ma_hist = []
complete = []

for x in range(3000):
    ma_hist.append(e.mov_avg)
    val = np.random.normal(loc=5, scale=1)
    complete.append(e.estimate(val))

pyplot.subplot(211)
pyplot.plot(ma_hist)
pyplot.subplot(212)
pyplot.plot(complete)
pyplot.show()
```

![demo](https://i.imgur.com/E5pXqou.png)
