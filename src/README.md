# RL Agent Notes and Setup


## To-Do

|Item #| Description                                                        | Done|
|:---: | :---------------------------------------------------------------   |:---:|
|1     | Normalise inputs and outputs to neural network                     |     |
|2     | Add lstm input to network                                          | Yes |
|3     | Add sharpe ratio as the models reward                              |     |
|4     | Integrate Advantage Model                                          |     |
|5     | Integrate Critic Model                                             |     |
|6     | Add multiple training agents                                       |     |
|7     | Train on exit with random entry (learn to exit)                    | WIP |
|8     | Train on a batch of trades (learn to maximise over several trades) |     |
|9     | Train with different market data                                   |     |
|10    | Refactor code into simpler classes                                 |     |
|11    | Fix bug with first few ticks of visualisation missing              |     |

## How to show the performance of the model

```bash
$ bokeh serve --show performance
$ python agent.py
```

![Bokeh](../misc/bokeh_performance.PNG)
