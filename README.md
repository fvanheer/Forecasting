# Commodity Price Forecasting

Forecasting is such an essential part of business and everyday life. It is in your weather app and your business plan. I did some research and discovered that Facebook open sourced their time series forecasting model Prophet. [Read More](https://facebook.github.io/prophet/)

## Getting Started

* [Python 3.7](https://www.python.org/downloads/release/python-370/) - More info
* [prophet installation guide](https://facebook.github.io/prophet/docs/installation.html) - More Info 

### Libraries to install 

```
pip install -r requirements.txt
```

or pip install 
```
pandas
glob2
numpy
fbprophet
plotly
```
You might find that fbprohet does not install via pip then you can run a conda-forge:
```
conda install -c conda-forge fbprophet
```

### Download the data

* [Time Series Minimum Temp 1981-1991](https://www.kaggle.com/shenba/time-series-datasets)

### The Model

You can find the code in the model model.py file

## Import the libraries and data

```Python
import pandas as pd 
import numpy as np
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import datetime as dt
```

```Python

```
![Min Temp Plot]('https://github.com/fvanheer/Forecasting/blob/master/images/min_temp_trend.png')
## Functions

```Python

```

## Author

* **Francois van Heerden** - *Experience* - [LinkedIn Profile](https://www.linkedin.com/in/francois-van-heerden-9589825a/)

## Acknowledgments

* Found inspiration from multiple fellow Data Scientists in the open source community