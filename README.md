# Disease

## Heart Disease Api
```python
import requests

data =  {
    "data" : [ 63.0,   1.0 ,   3.0 , 145.0 , 233.0 ,   1.0 ,   0.0 , 150.0 ,   0.0 ,
         2.3,   0.0 ,   0.0 ,   1.0 ]
}
res = requests.post(url = "https://diseaseapp1.herokuapp.com/pred1/", json = data)

res.json()

```
