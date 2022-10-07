## Pandas 相關程式碼解

### Filter
`.filter` 可以篩選出特定想關注之column
```
data = df2.filter(['close'])
```

`.filter` 也可以一次性篩選出多筆想關注的column
```
data = df2.filter(['close'])
```
### Value

`.value` 可以將Dataframe轉換成  2D numpy array
```
dataset = data.values
```

<br><br>

## Matplotlib
Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. Matplotlib makes easy things easy and hard things possible.  
使用Matplotlib時需先`import matplotlib.pyplot as plt`





## 其他想關程式碼解釋
### Math
`math.ceil` 可以向上取整數也就是說會回傳一無條件進位的結果

math無法直接呼叫使用的時後需係`import math`
```
training_data_len = math.ceil(len(dataset)*0.7)
```
