2. DataFrame 선언하기


```python
import pandas as pd
import numpy as np

dataset = np.array([['kor', 70], ['math', 80]])
df = pd.DataFrame(dataset, columns = ['class', 'score'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>kor</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>math</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
</div>



4. DataFrame 출력


```python
# !pip install scikit-learn
from sklearn.datasets import load_iris
iris = load_iris()
iris = pd.DataFrame(iris.data, columns = iris.feature_names)
```


```python
iris.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 4 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   sepal length (cm)  150 non-null    float64
     1   sepal width (cm)   150 non-null    float64
     2   petal length (cm)  150 non-null    float64
     3   petal width (cm)   150 non-null    float64
    dtypes: float64(4)
    memory usage: 4.8 KB
    


```python
iris.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.057333</td>
      <td>3.758000</td>
      <td>1.199333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.435866</td>
      <td>1.765298</td>
      <td>0.762238</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>



6. DataFrame 인덱스


```python
# df.index
print(df)
print(df.index)
print(list(df.index))
df.index = ['A', 'B']
print(df)
```

      class score
    0   kor    70
    1  math    80
    RangeIndex(start=0, stop=2, step=1)
    [0, 1]
      class score
    A   kor    70
    B  math    80
    


```python
# df.set_index
df.set_index('class', drop=True, append=False, inplace=True)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
    </tr>
    <tr>
      <th>class</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>kor</th>
      <td>70</td>
    </tr>
    <tr>
      <th>math</th>
      <td>80</td>
    </tr>
  </tbody>
</table>
</div>



7. DataFrame 컬럼명 확인 및 변경


```python
print(iris.columns)
iris.columns = ['sepal length', 'sepal width', 'petal length', 'petal width']
iris.head()
```

    Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
           'petal width (cm)'],
          dtype='object')
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length</th>
      <th>sepal width</th>
      <th>petal length</th>
      <th>petal width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris.columns = iris.columns.str.replace(' ', '_')
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



8. DataFrame 컬럼의 데이터 타입 확인 및 변경
- int, float, bool, datetime, category, object


```python
print(iris.dtypes)
```

    sepal_length    float64
    sepal_width     float64
    petal_length    float64
    petal_width     float64
    dtype: object
    


```python
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# astype을 이용하여 데이터 타입 변경
iris['sepal_length'] = iris['sepal_length'].astype('int')
iris[['sepal_width', 'petal_length']] = \
iris[['sepal_width', 'petal_length']].astype('int')

iris.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



### 제3절. row/column 선택, 추가, 삭제

1. row/column 선택


```python
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
iris = pd.DataFrame(iris.data, columns = iris.feature_names)
```


```python
# 행 선택
iris[1:4]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 열 선택
iris['sepal length (cm)'].head(4)
iris[['sepal width (cm)', 'sepal length (cm)']].head(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal width (cm)</th>
      <th>sepal length (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.5</td>
      <td>5.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.1</td>
      <td>4.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# iloc: integer location
iris.iloc[1:4]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris.iloc[[1, 3, 5], 2:4]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.7</td>
      <td>0.4</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris.iloc[:, [True, True, False, True]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>1.8</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 3 columns</p>
</div>




```python
# loc: location
iris.loc[1:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris.loc[[1, 2], 'sepal length (cm)':'petal length (cm)']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 선택한 값 변경하기
score = pd.DataFrame({'국어': [100, 80],
                      '수학': [75, 90], 
                      '영어': [90, 95], 
                      }, 
                      index = ['장화', '홍련'])
```


```python
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>국어</th>
      <th>수학</th>
      <th>영어</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>장화</th>
      <td>100</td>
      <td>75</td>
      <td>90</td>
    </tr>
    <tr>
      <th>홍련</th>
      <td>80</td>
      <td>90</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>




```python
# iloc으로도 되고
score.iloc[0, 1] = 100

# loc으로도 됨
score.loc['장화', '수학'] = 65
```


```python
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>국어</th>
      <th>수학</th>
      <th>영어</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>장화</th>
      <td>100</td>
      <td>65</td>
      <td>90</td>
    </tr>
    <tr>
      <th>홍련</th>
      <td>80</td>
      <td>90</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_students
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>국어</th>
      <th>수학</th>
      <th>영어</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>콩쥐</th>
      <td>70</td>
      <td>65</td>
      <td>96</td>
    </tr>
    <tr>
      <th>팥쥐</th>
      <td>85</td>
      <td>100</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>국어</th>
      <th>수학</th>
      <th>영어</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>장화</th>
      <td>100</td>
      <td>65</td>
      <td>90</td>
    </tr>
    <tr>
      <th>홍련</th>
      <td>80</td>
      <td>90</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_students = pd.DataFrame({'국어': [70, 85], 
                             '수학': [65, 100], 
                             '영어': [96, 65]}, 
                            index = ['콩쥐', '팥쥐'])
# append 사라짐
score = pd.concat([score, new_students])
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>국어</th>
      <th>수학</th>
      <th>영어</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>장화</th>
      <td>100</td>
      <td>65</td>
      <td>90</td>
    </tr>
    <tr>
      <th>홍련</th>
      <td>80</td>
      <td>90</td>
      <td>95</td>
    </tr>
    <tr>
      <th>콩쥐</th>
      <td>70</td>
      <td>65</td>
      <td>96</td>
    </tr>
    <tr>
      <th>팥쥐</th>
      <td>85</td>
      <td>100</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
science = [80, 70, 60, 50]
score['과학'] = science
score['학년'] = 1
```


```python
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>국어</th>
      <th>수학</th>
      <th>영어</th>
      <th>과학</th>
      <th>학년</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>장화</th>
      <td>100</td>
      <td>65</td>
      <td>90</td>
      <td>80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>홍련</th>
      <td>80</td>
      <td>90</td>
      <td>95</td>
      <td>70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>콩쥐</th>
      <td>70</td>
      <td>65</td>
      <td>96</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>팥쥐</th>
      <td>85</td>
      <td>100</td>
      <td>65</td>
      <td>50</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
score['과학'] = score['과학'] + 5
score['총점'] = score['국어'] + score['수학'] + score['영어'] + score['과학']
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>국어</th>
      <th>수학</th>
      <th>영어</th>
      <th>과학</th>
      <th>학년</th>
      <th>총점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>장화</th>
      <td>100</td>
      <td>65</td>
      <td>90</td>
      <td>90</td>
      <td>1</td>
      <td>345</td>
    </tr>
    <tr>
      <th>홍련</th>
      <td>80</td>
      <td>90</td>
      <td>95</td>
      <td>80</td>
      <td>1</td>
      <td>345</td>
    </tr>
    <tr>
      <th>콩쥐</th>
      <td>70</td>
      <td>65</td>
      <td>96</td>
      <td>70</td>
      <td>1</td>
      <td>301</td>
    </tr>
    <tr>
      <th>팥쥐</th>
      <td>85</td>
      <td>100</td>
      <td>65</td>
      <td>60</td>
      <td>1</td>
      <td>310</td>
    </tr>
  </tbody>
</table>
</div>



3. row/column 삭제


```python
print(score)
score.drop('장화', inplace=True)
score.drop(columns=['과학', '학년', '총점'], inplace=True)
print(score)
```

         국어   수학  영어  과학  학년   총점
    장화  100   65  90  90   1  345
    홍련   80   90  95  80   1  345
    콩쥐   70   65  96  70   1  301
    팥쥐   85  100  65  60   1  310
    

### 제4절.조건에 맞는 데이터 탐색 및 수정


```python
import pandas as pd
names = ['장화', '홍련', '콩쥐', '팥쥐', '해님', '달님']
korean_scores = [70, 85, None, 100, None, 85]
math_scores = [65, 100, 80, 95, None, 70]
students = pd.DataFrame({'이름': names, 
                         '국어': korean_scores, 
                         '수학': math_scores})

students[students['이름']=='장화']

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>국어</th>
      <th>수학</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>장화</td>
      <td>70.0</td>
      <td>65.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
students[(students['국어']>=80) & (students['수학']>=80)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>국어</th>
      <th>수학</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>홍련</td>
      <td>85.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>팥쥐</td>
      <td>100.0</td>
      <td>95.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
students.loc[6, '이름':'수학'] = ['별님', 50, 60]
students
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>국어</th>
      <th>수학</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>장화</td>
      <td>70.0</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>홍련</td>
      <td>85.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>콩쥐</td>
      <td>NaN</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>팥쥐</td>
      <td>100.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>해님</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>달님</td>
      <td>85.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>별님</td>
      <td>50.0</td>
      <td>60.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
students.loc[(students['국어']>=80)&(students['수학']>=70), '합격'] = 'Pass'
students
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>국어</th>
      <th>수학</th>
      <th>합격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>장화</td>
      <td>70.0</td>
      <td>65.0</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>홍련</td>
      <td>85.0</td>
      <td>100.0</td>
      <td>Pass</td>
    </tr>
    <tr>
      <th>2</th>
      <td>콩쥐</td>
      <td>NaN</td>
      <td>80.0</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>팥쥐</td>
      <td>100.0</td>
      <td>95.0</td>
      <td>Pass</td>
    </tr>
    <tr>
      <th>4</th>
      <td>해님</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>5</th>
      <td>달님</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>Pass</td>
    </tr>
    <tr>
      <th>6</th>
      <td>별님</td>
      <td>50.0</td>
      <td>60.0</td>
      <td>nan</td>
    </tr>
  </tbody>
</table>
</div>




```python
students.loc[students['합격']!='Pass', '합격'] = 'Fail'
students
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>국어</th>
      <th>수학</th>
      <th>합격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>장화</td>
      <td>70.0</td>
      <td>65.0</td>
      <td>Fail</td>
    </tr>
    <tr>
      <th>1</th>
      <td>홍련</td>
      <td>85.0</td>
      <td>100.0</td>
      <td>Pass</td>
    </tr>
    <tr>
      <th>2</th>
      <td>콩쥐</td>
      <td>NaN</td>
      <td>80.0</td>
      <td>Fail</td>
    </tr>
    <tr>
      <th>3</th>
      <td>팥쥐</td>
      <td>100.0</td>
      <td>95.0</td>
      <td>Pass</td>
    </tr>
    <tr>
      <th>4</th>
      <td>해님</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Fail</td>
    </tr>
    <tr>
      <th>5</th>
      <td>달님</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>Pass</td>
    </tr>
    <tr>
      <th>6</th>
      <td>별님</td>
      <td>50.0</td>
      <td>60.0</td>
      <td>Fail</td>
    </tr>
  </tbody>
</table>
</div>




```python
students
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>국어</th>
      <th>수학</th>
      <th>합격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>장화</td>
      <td>70.0</td>
      <td>65.0</td>
      <td>Fail</td>
    </tr>
    <tr>
      <th>1</th>
      <td>홍련</td>
      <td>85.0</td>
      <td>100.0</td>
      <td>Pass</td>
    </tr>
    <tr>
      <th>2</th>
      <td>콩쥐</td>
      <td>NaN</td>
      <td>80.0</td>
      <td>Fail</td>
    </tr>
    <tr>
      <th>3</th>
      <td>팥쥐</td>
      <td>100.0</td>
      <td>95.0</td>
      <td>Pass</td>
    </tr>
    <tr>
      <th>4</th>
      <td>해님</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Fail</td>
    </tr>
    <tr>
      <th>5</th>
      <td>달님</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>Pass</td>
    </tr>
    <tr>
      <th>6</th>
      <td>별님</td>
      <td>50.0</td>
      <td>60.0</td>
      <td>Fail</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 하나의 컬럼에 여러 개의 조건을 두어야 할 때
import numpy as np
condition_list = [(students['국어'] >= 90), # --> A
                  (students['국어'] >= 80) & (students['국어'] < 90),  # --> B
                  (students['국어'] >= 70) & (students['국어'] < 80)]  # --> C
choice_list = ['A', 'B', 'C']
students['점수'] = np.select(condition_list, choice_list, default='F')
students
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>국어</th>
      <th>수학</th>
      <th>합격</th>
      <th>점수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>장화</td>
      <td>70.0</td>
      <td>65.0</td>
      <td>Fail</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>홍련</td>
      <td>85.0</td>
      <td>100.0</td>
      <td>Pass</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>콩쥐</td>
      <td>NaN</td>
      <td>80.0</td>
      <td>Fail</td>
      <td>F</td>
    </tr>
    <tr>
      <th>3</th>
      <td>팥쥐</td>
      <td>100.0</td>
      <td>95.0</td>
      <td>Pass</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>해님</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Fail</td>
      <td>F</td>
    </tr>
    <tr>
      <th>5</th>
      <td>달님</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>Pass</td>
      <td>B</td>
    </tr>
    <tr>
      <th>6</th>
      <td>별님</td>
      <td>50.0</td>
      <td>60.0</td>
      <td>Fail</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 결측값 탐색 및 수정
print(students.isna())
print(students.isna().sum())
print(students.isna().sum(1))
```

          이름     국어     수학     합격     점수
    0  False  False  False  False  False
    1  False  False  False  False  False
    2  False   True  False  False  False
    3  False  False  False  False  False
    4  False   True   True  False  False
    5  False  False  False  False  False
    6  False  False  False  False  False
    이름    0
    국어    2
    수학    1
    합격    0
    점수    0
    dtype: int64
    0    0
    1    0
    2    1
    3    0
    4    2
    5    0
    6    0
    dtype: int64
    


```python
students.notna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>국어</th>
      <th>수학</th>
      <th>합격</th>
      <th>점수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# dropna
students.dropna()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>국어</th>
      <th>수학</th>
      <th>합격</th>
      <th>점수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>장화</td>
      <td>70.0</td>
      <td>65.0</td>
      <td>Fail</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>홍련</td>
      <td>85.0</td>
      <td>100.0</td>
      <td>Pass</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>팥쥐</td>
      <td>100.0</td>
      <td>95.0</td>
      <td>Pass</td>
      <td>A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>달님</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>Pass</td>
      <td>B</td>
    </tr>
    <tr>
      <th>6</th>
      <td>별님</td>
      <td>50.0</td>
      <td>60.0</td>
      <td>Fail</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
</div>




```python
students.dropna(thresh=4) # 결측값이 아닌 값이 4개보다 많은 행만 남기기
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>국어</th>
      <th>수학</th>
      <th>합격</th>
      <th>점수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>장화</td>
      <td>70.0</td>
      <td>65.0</td>
      <td>Fail</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>홍련</td>
      <td>85.0</td>
      <td>100.0</td>
      <td>Pass</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>콩쥐</td>
      <td>NaN</td>
      <td>80.0</td>
      <td>Fail</td>
      <td>F</td>
    </tr>
    <tr>
      <th>3</th>
      <td>팥쥐</td>
      <td>100.0</td>
      <td>95.0</td>
      <td>Pass</td>
      <td>A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>달님</td>
      <td>85.0</td>
      <td>70.0</td>
      <td>Pass</td>
      <td>B</td>
    </tr>
    <tr>
      <th>6</th>
      <td>별님</td>
      <td>50.0</td>
      <td>60.0</td>
      <td>Fail</td>
      <td>F</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 결측값 대체
health = pd.DataFrame({'연도': [2017, 2018, 2019, 2020, 2021, 2022] 
                       ,'키': [160, 162, 165, None, None, 166]
                       ,'몸무게': [53, 52, None, 50, 51, 54]
                       ,'시력': [1.2, None, 1.2, 1.2, 1.1, 0.8]
                       ,'병결': [None, None, None, 2, None, 1]})
health
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>연도</th>
      <th>키</th>
      <th>몸무게</th>
      <th>시력</th>
      <th>병결</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>160.0</td>
      <td>53.0</td>
      <td>1.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>162.0</td>
      <td>52.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>165.0</td>
      <td>NaN</td>
      <td>1.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020</td>
      <td>NaN</td>
      <td>50.0</td>
      <td>1.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021</td>
      <td>NaN</td>
      <td>51.0</td>
      <td>1.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022</td>
      <td>166.0</td>
      <td>54.0</td>
      <td>0.8</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
health.fillna(0) # 결측값 모두 0으로 채움
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>연도</th>
      <th>키</th>
      <th>몸무게</th>
      <th>시력</th>
      <th>병결</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>160.0</td>
      <td>53.0</td>
      <td>1.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>162.0</td>
      <td>52.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>165.0</td>
      <td>0.0</td>
      <td>1.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>1.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021</td>
      <td>0.0</td>
      <td>51.0</td>
      <td>1.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022</td>
      <td>166.0</td>
      <td>54.0</td>
      <td>0.8</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
health.fillna(health.mean())  # None값을 제외하고, 수치형 자료의 평균으로 모든 결측치를 대체
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>연도</th>
      <th>키</th>
      <th>몸무게</th>
      <th>시력</th>
      <th>병결</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>160.00</td>
      <td>53.0</td>
      <td>1.2</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>162.00</td>
      <td>52.0</td>
      <td>1.1</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>165.00</td>
      <td>52.0</td>
      <td>1.2</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020</td>
      <td>163.25</td>
      <td>50.0</td>
      <td>1.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021</td>
      <td>163.25</td>
      <td>51.0</td>
      <td>1.1</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022</td>
      <td>166.00</td>
      <td>54.0</td>
      <td>0.8</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
health['병결'] = health['병결'].fillna(0)
health
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>연도</th>
      <th>키</th>
      <th>몸무게</th>
      <th>시력</th>
      <th>병결</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>160.0</td>
      <td>53.0</td>
      <td>1.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>162.0</td>
      <td>52.0</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>165.0</td>
      <td>NaN</td>
      <td>1.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020</td>
      <td>NaN</td>
      <td>50.0</td>
      <td>1.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021</td>
      <td>NaN</td>
      <td>51.0</td>
      <td>1.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022</td>
      <td>166.0</td>
      <td>54.0</td>
      <td>0.8</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
health['몸무게'] = health['몸무게'].fillna(health['몸무게'].mean())
health
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>연도</th>
      <th>키</th>
      <th>몸무게</th>
      <th>시력</th>
      <th>병결</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>160.0</td>
      <td>53.0</td>
      <td>1.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>162.0</td>
      <td>52.0</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>165.0</td>
      <td>52.0</td>
      <td>1.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020</td>
      <td>NaN</td>
      <td>50.0</td>
      <td>1.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021</td>
      <td>NaN</td>
      <td>51.0</td>
      <td>1.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022</td>
      <td>166.0</td>
      <td>54.0</td>
      <td>0.8</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
health.fillna(method='pad', inplace=True) # 결측값이 나오기 전 바로 앞에 나온 값으로 대체
health

# health['키'].ffill()
# health['시력'].bfill()
```

    C:\Users\thdus\AppData\Local\Temp\ipykernel_27324\2472544055.py:1: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
      health.fillna(method='pad', inplace=True) # 결측값이 나오기 전 바로 앞에 나온 값으로 대체
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>연도</th>
      <th>키</th>
      <th>몸무게</th>
      <th>시력</th>
      <th>병결</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017</td>
      <td>160.0</td>
      <td>53.0</td>
      <td>1.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>162.0</td>
      <td>52.0</td>
      <td>1.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>165.0</td>
      <td>52.0</td>
      <td>1.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020</td>
      <td>165.0</td>
      <td>50.0</td>
      <td>1.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021</td>
      <td>165.0</td>
      <td>51.0</td>
      <td>1.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022</td>
      <td>166.0</td>
      <td>54.0</td>
      <td>0.8</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 중복행 삭제
health['키'].drop_duplicates()
```




    0    160.0
    1    162.0
    2    165.0
    5    166.0
    Name: 키, dtype: float64




```python
health[['시력', '병결']].drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>시력</th>
      <th>병결</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.8</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### 제5절. 데이터 정렬


```python
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
iris = pd.DataFrame(iris.data, columns=iris.feature_names)
iris
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 4 columns</p>
</div>




```python
# index 정렬
iris.sort_index(ascending=False, inplace=True)
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris.sort_index(axis=1, ascending=True, inplace=True) # 컬럼명을 기준으로 sorting
iris.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149</th>
      <td>5.1</td>
      <td>1.8</td>
      <td>5.9</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>5.4</td>
      <td>2.3</td>
      <td>6.2</td>
      <td>3.4</td>
    </tr>
    <tr>
      <th>147</th>
      <td>5.2</td>
      <td>2.0</td>
      <td>6.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>146</th>
      <td>5.0</td>
      <td>1.9</td>
      <td>6.3</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>145</th>
      <td>5.2</td>
      <td>2.3</td>
      <td>6.7</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 값 정렬
iris.sort_values('petal length (cm)', inplace=True)
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>1.0</td>
      <td>0.2</td>
      <td>4.6</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.1</td>
      <td>0.1</td>
      <td>4.3</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.2</td>
      <td>0.2</td>
      <td>5.8</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1.2</td>
      <td>0.2</td>
      <td>5.0</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1.3</td>
      <td>0.4</td>
      <td>5.4</td>
      <td>3.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris.sort_values(['petal length (cm)', 'sepal length (cm)'], inplace=True)
iris
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>1.0</td>
      <td>0.2</td>
      <td>4.6</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1.1</td>
      <td>0.1</td>
      <td>4.3</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1.2</td>
      <td>0.2</td>
      <td>5.0</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.2</td>
      <td>0.2</td>
      <td>5.8</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1.3</td>
      <td>0.2</td>
      <td>4.4</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>131</th>
      <td>6.4</td>
      <td>2.0</td>
      <td>7.9</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>105</th>
      <td>6.6</td>
      <td>2.1</td>
      <td>7.6</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>117</th>
      <td>6.7</td>
      <td>2.2</td>
      <td>7.7</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>122</th>
      <td>6.7</td>
      <td>2.0</td>
      <td>7.7</td>
      <td>2.8</td>
    </tr>
    <tr>
      <th>118</th>
      <td>6.9</td>
      <td>2.3</td>
      <td>7.7</td>
      <td>2.6</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 4 columns</p>
</div>



### 제6절. 데이터 결합

#### 1. 단순 연결


```python
import pandas as pd
HR1 = pd.DataFrame({'이름': ['장화', '홍련'], 
                    '부서': ['영업', '회계'], 
                    '직급': ['팀장', '사원']})
HR2 = pd.DataFrame({'이름': ['콩쥐', '팥쥐'], 
                    '부서': ['사원', '팀장'], 
                    '직급': ['영업', '인사']})
HR3 = pd.DataFrame({'이름': ['콩쥐', '팥쥐'], 
                    '부서': ['영업', '인사'], 
                    '급여': [3500, 2800]})

pd.concat([HR1, HR2], axis=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>부서</th>
      <th>직급</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>장화</td>
      <td>영업</td>
      <td>팀장</td>
    </tr>
    <tr>
      <th>1</th>
      <td>홍련</td>
      <td>회계</td>
      <td>사원</td>
    </tr>
    <tr>
      <th>0</th>
      <td>콩쥐</td>
      <td>사원</td>
      <td>영업</td>
    </tr>
    <tr>
      <th>1</th>
      <td>팥쥐</td>
      <td>팀장</td>
      <td>인사</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([HR1, HR2], axis=0, ignore_index=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>부서</th>
      <th>직급</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>장화</td>
      <td>영업</td>
      <td>팀장</td>
    </tr>
    <tr>
      <th>1</th>
      <td>홍련</td>
      <td>회계</td>
      <td>사원</td>
    </tr>
    <tr>
      <th>2</th>
      <td>콩쥐</td>
      <td>사원</td>
      <td>영업</td>
    </tr>
    <tr>
      <th>3</th>
      <td>팥쥐</td>
      <td>팀장</td>
      <td>인사</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([HR1, HR3], axis=0, ignore_index=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>부서</th>
      <th>직급</th>
      <th>급여</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>장화</td>
      <td>영업</td>
      <td>팀장</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>홍련</td>
      <td>회계</td>
      <td>사원</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>콩쥐</td>
      <td>영업</td>
      <td>NaN</td>
      <td>3500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>팥쥐</td>
      <td>인사</td>
      <td>NaN</td>
      <td>2800.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
HR4 = pd.Series({1: 2500}, name = '급여')
pd.concat([HR1, HR4], axis=1) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>부서</th>
      <th>직급</th>
      <th>급여</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>장화</td>
      <td>영업</td>
      <td>팀장</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>홍련</td>
      <td>회계</td>
      <td>사원</td>
      <td>2500.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
HR5 = pd.DataFrame({'급여': [4500, 3000, 3500]})
pd.concat([HR1, HR5], axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>이름</th>
      <th>부서</th>
      <th>직급</th>
      <th>급여</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>장화</td>
      <td>영업</td>
      <td>팀장</td>
      <td>4500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>홍련</td>
      <td>회계</td>
      <td>사원</td>
      <td>3000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3500</td>
    </tr>
  </tbody>
</table>
</div>



#### 2. 조인


```python
product = pd.DataFrame({'상품코드': ['G1', 'G2', 'G3', 'G4'], 
                        '상품명': ['우유', '감자', '빵', '치킨']})
sale = pd.DataFrame({'주문번호': [1001, 1002, 1002, 1003, 1004], 
                     '상품코드': ['G4', 'G3', 'G1', 'G3', 'G5'], 
                     '주문수량': [1, 44, 2, 2, 3]})

# inner join
sale.merge(product, on='상품코드', how='inner')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>주문번호</th>
      <th>상품코드</th>
      <th>주문수량</th>
      <th>상품명</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>G4</td>
      <td>1</td>
      <td>치킨</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002</td>
      <td>G3</td>
      <td>44</td>
      <td>빵</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1003</td>
      <td>G3</td>
      <td>2</td>
      <td>빵</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1002</td>
      <td>G1</td>
      <td>2</td>
      <td>우유</td>
    </tr>
  </tbody>
</table>
</div>




```python
# outer join
sale.merge(product, on='상품코드', how='outer', sort=True)   # sort 는 기준컬럼을 기준
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>주문번호</th>
      <th>상품코드</th>
      <th>주문수량</th>
      <th>상품명</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1002.0</td>
      <td>G1</td>
      <td>2.0</td>
      <td>우유</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>G2</td>
      <td>NaN</td>
      <td>감자</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1002.0</td>
      <td>G3</td>
      <td>44.0</td>
      <td>빵</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1003.0</td>
      <td>G3</td>
      <td>2.0</td>
      <td>빵</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1001.0</td>
      <td>G4</td>
      <td>1.0</td>
      <td>치킨</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1004.0</td>
      <td>G5</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
sale.merge(product, left_on='상품코드', right_on='상품코드', how='left')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>주문번호</th>
      <th>상품코드</th>
      <th>주문수량</th>
      <th>상품명</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>G4</td>
      <td>1</td>
      <td>치킨</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002</td>
      <td>G3</td>
      <td>44</td>
      <td>빵</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1002</td>
      <td>G1</td>
      <td>2</td>
      <td>우유</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1003</td>
      <td>G3</td>
      <td>2</td>
      <td>빵</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1004</td>
      <td>G5</td>
      <td>3</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### 제7절. 데이터 요약

#### 1. 그룹화와 집계


```python
import pandas as pd
from sklearn.datasets import load_iris
IRIS = load_iris()
iris = pd.DataFrame(IRIS.data, columns=IRIS.feature_names)
target = IRIS.target
iris['class'] = target
```


```python
iris['class'] = iris['class'].map({0:'setosa', 1:'versicolor', 2:'virginia'})
```


```python
iris
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginia</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>virginia</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>virginia</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginia</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginia</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>




```python

```
