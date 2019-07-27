# -*- coding: utf-8 -*-
'''a,b = 2,3
print("Sum is",a+b)'''

import numpy as np
a = np.array([1,2,3,4], dtype='float')
print(a)
print(a.shape)

a = np.array([1,2,3,4], dtype='complex')
print(a)
b = ('a','b')
a = np.array(b)
print(a)

b = [[1,2,3,4],[3,4,5,6]]
arr = np.array(b)
print(arr)
print(type(arr))
print(arr.shape)


c = np.zeros(shape = (3,3), dtype = 'int32')
print(c)

c = np.ones(shape = (2,3,4), dtype = 'int32')
print(c)

c = np.full((3,3),5)
print(c)



r = np.random.random((3,3))
print(r)

a = np.random.randint(5,100,size = (3,3))
print(a)
print(a.min())



l = np.arange(10,40,5)
print(l)

g = np.linspace(0,20,11)
print(g)


arr = np.array([[1,2,3,4],[5,2,3,4],[2,3,4,6],[3,3,3,4]])
#newarr = arr.reshape(2,2,4)
newarr = arr.reshape(-1,8)
print(arr)
print(newarr)


arr = np.array([[1,2,3,4],[5,2,3,4],[2,3,4,6],[3,3,3,4]])
new = arr.flatten()
print(arr)
print(arr.ndim)
print(arr.shape)
print(arr.size)
print(new)

#mixing indices and slicing
arr = np.array([[1,2,3,4],[5,2,3,4],[2,3,4,6],[3,3,3,4]])
temp = arr[0:3,1:3]
temp1 = arr[:3,1:3]
temp2 = arr[::2,::2]
temp3 = arr[:3,::2]
temp4 = arr[[1,3],:]
print(temp)
print(temp1)
print(temp4)


arr = np.array([[0+1,2-9,3,-4],[5,-2,3,4],[2,-3,4,6],[3,-3,3,4]])
cond = arr>0
print(cond)
temp = arr[cond]
print(temp)


#Sum,columnwise,rowwise
arr = np.array([[1,2,3,4],[5,2,3,4],[2,3,4,6],[3,3,3,4]])
add = np.sum(arr)
print(add)
print("Columnwise",np.sum(arr,axis=0))
print("Rowwise",np.sum(arr,axis=1))



#matrix operation
arr = np.random.randint(1,100,size = (5,5))
xmin,xmax = arr.min(),arr.max()
temp = (arr - xmin)/(xmax-xmin)
print(arr)
print(xmin,xmax)
print(temp)



#shuffling number
x = np.arange(11)
print(x)
temp = np.random.shuffle(x)
print(x)

#Pandas
import pandas as pd
df = pd.read_csv('~/Downloads/airquality.csv',delimiter = ',')
print(df.head(10))
print(df.shape)
print(df.describe())
print(df.isna().any().sum())
print(df.isna().sum())



df1 = df.fillna(0)
print(df1.head(10))
print(df1.isna().sum())


#df2 = df.fillna(method='pad')
df2 = df.fillna(method='ffill')
print(df2.head(10))
print(df2.isna().sum())

df3 = df.fillna(method='bfill')
print(df3.head(10))
print(df3.isna().sum())

df4 = df.fillna(df.mean())
print(df4.head(10))
print(df4.isna().sum())

df5 = df.dropna()
print(df.shape)
print(df5.shape)

#loc and iloc
print(df.index)
df.loc[[1,2,3,4,5],['Ozone','Solar.R']]
df.loc[[1,2,3,4,5],:]

df.iloc[[0,1,2],[0,1]]
df.iloc[[0,1,2],:]


#Line Plot
from matplotlib import pyplot as plt
x = range(153)
y = df['Temp']
plt.plot(x,y)
plt.title('Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()

from matplotlib import style
style.use('ggplot')

x = range(153)
y = df['Temp']
y2 = df['Wind']
plt.plot(x,y,linewidth=1)
plt.plot(x,y2,linewidth=2)
plt.title('Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()


from matplotlib import pyplot as plt
x = range(153)
y = df['Temp']
plt.scatter(x,y,color='g')
plt.title('Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()

from matplotlib import style
style.use('ggplot')

y = df['Day']
plt.hist(y)
plt.title('Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.grid(True)
plt.show()


