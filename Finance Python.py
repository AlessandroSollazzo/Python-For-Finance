#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from pandas_datareader import data as wb
import matplotlib.pyplot as plt


# In[2]:


import panda as pd


# In[3]:


import pandas as pd


# In[1]:


import pandas_datareader as web


# In[6]:


PG=web.DataReader ('PG', data_source='yahoo', start='1995-1-1')


# In[7]:


PG.head()


# In[7]:


PG.tail()


# In[11]:


PG['simple_return']=(PG['Adj Close']/PG['Adj Close'].shift(1))-1


# In[13]:


print PG['simple_return']


# In[14]:


print(PG['simple_return'])


# In[15]:


PG['simple_return'].plot(figsize=(8, 5))
plt.show()


# In[16]:


avg_returns_d=PG['simple_return'].mean()
avg_returns_d


# In[17]:


avg_returns_a=PG['simple_return'].mean()*250
avg_returns_a


# In[18]:


print str(round(avg_returns_a,5)*100)+'%'


# In[20]:


print str(round(avg_returns_a, 5)*100)+ '%'


# In[21]:


round ((avg_returns_a,2)*100)+ '%'


# In[22]:


round (avg_returns_a,2)+ '%'


# In[24]:


str(round (avg_returns_a,2)*100)+ '%'


# In[29]:


str(round (avg_returns_a,4)*100)+ '%'


# In[30]:


PG.head()


# In[31]:


PG['log_return']=np.log(PG['Adj Close']/PG['Adj Close'].shift(1))


# In[32]:


print PG['log_return']


# In[33]:


print(PG['log_return'])


# In[36]:


PG['log_return'].plot(figsize=(8, 5))
plt.show()


# In[37]:


log_return_d=PG['log_return'].mean()
log_return_d


# In[38]:


log_return_a=PG['log_return'].mean()*250
log_return_a


# In[39]:


str(round(log_return_a, 2)*100)+'%'


# In[40]:


PG.head()


# In[41]:


tickers=['PG', 'MSFT', 'F','GE']
mydata = pd.DataFrame()
for t in tickers:
    mydata[t] = web.DataReader(t, data_source='yahoo', start='1995-1-1')['Adj Close']


# In[42]:


mydata.info()


# In[43]:


mydata.head()


# In[44]:


mydata.tail()


# In[45]:


mydata.iloc[0]


# In[46]:


(mydata/mydata.iloc[0]*100).plot(figsize=(15, 6));
plt.show()


# In[47]:


mydata.plot(figsize=(15,6))
plt.show()


# In[48]:


mydata.loc[1995-01-03]


# In[49]:


mydata.loc['1995-01-03']


# In[50]:


mydata.iloc[0]


# In[51]:


(mydata/mydata.iloc[0]*100).plot(figsize=(15, 6));
plt.show()


# In[52]:


returns=(mydata/mydata.shift(1))-1


# In[53]:


returns.head


# In[54]:


returns.head()


# In[57]:


weights = np.array([0.25, 0.25, 0.25, 0.25])


# In[58]:


np.dot(returns, weights)


# In[59]:


annual_returns = returns.mean()*250


# In[60]:


annual_returns


# In[61]:


np.dot(returns, weights)


# In[63]:


np.dot(annual_returns, weights)


# In[64]:


pfolio_1=str(round(np.dot(annual_returns, weights),5)*100) + '%'


# In[66]:


print (pfolio_1)


# In[67]:


weight_2=np.array([0.4,0.4,0.15,0.05])


# In[69]:


pfolio_2= str(round(np.dot(annual_returns, weight_2), 5)*100)+'%'


# In[71]:


print (pfolio_1)
print (pfolio_2)


# In[74]:


tickers = ['^GSPC','^IXIC','^GDAXI']
ind_data = pd.DataFrame()
for t in tickers:
    ind_data[t] = web.DataReader(t, data_source = 'yahoo', start='1997-1-1')['Adj Close']


# In[75]:


ind_data.head()


# In[76]:


ind_data.tail()


# In[77]:


(ind_data/ind_data.iloc[0]*100).plot(figsize = (15, 6));
plt.show()


# In[78]:


ind_returns=(ind_data/ind_data.shift(1))-1
ind_returns.tail()


# In[79]:


annual_ind_returns=ind_returns.mean()*250


# In[80]:


annual_ind_returns


# In[81]:


tickers=['PG','^GSPC','^DJI']
data_2=pd.DataFrame()
for t in tickers:
    data_2[t]=web.DataReader(t,data_source='yahoo', start='2007-1-1') ['Adj Close']


# In[82]:


data_2.tail()


# In[83]:


(data_2/data_2.iloc[0]*100).plot(figsize=(15,6));


# In[84]:


data_2.tail()


# In[20]:


tickers=['PG', 'BEI.DE']
prova = pd.DataFrame()
for t in tickers:
    prova[t] = web.DataReader(t, data_source='yahoo', start='2007-1-1')['Adj Close']


# In[21]:


prova.tail()


# In[22]:


prova_returns=np.log(prova/prova.shift(1))


# In[23]:


prova_returns


# In[105]:


prova_returns['PG'].mean()


# In[106]:


prova_returns['PG'].mean()*250


# In[107]:


prova_returns['PG'].std()


# In[108]:


prova_returns['PG'].std()*250**0.5


# In[113]:


prova_returns['BEI.DE'].mean()


# In[111]:


prova_returns['BEI.DE'].mean()*250


# In[112]:


prova_returns['BEI.DE'].std()


# In[114]:


prova_returns['BEI.DE'].std()*250**0.5


# In[25]:


print (prova_returns['PG'].mean()*250)
print (prova_returns['BEI.DE'].mean()*250)


# In[26]:


prova_returns[['PG', 'BEI.DE']].mean()*250


# In[120]:


prova_returns[['PG', 'BEI.DE']].std()*250**0.5


# In[123]:


PG_Var = prova_returns ['PG'].var()
PG_Var


# In[124]:


BEI_Var = prova_returns ['BEI.DE'].var()


# In[125]:


BEI_Var


# In[126]:


PG_Var_a = prova_returns ['PG'].var()*250


# In[127]:


PG_Var_a 


# In[32]:


BEI_Var_a = prova_returns ['BEI.DE'].var()*250


# In[33]:


BEI_Var_a


# In[131]:


cov_matrix=prova_returns.cov()
cov_matrix


# In[132]:


cov_matrix_a=prova_returns.cov()*250


# In[133]:


cov_matrix_a


# In[134]:


corr_matrix=prova_returns.corr()


# In[135]:


corr_matrix


# In[136]:


corr_matrix_a=prova_returns.corr()*250
corr_matrix_a


# In[137]:


#Calculating Portfolio Risk


# In[139]:


#Equal Weighting scheme:
weights=np.array([0.5,0.5])


# In[40]:


#Portfolio Variance
pfolio_var=np.dot(weights.T, np.dot(prova_returns.cov()*250,weights))
pfolio_var


# In[141]:


#Portfolio Volatility
pfolio_vol=np.dot(weights.T, np.dot(prova_returns.cov()*250,weights))**0.5
pfolio_vol


# In[142]:


print (str(round(pfolio_vol, 5)*100)+ '%')


# In[1]:


#Calculating Diversifable and Non-Diversifable Risk of a Portfolio


# In[27]:


weights= np.array([0.5,0.5])


# In[3]:


import pandas as pd


# In[8]:


import pandas_datareader as web


# In[28]:


weights= np.array([0.5,0.5])


# In[10]:


import numpy as np


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


weights= np.array([0.5,0.5])


# In[13]:


weights[0]


# In[14]:


weights[1]


# In[30]:


PG_var_a=prova_returns[['PG']].var()*250


# In[31]:


PG_var_a


# In[35]:


BEI_var_a=prova_returns[['BEI.DE']].var()*250


# In[36]:


BEI_var_a


# In[37]:


float(PG_var_a)


# In[44]:


PG_var_a=prova_returns['PG'].var()*250
PG_var_a


# In[46]:


BEI_var_a=prova_returns['BEI.DE'].var()*250
BEI_var_a


# In[48]:


dr=pfolio_var-(weights[0]**2*PG_var_a)-(weights[1]**2*BEI_var_a)


# In[49]:


dr


# In[56]:


print (str(round(dr*100,3)) + '%')


# In[57]:


n_dr_1=pfolio_var-dr


# In[58]:


n_dr_1


# In[59]:


import numpy as np


# In[60]:


import pandas as pd


# In[61]:


from scipy import stats


# In[62]:


import statsmodels.api as sm


# In[63]:


import matplotlib.pyplot as plt


# In[67]:


data=pd.read_excel ("/Users/alessandrosollazzo/Desktop/original.xlsx")


# In[68]:


data


# In[70]:


data[['House Price','House Size (sq.ft.)']]


# In[71]:


X=data ['House Size (sq.ft.)']
Y=data['House Price']


# In[72]:


X


# In[73]:


Y


# In[74]:


plt.scatter(X,Y)
plt.show()


# In[75]:


plt.scatter(X,Y)
plt.axis([0, 2500, 0, 1500000])
plt.show()


# In[76]:


plt.scatter(X,Y)
plt.axis([0, 2500, 0, 1500000])
plt.ylabel('House Price')
plt.xlabel('House Size (sq.ft.)')
plt.show()


# In[77]:


X1=sm.add_constant(X)


# In[78]:


reg=sm.OLS(Y, X1).fit()


# In[79]:


reg.summary()


# In[80]:


#Expected Value of Y:


# 260800 + 402*1000

# In[81]:


260800 + 402*1000


# In[83]:


slope, intercept, r_value, p_value, std_error = stats.linregress(X,Y)


# In[84]:


slope


# In[85]:


intercept


# In[86]:


r_value


# In[87]:


r_value**2


# In[88]:


p_value


# In[89]:


std_error


# In[90]:


#Efficient Frontier


# In[91]:


import numpy as np
import pandas as pd
from pandas_datareader import data as web 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[92]:


assets=['PG', '^GSPC']
pf_data=pd.DataFrame()

for a in assets:
    pf_data[a]=web.DataReader(a,data_source='yahoo', start = '2010-1-1')['Adj Close']


# In[94]:


pf_data.tail()


# In[96]:


(pf_data/pf_data.iloc[0]*100).plot(figsize=(10, 5))


# In[ ]:





# In[97]:


log_returns=np.log(pf_data/pf_data.shift(1))


# In[98]:


log_returns.mean()*250


# In[99]:


log_returns.cov()*250


# In[101]:


log_returns.corr()


# In[102]:


num_assets=len(assets)


# In[103]:


num_assets


# In[104]:


arr=np.random.random(2)
arr


# In[105]:


weights=np.random.random(num_assets)


# In[106]:


weights/=np.sum(weights)


# In[107]:


weights


# In[108]:


arr[0]+arr[1]


# In[109]:


weights[0]+weights[1]


# In[110]:


#Expected Portfolio Return


# In[112]:


np.sum(weights*log_returns.mean())*250


# In[113]:


np.dot(weights.T,np.dot(log_returns.cov()*250,weights))


# In[114]:


np.sqrt(np.dot(weights.T,np.dot(log_returns.cov()*250,weights)))


# In[116]:


pfolio_returns=[]
pfolio_volatilities=[]
for x in range (1000):
    weights=np.random.random(num_assets)
    weights/=np.sum(weights)
    pfolio_returns.append(np.sum(weights*log_returns.mean())*250)
    pfolio_volatilities.append(np.sqrt(np.dot(weights.T,np.dot(log_returns.cov()*250,weights))))
    
pfolio_returns=np.array(pfolio_returns)
pfolio_volatiltites=np.array(pfolio_volatilities)

pfolio_returns, pfolio_volatilities


# In[119]:


portfolios=pd.DataFrame({'Return':pfolio_returns,'volatility':pfolio_volatilities})


# In[120]:


portfolios.head()


# In[121]:


portfolios.tail()


# In[123]:


portfolios.plot(x='volatility',y='Return',kind='scatter', figsize=(10, 6));
plt.xlabel('Expected Volatilitiy')
plt.ylabel('Expected Return')


# In[ ]:





# In[125]:


import numpy as np
import pandas as pd
from pandas_datareader import data as web 

tickers = ['PG','^GSPC']
data=pd.DataFrame()
for t in tickers:
    data[t]=web.DataReader(t,data_source='yahoo', start='2012-1-1',end='2016-12-31')['Adj Close']


# In[130]:


sec_returns=np.log(data/data.shift(1))


# In[131]:


cov=sec_returns.cov()*250
cov


# In[132]:


cov_with_market=cov.iloc[0,1]


# In[133]:


cov_with_market


# In[134]:


market_var=sec_returns['^GSPC'].var()*250
market_var


# In[135]:


#BETA


# In[136]:


PG_beta=cov_with_market/market_var


# In[137]:


PG_beta


# In[138]:


#CAPM


# In[139]:


PG_er=0.025+PG_beta*0.05
PG_er


# In[140]:


#SharpeRatio


# In[141]:


Sharpe=(PG_er-0.025)/(sec_returns['PG'].std()*250**0.5)
Sharpe


# In[151]:


import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
data=pd.read_excel ("/Users/alessandrosollazzo/Desktop/original.xlsx")
data


# In[158]:


X=data[['House Size (sq.ft.)','Number of Rooms','Year of Construction']]
Y=data['House Price']


# In[159]:


X1=sm.add_constant(x)


# In[160]:


from scipy import stats
import statsmodels.api as sm


# In[161]:


X1=sm.add_constant(x)


# In[162]:


data[['House Price','House Size (sq.ft.)','Year of Construction']]


# In[163]:


X=data[['House Size (sq.ft.)','Number of Rooms','Year of Construction']]
Y=data['House Price']


# In[164]:


X1=sm.add_constant(x)


# In[165]:


reg=sm.OLS(Y,X1).fit()


# In[166]:


import numpy as np


# In[169]:


import pandas as pd


# In[170]:


from scipy import stats


# In[171]:


import statsmodels.api as sm


# In[172]:


import matplotlib.pyplot as plt


# In[173]:


data=pd.read_excel ("/Users/alessandrosollazzo/Desktop/original.xlsx")


# In[174]:


data


# In[175]:


X=data[['House Size (sq.ft.)','Number of Rooms','Year of Construction']]
Y=data['House Price']


# In[176]:


X


# In[178]:


Y


# In[179]:


X1=sm.add_constant(x)


# In[180]:


reg=sm.OLS(Y,X1).fit()


# In[ ]:




