# T-SNE
###### 本程式全部透過讀完Hintton的論文以及[這篇文章](https://tomohiroliu22.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98%E7%B3%BB%E5%88%97-78-t-%E9%9A%A8%E6%A9%9F%E9%84%B0%E8%BF%91%E5%B5%8C%E5%85%A5%E6%B3%95-t-distributed-stochastic-neighbor-embedding-a0ed57759769)後自行寫出的

### 1. 數學原理


### 2. 程式碼解釋

### 3. 優缺點

### 4. 執行結果
##### 因為TSNE必須經過多次計算資料之間的高斯概率分布並且電腦無法承受大量的手寫辨識資料，因此我將這個網站提供的程式訓練資料及從1700個降低至1000個。而自己的程式碼訓練資料也降低至1000個來進行比較，以下是比較結果 (隨著訓練資料越多T-SNE的分群準度越高，這點在[這篇文章中所使用的程式碼可以證明](https://tomohiroliu22.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98%E7%B3%BB%E5%88%97-78-t-%E9%9A%A8%E6%A9%9F%E9%84%B0%E8%BF%91%E5%B5%8C%E5%85%A5%E6%B3%95-t-distributed-stochastic-neighbor-embedding-a0ed57759769): 

##### 我自己寫的TSNE結果

![](https://lh3.googleusercontent.com/drive-viewer/AFGJ81o6X9rASGpH5JMP7c_iXSRSzR-W6d3H-Q_8H4do1KFO6HEUY_od00S2UN067yM0L1F7-MZDgDb_ZsSP8dKgMcXhRewi6Q=w1920-h902)

##### 別人寫的TSNE結果

![](https://lh3.googleusercontent.com/drive-viewer/AFGJ81oF8PlVifwzVweitpTQAF8a2zMh5Py3Z2kaETcp_6DEqe-96rby-WSxrqw1TQIc-_xDBy2M5ts2aoQCTtl3fhZKb9_I=w1920-h902)