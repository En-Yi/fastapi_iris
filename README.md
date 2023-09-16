# fastapi_iris
建立一個簡易 fastapi 於 iris data

資料庫 : xampp -> phpadmin mysql

# CRUD
Create : Post
新增一筆 iris data

Read : Get
1. 取得前幾筆 iris data 
2. 隨機取得數筆 iris data
3. 隨機取得數筆 iris data (非同步)

Update : Put
更新一筆 iris data (指定 id)

Delete : Delete
刪除一筆 iris data (指定 id)

# Prediction
從 sklearn 匯入 iris data，並用隨機森林模型
Post : 輸入 iris data 的四個特徵 (SepalLength, SepalWidth, PetalLength, PetalWidth) 預測類別
