from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sqlalchemy
# 建立 FastAPI 應用程式
app = FastAPI()

# 建立資料庫連線
DATABASE_URL = "mysql+mysqlconnector://tobio:u/3g0 zo vm/6@localhost/pydb"
engine = create_engine(DATABASE_URL)

# 建立資料庫 Session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 定義資料模型（假設已經有 iris 表）
Base = sqlalchemy.orm.declarative_base()

class Iris(Base):
    __tablename__ = "iris"

    id = Column(Integer, primary_key=True, index=True)
    sepal_length = Column(Float)
    sepal_width = Column(Float)
    petal_length = Column(Float)
    petal_width = Column(Float)
    species = Column(String)

# 定義 Pydantic 模型
class IrisCreate(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str

# # 處理新增 iris 資料的請求
@app.post("/iris/")
def create_iris(iris: IrisCreate):
    db = SessionLocal()
    db_iris = Iris(**iris.dict())
    db.add(db_iris)
    db.commit()
    db.refresh(db_iris)
    db.close()
    return db_iris


# 處理取得前幾筆 iris 資料的請求
@app.get("/iris/")
def get_iris(limit: int):
    db = SessionLocal()
    iris_data = db.query(Iris).limit(limit).all()
    db.close()
    return iris_data

# 定義要更新的 iris 資料的 Pydantic 模型
class IrisUpdate(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str

# 處理更新 iris 資料的請求
@app.put("/iris/{iris_id}")
def update_iris(iris_id: int, iris_data: IrisUpdate):
    db = SessionLocal()
    db_iris = db.query(Iris).filter(Iris.id == iris_id).first()

    if db_iris is None:
        db.close()
        raise HTTPException(status_code=404, detail="Iris data not found")

    # 更新 iris 資料
    db_iris.sepal_length = iris_data.sepal_length
    db_iris.sepal_width = iris_data.sepal_width
    db_iris.petal_length = iris_data.petal_length
    db_iris.petal_width = iris_data.petal_width
    db_iris.species = iris_data.species

    db.commit()
    db.refresh(db_iris)
    db.close()
    return db_iris

# 處理刪除 iris 資料的請求
@app.delete("/iris/{iris_id}")
def delete_iris(iris_id: int):
    db = SessionLocal()
    db_iris = db.query(Iris).filter(Iris.id == iris_id).first()

    if db_iris is None:
        db.close()
        raise HTTPException(status_code=404, detail="Iris data not found")
    
    # 刪除 iris 資料
    db.delete(db_iris)
    db.commit()
    db.close()
    return {"message": "Iris data deleted successfully"}



# 定義輸入資料的 Pydantic 模型
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 定義預測路由
@app.post("/predict/")
async def predict_iris_species(input_data: IrisInput):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    # 載入 Iris 資料集
    iris = load_iris()
    X = iris.data
    y = iris.target
    # 訓練一個隨機森林分類器
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    if input_data is not None:    
    # 將輸入特徵轉換為 NumPy 陣列
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])
        
        # 使用模型進行預測
        prediction = model.predict(features)
        # 返回預測結果
        return {"predicted_species": int(prediction[0])}
    else:
        raise HTTPException(status_code=404, detail="Prediction failed")


# # 啟動 FastAPI 應用程式
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)