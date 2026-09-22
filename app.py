from flask import Flask, Response, jsonify, request
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from flask_cors import CORS

load_dotenv()

# INIT API
app = Flask(__name__)
CORS(app)

DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def load_data():
    return pd.read_sql("SELECT * FROM players", engine)

@app.route("/")
def home():
    res = jsonify({"message":"Serie A API is running!"})
    return res

@app.get("/items")
def get_all():
    df = load_data()
    return Response(df.to_json(orient="records"), mimetype="application/json")


if __name__ == "__main__":
    app.run(debug=True)