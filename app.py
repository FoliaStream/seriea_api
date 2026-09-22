from flask import Flask, Response, jsonify, request
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from flask_cors import CORS
import uuid
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def load_data():
    return pd.read_sql("SELECT * FROM players", engine)

@app.route("/")
def home():
    return jsonify({"message": "Serie A API is running!"})

@app.get("/items")
def get_all():
    df = load_data()
    return Response(df.to_json(orient="records"), mimetype="application/json")


# ---- Teams ----

@app.get("/teams")
def get_teams():
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, name, player_ids FROM teams ORDER BY created_at"
        )).mappings().all()
    return jsonify([dict(r) for r in rows])

@app.post("/teams")
def create_team():
    data = request.get_json()
    name = (data.get("name") or "Untitled team").strip()
    team_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO teams (id, name, player_ids) VALUES (:id, :name, '[]'::jsonb)"
        ), {"id": team_id, "name": name})
    return jsonify({"id": team_id, "name": name, "player_ids": []}), 201

@app.delete("/teams/<team_id>")
def delete_team(team_id):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM teams WHERE id = CAST(:id AS uuid)"), {"id": team_id})
    return "", 204

@app.put("/teams/<team_id>/players")
def update_team_players(team_id):
    data = request.get_json()
    player_ids = data.get("player_ids", [])
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE teams SET player_ids = CAST(:ids AS jsonb), updated_at = now() WHERE id = CAST(:id AS uuid)"
        ), {"ids": json.dumps(player_ids), "id": team_id})
    return "", 204


if __name__ == "__main__":
    app.run(debug=True)