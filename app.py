from flask import Flask, Response, jsonify, request
from player_attribution import fantasy_monte_carlo, load_lineups_file
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from flask_cors import CORS
import uuid
import json
from player_attribution import (
    DEFAULT_REAL_FORMATION,
    load_squad_pool,
    pick_auto_xi,
)

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

# ---- Simulator ----

@app.post("/simulate")
def simulate():
    data = request.get_json() or {}

    team_a = data.get("team_a")
    team_b = data.get("team_b")
    season = data.get("season", "2026-27")
    giornata = data.get("giornata")
    sims = int(data.get("sims", 1000))
    seed = data.get("seed")           # optional
    lineups = data.get("lineups")     # optional: {club: [player_id, ...]}

    if not team_a or not team_b:
        return jsonify({"error": "team_a and team_b are required"}), 400
    if giornata is None:
        return jsonify({"error": "giornata is required"}), 400

    try:
        result = fantasy_monte_carlo(
            name_a=team_a,
            name_b=team_b,
            season=season,
            giornata=int(giornata),
            num_sims=sims,
            lineups=lineups,
            seed=seed,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Simulation failed: {e}"}), 500

    return jsonify(result)


@app.get("/lineups")
def get_lineups():
    season = request.args.get("season", "2026-27")
    giornata = request.args.get("giornata", type=int)
    if giornata is None:
        return jsonify({"error": "giornata is required"}), 400

    fixtures = pd.read_sql(text("""
        SELECT home_team, away_team FROM calendar_2627
        WHERE season = :season AND giornata = :giornata
    """), engine, params={"season": season, "giornata": giornata})
    if fixtures.empty:
        return jsonify({"error": f"No fixtures for {season} giornata {giornata}"}), 404

    squads = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    pool = load_squad_pool(season, squads)

    clubs = {}
    for squad in squads:
        squad_pool = pool[pool["squad"] == squad]
        xi = pick_auto_xi(squad_pool, DEFAULT_REAL_FORMATION, squad)
        clubs[squad] = {
            "xi": [
                {
                    "player_id": p["player_id"],
                    "long_name": p["long_name"],
                    "line": p["line"],
                    "slot": p.get("slot"),
                    "overall": p["overall_rating"],
                }
                for p in xi
            ],
            "pool": [
                {
                    "player_id": int(r["player_id"]),
                    "long_name": r["long_name"],
                    "line": r["line"],
                    "overall": None if pd.isna(r["overall"]) else float(r["overall"]),
                }
                for _, r in squad_pool.iterrows()
            ],
        }

    return jsonify({"season": season, "giornata": giornata, "clubs": clubs})

if __name__ == "__main__":
    app.run(debug=True)