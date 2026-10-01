from flask import Flask, Response, jsonify, request
from player_attribution import fantasy_monte_carlo, load_lineups_file
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from flask_cors import CORS
import requests as http_requests
import secrets
import uuid
import json
from player_attribution import (
    DEFAULT_REAL_FORMATION,
    load_squad_pool,
    pick_auto_xi,
)
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta, timezone
from functools import wraps

load_dotenv()

app = Flask(__name__)
CORS(app)

DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL)

JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable is not set")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24 * 7  # 1 week
FROM_NAME = os.environ.get("FROM_NAME", "FantaLeague")
FROM_EMAIL = os.environ.get("FROM_EMAIL")
BREVO_API_KEY = os.environ.get("BREVO_API_KEY")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")
RESET_TOKEN_EXPIRY_MINUTES = 30


def create_token(user_id):
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        token = auth_header.removeprefix("Bearer ").strip()
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        request.user_id = payload["user_id"]
        return f(*args, **kwargs)
    return wrapper


def load_data():
    return pd.read_sql("SELECT * FROM players", engine)

@app.route("/")
def home():
    return jsonify({"message": "Serie A API is running!"})

@app.get("/items")
def get_all():
    df = load_data()
    return Response(df.to_json(orient="records"), mimetype="application/json")


# ---- Auth ----

@app.post("/auth/register")
def register():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    if not email or "@" not in email:
        return jsonify({"error": "A valid email is required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    password_hash = generate_password_hash(password)

    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                INSERT INTO users (username, email, password_hash)
                VALUES (:username, :email, :password_hash)
                RETURNING id
            """), {"username": username, "email": email, "password_hash": password_hash})
            user_id = result.scalar()
    except Exception:
        # unique constraint could be on username OR email
        return jsonify({"error": "Username or email already taken"}), 409

    token = create_token(user_id)
    return jsonify({"token": token, "username": username}), 201


@app.post("/auth/login")
def login():
    data = request.get_json()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id, password_hash FROM users WHERE username = :username
        """), {"username": username}).mappings().first()

    if row is None or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Invalid username or password"}), 401

    token = create_token(row["id"])
    return jsonify({"token": token, "username": username})

@app.post("/auth/forgot-password")
def forgot_password():
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()

    if not email:
        return jsonify({"error": "Email is required"}), 400

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT id FROM users WHERE LOWER(email) = :email
        """), {"email": email}).mappings().first()

    # Always return the same response — don't reveal whether the email exists.
    if row is None:
        return jsonify({"message": "If that email is registered, we've sent a reset link."}), 200

    token = create_reset_token(row["id"])
    reset_link = f"{FRONTEND_URL}/reset-password?token={token}"

    try:
        send_reset_email(email, reset_link)
    except Exception as e:
        # log server-side, but still return 200 so we don't leak whether the email exists
        print(f"Failed to send reset email to {email}: {e}")
        return jsonify({"message": "If that email is registered, we've sent a reset link."}), 200

    return jsonify({"message": "If that email is registered, we've sent a reset link."}), 200


@app.post("/auth/reset-password")
def reset_password():
    data = request.get_json() or {}
    token = data.get("token") or ""
    new_password = data.get("password") or ""

    if not token or not new_password:
        return jsonify({"error": "Token and password are required"}), 400
    if len(new_password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "This reset link has expired. Request a new one."}), 400
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid reset link"}), 400

    if payload.get("purpose") != "reset":
        return jsonify({"error": "Invalid reset link"}), 400

    user_id = payload.get("user_id")

    with engine.begin() as conn:
        row = conn.execute(text(
            "SELECT id FROM users WHERE id = :id"
        ), {"id": user_id}).mappings().first()

        if row is None:
            return jsonify({"error": "Invalid reset link"}), 400

        conn.execute(text(
            "UPDATE users SET password_hash = :hash WHERE id = :id"
        ), {"hash": generate_password_hash(new_password), "id": user_id})

    return jsonify({"message": "Password updated. You can now sign in."}), 200

# ---- Teams ----

@app.get("/teams")
@require_auth
def get_teams():
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, name, player_ids
            FROM teams
            WHERE user_id = :user_id
            ORDER BY created_at
        """), {"user_id": request.user_id}).mappings().all()
    return jsonify([dict(r) for r in rows])

@app.post("/teams")
@require_auth
def create_team():
    data = request.get_json()
    name = (data.get("name") or "Untitled team").strip()
    team_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO teams (id, name, player_ids, user_id) VALUES (:id, :name, '[]'::jsonb, :user_id)"
        ), {"id": team_id, "name": name, "user_id": request.user_id})
    return jsonify({"id": team_id, "name": name, "player_ids": [], "is_shared": False, "is_owner": True}), 201

@app.delete("/teams/<team_id>")
@require_auth
def delete_team(team_id):
    with engine.begin() as conn:
        row = conn.execute(text(
            "SELECT user_id FROM teams WHERE id = CAST(:id AS uuid)"
        ), {"id": team_id}).mappings().first()

        if row is None:
            return jsonify({"error": "Team not found"}), 404
        if row["user_id"] != request.user_id:
            return jsonify({"error": "You can only delete your own teams"}), 403

        conn.execute(text("DELETE FROM teams WHERE id = CAST(:id AS uuid)"), {"id": team_id})
    return "", 204

@app.put("/teams/<team_id>/players")
@require_auth
def update_team_players(team_id):
    data = request.get_json()
    player_ids = data.get("player_ids", [])

    with engine.begin() as conn:
        row = conn.execute(text(
            "SELECT user_id FROM teams WHERE id = CAST(:id AS uuid)"
        ), {"id": team_id}).mappings().first()

        if row is None:
            return jsonify({"error": "Team not found"}), 404
        if row["user_id"] != request.user_id:
            return jsonify({"error": "You can only edit your own teams"}), 403

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

def create_reset_token(user_id):
    """A short-lived JWT that can ONLY be used to reset a password."""
    payload = {
        "user_id": user_id,
        "purpose": "reset",  # distinguishes this from a login token
        "jti": secrets.token_urlsafe(16),  # unique id, for logging/debugging
        "exp": datetime.now(timezone.utc) + timedelta(minutes=RESET_TOKEN_EXPIRY_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def send_reset_email(to_email, reset_link):
    """Send the reset link via Brevo. Raises on failure."""
    if not BREVO_API_KEY:
        raise RuntimeError("BREVO_API_KEY is not set")
    if not FROM_EMAIL:
        raise RuntimeError("FROM_EMAIL is not set")

    resp = http_requests.post(
        "https://api.brevo.com/v3/smtp/email",
        headers={
            "api-key": BREVO_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "sender": {"name": FROM_NAME, "email": FROM_EMAIL},
            "to": [{"email": to_email}],
            "subject": "Reset your FantaLeague password",
            "htmlContent": f"""
                <p>Someone requested a password reset for your FantaLeague account.</p>
                <p><a href="{reset_link}">Click here to set a new password</a></p>
                <p>This link expires in {RESET_TOKEN_EXPIRY_MINUTES} minutes.</p>
                <p>If you didn't request this, you can safely ignore this email.</p>
            """,
        },
        timeout=10,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Brevo error {resp.status_code}: {resp.text[:200]}")

if __name__ == "__main__":
    app.run(debug=True)