import os
import sys
from collections import Counter
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

BASELINE_SEASON = "2025-26"
HOME_ADVANTAGE = 1.2  # home team's expected goals are boosted by this factor


def get_team_strength(squad):
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT attack_strength, defense_strength FROM team_strength_current WHERE squad = :squad"
        ), {"squad": squad}).mappings().first()
    if row is None:
        raise ValueError(f"No strength rating found for squad: {squad}")
    return float(row["attack_strength"]), float(row["defense_strength"])


def get_league_avg_goals():
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT league_avg_goals FROM team_strength WHERE season = :season LIMIT 1"
        ), {"season": BASELINE_SEASON}).mappings().first()
    if row is None:
        raise ValueError(f"No league average found for season: {BASELINE_SEASON}")
    return float(row["league_avg_goals"])


def simulate_match(home_team, away_team, num_simulations=1000, seed=None):
    home_attack, home_defense = get_team_strength(home_team)
    away_attack, away_defense = get_team_strength(away_team)
    league_avg = get_league_avg_goals()

    home_xg = league_avg * home_attack * away_defense * HOME_ADVANTAGE
    away_xg = league_avg * away_attack * home_defense

    rng = np.random.default_rng(seed)
    home_goals = rng.poisson(home_xg, size=num_simulations)
    away_goals = rng.poisson(away_xg, size=num_simulations)

    home_wins = int((home_goals > away_goals).sum())
    draws = int((home_goals == away_goals).sum())
    away_wins = int((home_goals < away_goals).sum())

    scorelines = Counter(zip(home_goals.tolist(), away_goals.tolist()))
    top_score, top_count = scorelines.most_common(1)[0]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "num_simulations": num_simulations,
        "home_expected_goals": round(home_xg, 2),
        "away_expected_goals": round(away_xg, 2),
        "home_win_pct": round(100 * home_wins / num_simulations, 1),
        "draw_pct": round(100 * draws / num_simulations, 1),
        "away_win_pct": round(100 * away_wins / num_simulations, 1),
        "avg_home_goals": round(float(home_goals.mean()), 2),
        "avg_away_goals": round(float(away_goals.mean()), 2),
        "most_common_scoreline": f"{top_score[0]}-{top_score[1]} ({top_count}/{num_simulations} times)",
    }




if __name__ == "__main__":
    home = sys.argv[1] if len(sys.argv) > 1 else "Inter"
    away = sys.argv[2] if len(sys.argv) > 2 else "Milan"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10000

    result = simulate_match(home, away, num_simulations=n)
    for k, v in result.items():
        print(f"{k}: {v}")