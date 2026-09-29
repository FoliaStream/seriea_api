import argparse
import json
import math
import os
import time
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text

from match_simulator import HOME_ADVANTAGE, get_league_avg_goals, get_team_strength

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

DECAY = 0.7
SEASON_ORDER = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
MOST_RECENT_INDEX = len(SEASON_ORDER) - 1
MIN_NINETIES_FOR_HISTORY = 5.0  # below this, use the fallback prior instead

POSITION_TO_LINE = {
    "GK": "P",
    "CB": "D", "RB": "D", "LB": "D", "RWB": "D", "LWB": "D",
    "CDM": "C", "CM": "C", "CAM": "C", "RM": "C", "LM": "C",
    "RW": "A", "LW": "A", "ST": "A", "CF": "A",
}

# heuristic fallback baselines (per-90), used only when a player lacks enough history
FALLBACK_BASE = {
    "P": {"goals": 0.00, "assists": 0.00, "yellow": 0.10, "red": 0.01},
    "D": {"goals": 0.03, "assists": 0.05, "yellow": 0.18, "red": 0.015},
    "C": {"goals": 0.10, "assists": 0.15, "yellow": 0.15, "red": 0.01},
    "A": {"goals": 0.35, "assists": 0.15, "yellow": 0.12, "red": 0.01},
}
OVERALL_BASELINE = 75  # rating at which the fallback prior applies unscaled

BONUS_GOAL = 3
BONUS_ASSIST = 1
MALUS_YELLOW = -0.5
MALUS_RED = -1
MALUS_CONCEDED_PER_GOAL = -1  # goalkeeper only

# used when no real lineup is provided for a club
DEFAULT_REAL_FORMATION = {"P": 1, "D": 4, "C": 3, "A": 3}


# ---------------------------------------------------------------- helpers

def recency_weight(season):
    seasons_ago = MOST_RECENT_INDEX - SEASON_ORDER.index(season)
    return DECAY ** seasons_ago


def round_half(x):
    """Round to the nearest 0.5, half rounding up (not banker's rounding)."""
    return math.floor(x * 2 + 0.5) / 2


def grades_to_goals(total):
    """Convert summed grades to goals.
    66 -> 1, 72 -> 2, 78 -> 3, 84 -> 4, and +1 goal per 6 points above that.
    Below 66 -> 0.
    """
    return max(0, math.floor((total - 60) / 6))


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFD", name)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower().strip()


def read_in(sql, **params):
    """pd.read_sql for queries that use `IN :name` with a list parameter."""
    stmt = text(sql)
    clean = {}
    for key, value in params.items():
        if isinstance(value, (list, tuple, set)):
            stmt = stmt.bindparams(bindparam(key, expanding=True))
            value = list(value)
        clean[key] = value
    return pd.read_sql(stmt, engine, params=clean)


def add_line_column(df):
    """Fantacalcio line (P/D/C/A) from the PRIMARY position only."""
    df = df.copy()
    primary = df["player_positions"].astype(str).str.replace(",", " ").str.split().str[0]
    df["line"] = primary.map(POSITION_TO_LINE)
    return df


def make_player(row, club):
    overall, fbref_id = row["overall"], row["fbref_id"]
    return {
        "player_id": int(row["player_id"]),
        "long_name": row["long_name"],
        "line": row["line"],
        "overall_rating": None if pd.isna(overall) else float(overall),
        "fbref_id": None if pd.isna(fbref_id) else fbref_id,
        "club": club,
    }


# ------------------------------------------------------ per-player rates

def blend_rates(df):
    """df: rows of player_season_stats for ONE player. Returns per-90 rates or None."""
    df = df[df["season"].isin(SEASON_ORDER)].copy()
    if df.empty:
        return None
    df["nineties"] = df["minutes"] / 90
    if df["nineties"].sum() < MIN_NINETIES_FOR_HISTORY:
        return None

    df["weight"] = df["season"].apply(recency_weight) * df["nineties"]
    total_weight = df["weight"].sum()
    if not total_weight > 0:
        return None

    def per90_rate(col):
        per90 = (df[col] / df["nineties"].replace(0, np.nan)).fillna(0)
        return float((per90 * df["weight"]).sum() / total_weight)

    return {
        "goals": per90_rate("goals"),
        "assists": per90_rate("assists"),
        "yellow": per90_rate("yellow_cards"),
        "red": per90_rate("red_cards"),
        "source": "history",
    }


def get_blended_player_rates(fbref_id):
    """Single-player lookup (used by the single-match demo)."""
    if not fbref_id:
        return None
    df = pd.read_sql(text("""
        SELECT season, minutes, goals, assists, yellow_cards, red_cards
        FROM player_season_stats WHERE fbref_id = :fbref_id
    """), engine, params={"fbref_id": fbref_id})
    return None if df.empty else blend_rates(df)


def get_fallback_rates(line, overall_rating):
    base = FALLBACK_BASE.get(line, FALLBACK_BASE["C"])
    scale = (overall_rating or OVERALL_BASELINE) / OVERALL_BASELINE
    return {
        "goals": base["goals"] * scale,
        "assists": base["assists"] * scale,
        "yellow": base["yellow"],  # cards not scaled by overall, not a skill signal
        "red": base["red"],
        "source": "fallback",
    }


def get_player_rates(player):
    """player: dict with fbref_id, line, overall_rating."""
    rates = get_blended_player_rates(player.get("fbref_id"))
    if rates is None:
        rates = get_fallback_rates(player["line"], player.get("overall_rating"))
    return rates


def attach_rates(players):
    """Same result as get_player_rates for each player, but ONE query for everyone."""
    ids = sorted({p["fbref_id"] for p in players if p.get("fbref_id")})
    history = {}
    if ids:
        df = read_in("""
            SELECT fbref_id, season, minutes, goals, assists, yellow_cards, red_cards
            FROM player_season_stats WHERE fbref_id IN :ids
        """, ids=ids)
        for fbref_id, group in df.groupby("fbref_id"):
            rates = blend_rates(group)
            if rates:
                history[fbref_id] = rates
    for p in players:
        p["rates"] = history.get(p.get("fbref_id")) or get_fallback_rates(p["line"], p.get("overall_rating"))


# ------------------------------------------------- events and the grades

def attribute_team_events(team_players, team_goals, assist_ratio=0.65, rng=None):
    """
    team_players: list of dicts, each with player_id, line, rates
    Returns: dict player_id -> {"goals": int, "assists": int, "yellow": int, "red": int}
    """
    rng = rng or np.random.default_rng()
    events = {p["player_id"]: {"goals": 0, "assists": 0, "yellow": 0, "red": 0} for p in team_players}
    if not team_players:
        return events

    # --- goals ---
    goal_weights = np.array([max(p["rates"]["goals"], 1e-6) for p in team_players])
    goal_weights = goal_weights / goal_weights.sum()
    if team_goals > 0:
        scorer_idxs = rng.choice(len(team_players), size=team_goals, p=goal_weights)
        for idx in scorer_idxs:
            scorer_id = team_players[idx]["player_id"]
            events[scorer_id]["goals"] += 1

            if rng.random() < assist_ratio:
                assist_weights = np.array([
                    max(p["rates"]["assists"], 1e-6) if p["player_id"] != scorer_id else 0
                    for p in team_players
                ])
                if assist_weights.sum() > 0:
                    assist_weights = assist_weights / assist_weights.sum()
                    assist_idx = rng.choice(len(team_players), p=assist_weights)
                    events[team_players[assist_idx]["player_id"]]["assists"] += 1

    # --- cards (independent per player) ---
    for p in team_players:
        yellow = rng.poisson(p["rates"]["yellow"])
        red = rng.poisson(p["rates"]["red"])
        events[p["player_id"]]["yellow"] += min(yellow, 2)  # cap, 2 yellows = red in reality but keep simple
        events[p["player_id"]]["red"] += min(red, 1)

    return events


def compute_grade(player, player_events, team_won, team_drew, goals_conceded, rng=None):
    rng = rng or np.random.default_rng()

    goals = player_events["goals"]
    assists = player_events["assists"]
    yellow = player_events["yellow"]
    red = player_events["red"]

    event_bump = (0.5 if goals > 0 else 0) + (0.3 if assists > 0 else 0)
    team_result_adjustment = 0.3 if team_won else (-0.3 if not team_drew else 0)
    noise = rng.normal(0, 0.3)

    bonus_malus = (
        BONUS_GOAL * goals
        + BONUS_ASSIST * assists
        + MALUS_YELLOW * yellow
        + MALUS_RED * red
    )
    if player["line"] == "P":
        bonus_malus += MALUS_CONCEDED_PER_GOAL * goals_conceded

    raw = 6.0 + event_bump + team_result_adjustment + noise + bonus_malus
    return round_half(raw)


# ------------------------------------------------------- real lineups

def load_squad_pool(season, squads):
    """One query: every mapped player with a FBref row this season, deduped to ONE club each
    (the club where he has played the most minutes this season), then filtered to `squads`."""
    df = pd.read_sql(text("""
        SELECT s.squad, p.player_id, p.long_name, p.player_positions, p.overall,
               m.fbref_id, s.minutes
        FROM players p
        JOIN player_id_map m ON m.fantacalcio_player_id = p.player_id
        JOIN player_season_stats s ON s.fbref_id = m.fbref_id
        WHERE s.season = :season
    """), engine, params={"season": season})
    df["minutes"] = df["minutes"].fillna(0)
    df = df.sort_values("minutes", ascending=False).drop_duplicates(subset="player_id", keep="first")
    df = add_line_column(df).dropna(subset=["line"])
    return df[df["squad"].isin(squads)]


def pick_auto_xi(pool, formation_counts, squad):
    """No lineup provided: best players by overall for each line, with fallback A -> C -> D."""
    pool = pool.sort_values("overall", ascending=False)
    fallback = {"P": ["P"], "D": ["D"], "C": ["C", "D"], "A": ["A", "C", "D"]}

    used, xi = set(), []
    for slot_line, needed in formation_counts.items():
        remaining = needed
        for source_line in fallback[slot_line]:
            if remaining <= 0:
                break
            picks = pool[(pool["line"] == source_line) & (~pool["player_id"].isin(used))].head(remaining)
            for _, row in picks.iterrows():
                player = make_player(row, squad)
                player["slot"] = slot_line
                xi.append(player)
                used.add(int(row["player_id"]))
                remaining -= 1
        if remaining > 0:
            print(f"WARNING: {squad} is short {remaining} player(s) for line {slot_line}")
    return xi


def build_demo_xi(squad, formation_counts, season="2026-27"):
    """Auto XI for a single club (used by the single-match demo)."""
    return pick_auto_xi(load_squad_pool(season, [squad]), formation_counts, squad)


def load_player_details(player_ids, season):
    df = read_in("""
        SELECT p.player_id, p.long_name, p.player_positions, p.overall, m.fbref_id,
               COALESCE(SUM(s.minutes), 0) AS minutes
        FROM players p
        LEFT JOIN player_id_map m ON m.fantacalcio_player_id = p.player_id
        LEFT JOIN player_season_stats s ON s.fbref_id = m.fbref_id AND s.season = :season
        WHERE p.player_id IN :ids
        GROUP BY p.player_id, p.long_name, p.player_positions, p.overall, m.fbref_id
    """, season=season, ids=[int(i) for i in player_ids])
    df = df.sort_values("minutes", ascending=False).drop_duplicates(subset="player_id", keep="first")
    df = add_line_column(df)
    return {int(r["player_id"]): r for _, r in df.iterrows()}


def build_provided_xi(squad, player_ids, details):
    xi = []
    for pid in player_ids:
        row = details.get(int(pid))
        if row is None:
            print(f"WARNING: player_id {pid} (in {squad} lineup) not found in players table, skipped")
        elif pd.isna(row["line"]):
            print(f"WARNING: {row['long_name']} (in {squad} lineup) has an unknown position, skipped")
        else:
            xi.append(make_player(row, squad))
    goalkeepers = sum(1 for p in xi if p["line"] == "P")
    if len(xi) != 11 or goalkeepers != 1:
        print(f"WARNING: {squad} lineup has {len(xi)} players and {goalkeepers} goalkeeper(s) (expected 11 and 1)")
    return xi


def resolve_lineups(raw):
    """raw: {club: [player_id or name, ...]}  ->  {club: [player_id, ...]}
    A name must match exactly one player (accents/case ignored); otherwise use the player_id."""
    index = None
    resolved = {}
    for club, entries in raw.items():
        ids = []
        for entry in entries:
            if isinstance(entry, int) or (isinstance(entry, str) and entry.strip().isdigit()):
                ids.append(int(entry))
                continue
            if index is None:
                df = pd.read_sql(text("SELECT player_id, long_name FROM players"), engine)
                index = [(int(r.player_id), r.long_name, normalize_name(r.long_name)) for r in df.itertuples()]
            query = normalize_name(entry)
            matches = [m for m in index if m[2] == query] or [m for m in index if query in m[2]]
            if not matches:
                raise ValueError(f"{club}: no player matches {entry!r}")
            if len(matches) > 1:
                options = ", ".join(f"{name} ({pid})" for pid, name, _ in matches[:6])
                raise ValueError(f"{club}: {entry!r} is ambiguous ({options}). Use the player_id instead.")
            ids.append(matches[0][0])
        resolved[club] = ids
    return resolved


def load_lineups_file(path):
    """JSON: {"Inter": ["Josep Martínez", 224232, ...], "Milan": [...]}
    Each entry is a player_id or a full/partial name. Club names as in the calendar."""
    with open(path, encoding="utf-8") as f:
        return resolve_lineups(json.load(f))


# --------------------------------------- prepare once, simulate many

def load_team_strengths():
    df = pd.read_sql(text("SELECT squad, attack_strength, defense_strength FROM team_strength_current"), engine)
    return {r.squad: (float(r.attack_strength), float(r.defense_strength)) for r in df.itertuples()}


def prepare_giornata(season, giornata, fantasy_player_ids=None, lineups=None,
                     formation=DEFAULT_REAL_FORMATION):
    """ALL database access for a simulation happens here, once.
    If fantasy_player_ids is given, only real fixtures involving one of those players are kept."""
    fixtures_df = pd.read_sql(text("""
        SELECT home_team, away_team FROM calendar_2627
        WHERE season = :season AND giornata = :giornata
        ORDER BY id
    """), engine, params={"season": season, "giornata": giornata})
    if fixtures_df.empty:
        raise ValueError(f"No fixtures for {season} giornata {giornata}")

    squads = sorted(set(fixtures_df["home_team"]) | set(fixtures_df["away_team"]))
    canonical = {s.lower(): s for s in squads}

    provided = {}
    for key, ids in (lineups or {}).items():
        squad = canonical.get(key.lower())
        if squad is None:
            print(f"NOTE: ignoring lineup for '{key}' (not playing in giornata {giornata})")
        else:
            provided[squad] = ids

    xis, sources = {}, {}
    if provided:
        details = load_player_details(sorted({int(p) for ids in provided.values() for p in ids}), season)
        for squad, ids in provided.items():
            xis[squad], sources[squad] = build_provided_xi(squad, ids, details), "provided"

    auto_squads = [s for s in squads if s not in xis]
    if auto_squads:
        pool = load_squad_pool(season, auto_squads)
        for squad in auto_squads:
            xis[squad] = pick_auto_xi(pool[pool["squad"] == squad], formation, squad)
            sources[squad] = "auto"

    strengths = load_team_strengths()
    league_avg = get_league_avg_goals()

    fixtures = []
    for f in fixtures_df.itertuples():
        for squad in (f.home_team, f.away_team):
            if squad not in strengths:
                raise ValueError(f"No strength rating found for squad: {squad}")
        home_att, home_def = strengths[f.home_team]
        away_att, away_def = strengths[f.away_team]
        fixtures.append({
            "home": f.home_team,
            "away": f.away_team,
            "home_xg": league_avg * home_att * away_def * HOME_ADVANTAGE,
            "away_xg": league_avg * away_att * home_def,
            "home_xi": xis[f.home_team],
            "away_xi": xis[f.away_team],
            "home_src": sources[f.home_team],
            "away_src": sources[f.away_team],
        })

    total_fixtures = len(fixtures)
    if fantasy_player_ids is not None:
        fixtures = [
            fx for fx in fixtures
            if any(p["player_id"] in fantasy_player_ids for p in fx["home_xi"] + fx["away_xi"])
        ]

    attach_rates([p for fx in fixtures for p in fx["home_xi"] + fx["away_xi"]])
    return {"fixtures": fixtures, "total_fixtures": total_fixtures}


def run_simulation(fixtures, rng):
    """One simulated giornata. No database access. Returns ({player_id: grade}, scorelines)."""
    grades, scorelines = {}, []
    for fx in fixtures:
        home_goals = int(rng.poisson(fx["home_xg"]))
        away_goals = int(rng.poisson(fx["away_xg"]))
        scorelines.append((fx["home"], fx["away"], home_goals, away_goals))

        home_events = attribute_team_events(fx["home_xi"], home_goals, rng=rng)
        away_events = attribute_team_events(fx["away_xi"], away_goals, rng=rng)

        home_won, away_won = home_goals > away_goals, away_goals > home_goals
        drew = home_goals == away_goals

        for p in fx["home_xi"]:
            grades[p["player_id"]] = compute_grade(p, home_events[p["player_id"]], home_won, drew, away_goals, rng=rng)
        for p in fx["away_xi"]:
            grades[p["player_id"]] = compute_grade(p, away_events[p["player_id"]], away_won, drew, home_goals, rng=rng)
    return grades, scorelines


# ------------------------------------------------- fantasy Monte Carlo

def load_fantasy_team(name):
    df = pd.read_sql(text("SELECT id, name, player_ids FROM teams WHERE name = :name"),
                     engine, params={"name": name})
    if df.empty:
        raise ValueError(f"No fantasy team named {name!r}")
    return df.iloc[0]


def parse_player_ids(value):
    if isinstance(value, str):
        value = json.loads(value)
    return [int(v) for v in value]


def fantasy_monte_carlo(name_a, name_b, season, giornata, num_sims=1000, lineups=None, seed=None):
    if num_sims < 1:
        raise ValueError("--sims must be at least 1")

    ids_a = parse_player_ids(load_fantasy_team(name_a)["player_ids"])
    ids_b = parse_player_ids(load_fantasy_team(name_b)["player_ids"])
    fantasy_ids = set(ids_a) | set(ids_b)

    prepared = prepare_giornata(season, giornata, fantasy_player_ids=fantasy_ids, lineups=lineups)
    fixtures = prepared["fixtures"]
    rng = np.random.default_rng(seed)

    # per-fixture accumulator, one bucket per real match
    fixture_stats = [
        {"home": fx["home"], "away": fx["away"], "home_src": fx["home_src"], "away_src": fx["away_src"],
         "home_goals": [], "away_goals": []}
        for fx in fixtures
    ]
    last_scorelines = None

    scores_a, scores_b = np.zeros(num_sims), np.zeros(num_sims)
    grade_sums = defaultdict(float)

    started = time.perf_counter()
    for i in range(num_sims):
        grades, sim_scorelines = run_simulation(fixtures, rng)
        scores_a[i] = sum(grades.get(pid, 0.0) for pid in ids_a)
        scores_b[i] = sum(grades.get(pid, 0.0) for pid in ids_b)
        for pid in fantasy_ids:
            if pid in grades:
                grade_sums[pid] += grades[pid]
        for j, (_, _, hg, ag) in enumerate(sim_scorelines):
            fixture_stats[j]["home_goals"].append(hg)
            fixture_stats[j]["away_goals"].append(ag)
        last_scorelines = sim_scorelines
    elapsed = time.perf_counter() - started

    goals_a = np.array([grades_to_goals(s) for s in scores_a])
    goals_b = np.array([grades_to_goals(s) for s in scores_b])
    top_score, top_count = Counter(zip(goals_a.tolist(), goals_b.tolist())).most_common(1)[0]

    # summarize each real fixture across all sims
    fixture_summary = []
    for fs in fixture_stats:
        hg = np.array(fs["home_goals"])
        ag = np.array(fs["away_goals"])
        n = len(hg)
        ts, tc = Counter(zip(hg.tolist(), ag.tolist())).most_common(1)[0]
        fixture_summary.append({
            "home": fs["home"], "away": fs["away"],
            "home_src": fs["home_src"], "away_src": fs["away_src"],
            "avg_home_goals": float(hg.mean()),
            "avg_away_goals": float(ag.mean()),
            "home_win_pct": 100 * int((hg > ag).sum()) / n,
            "draw_pct":     100 * int((hg == ag).sum()) / n,
            "away_win_pct": 100 * int((hg < ag).sum()) / n,
            "top_scoreline": (int(ts[0]), int(ts[1]), 100 * tc / n),
        })

    club_of = {p["player_id"]: p["club"] for fx in fixtures for p in fx["home_xi"] + fx["away_xi"]}
    names_df = read_in("SELECT player_id, long_name FROM players WHERE player_id IN :ids",
                       ids=sorted(fantasy_ids))
    names = {int(r.player_id): r.long_name for r in names_df.itertuples()}

    def team_block(name, ids, scores, goals, own_wins):
        return {
            "name": name,
            "avg_points": float(scores.mean()),
            "std_points": float(scores.std()),
            "avg_goals": float(goals.mean()),
            "win_pct": 100 * own_wins / num_sims,
            "players": [{
                "player_id": pid,
                "name": names.get(pid, str(pid)),
                "club": club_of.get(pid),
                "avg_grade": grade_sums[pid] / num_sims if pid in club_of else None,
            } for pid in ids],
        }

    return {
        "season": season,
        "giornata": giornata,
        "num_sims": num_sims,
        "elapsed_seconds": elapsed,
        "fixture_summary": fixture_summary,
        "total_fixtures": prepared["total_fixtures"],
        "sample_scorelines": last_scorelines,   # the last sim's exact scorelines
        "a": team_block(name_a, ids_a, scores_a, goals_a, int((goals_a > goals_b).sum())),
        "b": team_block(name_b, ids_b, scores_b, goals_b, int((goals_b > goals_a).sum())),
        "draw_pct": 100 * int((goals_a == goals_b).sum()) / num_sims,
        "top_scoreline": (int(top_score[0]), int(top_score[1]), 100 * top_count / num_sims),
        "more_points_a_pct": 100 * int((scores_a > scores_b).sum()) / num_sims,
        "more_points_b_pct": 100 * int((scores_b > scores_a).sum()) / num_sims,
    }


def print_report(r):
    a, b = r["a"], r["b"]
    print(f"\n=== {a['name']} vs {b['name']} | {r['season']} giornata {r['giornata']} | {r['num_sims']} simulation(s) ===")
    print(f"Real fixtures simulated: {len(r['fixture_summary'])} of {r['total_fixtures']} "
          f"(only those with players from these teams)")

    print("\nReal-fixture outcomes (averaged over all simulations):")
    print(f"  {'Match':38s} {'avg':>9s}   {'H%':>5s} {'D%':>5s} {'A%':>5s}   {'top score':>10s}")
    for fs in r["fixture_summary"]:
        label = f"{fs['home']} vs {fs['away']}"
        avg = f"{fs['avg_home_goals']:.2f}-{fs['avg_away_goals']:.2f}"
        top_hg, top_ag, top_pct = fs["top_scoreline"]
        top_pct = fs["top_scoreline"][1]
        print(f"  {label:38s} {avg:>9s}   "
              f"{fs['home_win_pct']:>5.1f} {fs['draw_pct']:>5.1f} {fs['away_win_pct']:>5.1f}   "
              f"{top_hg}-{top_ag} ({top_pct:.0f}%)")

    if r["sample_scorelines"]:
        print("\nSample simulation (last run):")
        for home, away, hg, ag in r["sample_scorelines"]:
            print(f"  {home} {hg} - {ag} {away}")

    print("\nFantasy result (points converted to goals)")
    for t in (a, b):
        print(f"  {t['name']}: avg {t['avg_points']:.1f} pts (sd {t['std_points']:.1f}), avg {t['avg_goals']:.2f} goals")
    print(f"  {a['name']} wins {a['win_pct']:.1f}% | draws {r['draw_pct']:.1f}% | {b['name']} wins {b['win_pct']:.1f}%")
    ga, gb, pct = r["top_scoreline"]
    print(f"  most common result: {ga}-{gb} ({pct:.1f}%)")
    print(f"  more raw points: {a['name']} {r['more_points_a_pct']:.1f}% | {b['name']} {r['more_points_b_pct']:.1f}%")

    for t in (a, b):
        print(f"\n--- {t['name']}: average grade per player ---")
        for p in t["players"]:
            if p["avg_grade"] is None:
                print(f"  {p['name']:32s} not in a real lineup, counts 0")
            else:
                print(f"  {p['name']:32s} {p['avg_grade']:5.2f}   ({p['club']})")
    print(f"\nRan {r['num_sims']} simulation(s) in {r['elapsed_seconds']:.2f}s")


# ---------------------------------------------- single real match demo

def simulate_one_match_goals(home_team, away_team, rng=None):
    """Single Poisson draw (not the aggregated Monte Carlo summary from match_simulator)."""
    rng = rng or np.random.default_rng()
    home_attack, home_defense = get_team_strength(home_team)
    away_attack, away_defense = get_team_strength(away_team)
    league_avg = get_league_avg_goals()

    home_xg = league_avg * home_attack * away_defense * HOME_ADVANTAGE
    away_xg = league_avg * away_attack * home_defense
    return int(rng.poisson(home_xg)), int(rng.poisson(away_xg))


def run_single_match_demo(home, away):
    rng = np.random.default_rng()
    home_xi = build_demo_xi(home, DEFAULT_REAL_FORMATION)
    away_xi = build_demo_xi(away, DEFAULT_REAL_FORMATION)
    for p in home_xi + away_xi:
        p["rates"] = get_player_rates(p)

    home_goals, away_goals = simulate_one_match_goals(home, away, rng=rng)
    print(f"{home} {home_goals} - {away_goals} {away}\n")

    home_events = attribute_team_events(home_xi, home_goals, rng=rng)
    away_events = attribute_team_events(away_xi, away_goals, rng=rng)
    home_won, away_won = home_goals > away_goals, away_goals > home_goals
    drew = home_goals == away_goals

    for label, xi, events, won, conceded in [
        (home, home_xi, home_events, home_won, away_goals),
        (away, away_xi, away_events, away_won, home_goals),
    ]:
        print(f"--- {label} ---")
        for p in xi:
            ev = events[p["player_id"]]
            grade = compute_grade(p, ev, won, drew, conceded, rng=rng)
            print(f"{p['long_name']:25s} {p['line']}  G:{ev['goals']} A:{ev['assists']} "
                  f"Y:{ev['yellow']} R:{ev['red']}  -> {grade}  ({p['rates']['source']})")
        print()


# ------------------------------------------------------------------ CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fantasy Monte Carlo / single match demo")
    parser.add_argument("teams", nargs="*", help="single-match demo: HOME AWAY")
    parser.add_argument("--fantasy", nargs=2, metavar=("TEAM_A", "TEAM_B"))
    parser.add_argument("--giornata", type=int)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--sims", type=int, default=1000, help="number of simulations to run")
    parser.add_argument("--lineups", help="JSON file with the real XI per club")
    parser.add_argument("--seed", type=int, help="fix the random seed for a reproducible run")
    args = parser.parse_args()

    if args.fantasy:
        if args.giornata is None:
            raise SystemExit("--giornata <n> is required")
        lineups = load_lineups_file(args.lineups) if args.lineups else None
        result = fantasy_monte_carlo(
            args.fantasy[0], args.fantasy[1], args.season, args.giornata,
            num_sims=args.sims, lineups=lineups, seed=args.seed,
        )
        print_report(result)
    else:
        home = args.teams[0] if len(args.teams) > 0 else "Inter"
        away = args.teams[1] if len(args.teams) > 1 else "Milan"
        run_single_match_demo(home, away)