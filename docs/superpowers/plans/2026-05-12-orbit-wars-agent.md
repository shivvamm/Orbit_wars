# Orbit Wars Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a competitive heuristic-based adaptive AI agent for the Orbit Wars Kaggle competition, submitted as a single `main.py`.

**Architecture:** Single-file agent with six logical sections: GameState parsing, OrbitalPredictor, ThreatAnalyzer, TargetEvaluator, FleetCoordinator, and PhaseManager. The `agent(obs)` entry point orchestrates them each turn within a 1-second budget. No external dependencies beyond `math`.

**Tech Stack:** Python 3, `math` stdlib, `kaggle_environments` named tuples (Planet, Fleet)

---

### Task 1: Core Game State Parsing & Utility Functions

**Files:**
- Create: `main.py`
- Test: `test_agent.py`

This task builds the foundation: observation parsing, fleet speed calculation, distance helpers, sun-crossing detection, and orbital position prediction. Everything else depends on these.

- [ ] **Step 1: Write failing tests for utility functions**

```python
# test_agent.py
import math
import pytest

# We'll import from main once it exists
from main import (
    fleet_speed,
    travel_time,
    dist,
    crosses_sun,
    predict_planet_position,
    angle_to_target,
)

def test_fleet_speed_single_ship():
    assert fleet_speed(1) == pytest.approx(1.0)

def test_fleet_speed_1000_ships():
    assert fleet_speed(1000) == pytest.approx(6.0)

def test_fleet_speed_scales_logarithmically():
    s10 = fleet_speed(10)
    s100 = fleet_speed(100)
    s500 = fleet_speed(500)
    assert 1.0 < s10 < s100 < s500 < 6.0

def test_travel_time():
    d = 30.0
    ships = 100
    speed = fleet_speed(ships)
    expected = d / speed
    assert travel_time(d, ships) == pytest.approx(expected)

def test_dist():
    assert dist(0, 0, 3, 4) == pytest.approx(5.0)
    assert dist(50, 50, 50, 50) == pytest.approx(0.0)

def test_crosses_sun_direct_path():
    # Path from (20, 50) to (80, 50) crosses through center sun
    assert crosses_sun(20, 50, 80, 50) is True

def test_crosses_sun_safe_path():
    # Path along top edge, nowhere near sun
    assert crosses_sun(10, 10, 90, 10) is False

def test_crosses_sun_tangent():
    # Path that just barely misses the sun (distance > 10 from center)
    assert crosses_sun(10, 39, 90, 39) is False

def test_predict_planet_position_static():
    # Planet far from center (orbital_radius + radius >= 50) doesn't rotate
    ix, iy = 80.0, 80.0
    x, y = predict_planet_position(ix, iy, 2.0, 0.03, 100)
    assert x == pytest.approx(80.0)
    assert y == pytest.approx(80.0)

def test_predict_planet_position_orbiting():
    # Planet near center rotates
    ix, iy = 65.0, 50.0  # orbital_radius=15, radius~2.6 => 15+2.6<50 => orbits
    radius = 1 + math.log(5)  # prod=5 => ~2.6
    x, y = predict_planet_position(ix, iy, radius, 0.04, 10)
    orbital_r = math.hypot(ix - 50, iy - 50)  # 15
    init_angle = math.atan2(iy - 50, ix - 50)  # 0
    new_angle = init_angle + 0.04 * 10
    assert x == pytest.approx(50 + orbital_r * math.cos(new_angle))
    assert y == pytest.approx(50 + orbital_r * math.sin(new_angle))

def test_angle_to_target():
    # From (0,0) to (1,0) should be angle 0
    assert angle_to_target(0, 0, 1, 0) == pytest.approx(0.0)
    # From (0,0) to (0,1) should be pi/2
    assert angle_to_target(0, 0, 0, 1) == pytest.approx(math.pi / 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v`
Expected: FAIL — `main` module not found or functions not defined

- [ ] **Step 3: Implement utility functions in main.py**

```python
# main.py
import math

CENTER = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
TOTAL_STEPS = 500


def dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(s, MAX_SPEED)


def travel_time(distance, ships):
    return distance / fleet_speed(ships)


def point_to_segment_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return dist(px, py, ax, ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / l2))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return dist(px, py, proj_x, proj_y)


def crosses_sun(x1, y1, x2, y2):
    return point_to_segment_dist(CENTER, CENTER, x1, y1, x2, y2) < SUN_RADIUS


def predict_planet_position(init_x, init_y, radius, angular_velocity, turn):
    dx = init_x - CENTER
    dy = init_y - CENTER
    orbital_r = math.hypot(dx, dy)
    if orbital_r + radius >= ROTATION_RADIUS_LIMIT:
        return init_x, init_y
    init_angle = math.atan2(dy, dx)
    new_angle = init_angle + angular_velocity * turn
    return CENTER + orbital_r * math.cos(new_angle), CENTER + orbital_r * math.sin(new_angle)


def angle_to_target(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add main.py test_agent.py
git commit -m "feat: add core utility functions — fleet speed, distance, sun crossing, orbital prediction"
```

---

### Task 2: Game State Parser & Phase Manager

**Files:**
- Modify: `main.py`
- Modify: `test_agent.py`

Parse the raw observation into structured data. Detect game phase (early/mid/late) and player count. Track the current turn number.

- [ ] **Step 1: Write failing tests**

```python
# append to test_agent.py
from main import parse_game_state

def _make_obs(player=0, planets=None, fleets=None, angular_velocity=0.03,
              initial_planets=None, comets=None, comet_planet_ids=None, step=0):
    return {
        "player": player,
        "planets": planets or [],
        "fleets": fleets or [],
        "angular_velocity": angular_velocity,
        "initial_planets": initial_planets or planets or [],
        "comets": comets or [],
        "comet_planet_ids": comet_planet_ids or [],
        "step": step,
        "remainingOverageTime": 60.0,
    }

def test_parse_game_state_basic():
    planets = [
        [0, 0, 80, 80, 2.0, 10, 3],   # mine
        [1, -1, 70, 70, 1.5, 20, 2],   # neutral
        [2, 1, 20, 20, 2.0, 15, 3],    # enemy
    ]
    obs = _make_obs(player=0, planets=planets, step=10)
    gs = parse_game_state(obs)
    assert gs["player"] == 0
    assert gs["turn"] == 10
    assert len(gs["my_planets"]) == 1
    assert len(gs["enemy_planets"]) == 1
    assert len(gs["neutral_planets"]) == 1
    assert gs["my_planets"][0].id == 0

def test_parse_game_state_phase_early():
    obs = _make_obs(step=5)
    gs = parse_game_state(obs)
    assert gs["phase"] == "early"

def test_parse_game_state_phase_mid():
    obs = _make_obs(step=150)
    gs = parse_game_state(obs)
    assert gs["phase"] == "mid"

def test_parse_game_state_phase_late():
    obs = _make_obs(step=400)
    gs = parse_game_state(obs)
    assert gs["phase"] == "late"

def test_parse_game_state_num_players():
    planets = [
        [0, 0, 80, 80, 2.0, 10, 3],
        [1, 1, 20, 20, 2.0, 10, 3],
        [2, 2, 20, 80, 2.0, 10, 3],
        [3, 3, 80, 20, 2.0, 10, 3],
    ]
    obs = _make_obs(planets=planets)
    gs = parse_game_state(obs)
    assert gs["num_players"] == 4

def test_parse_game_state_enemy_strength():
    planets = [
        [0, 0, 80, 80, 2.0, 10, 3],
        [1, 1, 20, 20, 2.0, 30, 3],
    ]
    fleets = [
        [0, 1, 50, 50, 0.0, 1, 20],
    ]
    obs = _make_obs(planets=planets, fleets=fleets)
    gs = parse_game_state(obs)
    assert gs["enemy_strength"][1] == 50  # 30 on planet + 20 in fleet
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v -k "parse_game_state"`
Expected: FAIL — `parse_game_state` not defined

- [ ] **Step 3: Implement parse_game_state**

Add to `main.py`:

```python
from collections import namedtuple

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])


def parse_game_state(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else getattr(obs, "player", 0)
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    angular_velocity = obs.get("angular_velocity", 0.03) if isinstance(obs, dict) else getattr(obs, "angular_velocity", 0.03)
    raw_initial = obs.get("initial_planets", raw_planets) if isinstance(obs, dict) else getattr(obs, "initial_planets", raw_planets)
    raw_comets = obs.get("comets", []) if isinstance(obs, dict) else getattr(obs, "comets", [])
    comet_ids = set(obs.get("comet_planet_ids", []) if isinstance(obs, dict) else getattr(obs, "comet_planet_ids", []))
    step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    initial_planets = {p[0]: p for p in raw_initial}

    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner != player and p.owner != -1]
    neutral_planets = [p for p in planets if p.owner == -1]
    my_fleets = [f for f in fleets if f.owner == player]
    enemy_fleets = [f for f in fleets if f.owner != player]

    owners = set()
    for p in planets:
        if p.owner >= 0:
            owners.add(p.owner)
    for f in fleets:
        owners.add(f.owner)
    num_players = max(len(owners), 2)

    enemy_strength = {}
    for p in planets:
        if p.owner >= 0 and p.owner != player:
            enemy_strength[p.owner] = enemy_strength.get(p.owner, 0) + p.ships
    for f in fleets:
        if f.owner != player:
            enemy_strength[f.owner] = enemy_strength.get(f.owner, 0) + f.ships

    my_total = sum(p.ships for p in my_planets) + sum(f.ships for f in my_fleets)

    if step < 80:
        phase = "early"
    elif step < 300:
        phase = "mid"
    else:
        phase = "late"

    return {
        "player": player,
        "turn": step,
        "phase": phase,
        "num_players": num_players,
        "planets": planets,
        "fleets": fleets,
        "my_planets": my_planets,
        "enemy_planets": enemy_planets,
        "neutral_planets": neutral_planets,
        "my_fleets": my_fleets,
        "enemy_fleets": enemy_fleets,
        "enemy_strength": enemy_strength,
        "my_total": my_total,
        "angular_velocity": angular_velocity,
        "initial_planets": initial_planets,
        "comet_ids": comet_ids,
        "comets": raw_comets,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add main.py test_agent.py
git commit -m "feat: add game state parser with phase detection and player strength tracking"
```

---

### Task 3: Threat Analyzer

**Files:**
- Modify: `main.py`
- Modify: `test_agent.py`

Detect enemy fleets heading toward our planets, estimate arrival turns, and compute reinforcement needs.

- [ ] **Step 1: Write failing tests**

```python
# append to test_agent.py
from main import analyze_threats

def test_analyze_threats_no_enemies():
    my_planets = [Planet(0, 0, 80, 80, 2.0, 10, 3)]
    threats = analyze_threats(my_planets, [], 0.03, {}, 10)
    assert threats == {}

def test_analyze_threats_incoming_fleet():
    my_planets = [Planet(0, 0, 80, 80, 2.0, 10, 3)]
    # Fleet heading directly at planet from (60, 80), angle=0 (rightward)
    enemy_fleets = [Fleet(1, 1, 60, 80, 0.0, 99, 20)]
    threats = analyze_threats(my_planets, enemy_fleets, 0.03, {}, 10)
    assert 0 in threats
    assert len(threats[0]) > 0
    assert threats[0][0]["fleet"].id == 1
    assert threats[0][0]["ships"] == 20

def test_analyze_threats_fleet_misses():
    my_planets = [Planet(0, 0, 80, 80, 2.0, 10, 3)]
    # Fleet heading upward, will miss the planet
    enemy_fleets = [Fleet(1, 1, 60, 60, -math.pi / 2, 99, 20)]
    threats = analyze_threats(my_planets, enemy_fleets, 0.03, {}, 10)
    assert threats.get(0, []) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v -k "analyze_threats"`
Expected: FAIL — `analyze_threats` not defined

- [ ] **Step 3: Implement analyze_threats**

Add to `main.py`:

```python
def analyze_threats(my_planets, enemy_fleets, angular_velocity, initial_planets, turn):
    threats = {}
    for mp in my_planets:
        threats[mp.id] = []

    for ef in enemy_fleets:
        speed = fleet_speed(ef.ships)
        fx, fy = ef.x, ef.y
        cos_a = math.cos(ef.angle)
        sin_a = math.sin(ef.angle)

        for mp in my_planets:
            init_p = initial_planets.get(mp.id)
            if init_p is not None:
                init_x, init_y = init_p[2], init_p[3]
            else:
                init_x, init_y = mp.x, mp.y

            best_dist = float("inf")
            best_eta = None
            for future_t in range(1, 80):
                fleet_x = fx + cos_a * speed * future_t
                fleet_y = fy + sin_a * speed * future_t
                if not (0 <= fleet_x <= BOARD_SIZE and 0 <= fleet_y <= BOARD_SIZE):
                    break
                px, py = predict_planet_position(init_x, init_y, mp.radius, angular_velocity, turn + future_t)
                d = dist(fleet_x, fleet_y, px, py)
                if d < mp.radius + 0.5:
                    best_dist = d
                    best_eta = future_t
                    break
                if d < best_dist:
                    best_dist = d

            if best_eta is not None:
                threats[mp.id].append({
                    "fleet": ef,
                    "ships": ef.ships,
                    "eta": best_eta,
                    "urgency": "imminent" if best_eta <= 3 else ("medium" if best_eta <= 8 else "distant"),
                })

    return threats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add main.py test_agent.py
git commit -m "feat: add threat analyzer — detects enemy fleets heading toward owned planets"
```

---

### Task 4: Target Evaluator

**Files:**
- Modify: `main.py`
- Modify: `test_agent.py`

Score every non-owned planet for attack priority based on production value, garrison cost, distance, and strategic position.

- [ ] **Step 1: Write failing tests**

```python
# append to test_agent.py
from main import evaluate_targets

def test_evaluate_targets_prefers_high_production():
    my_planets = [Planet(0, 0, 80, 80, 2.0, 50, 3)]
    targets = [
        Planet(1, -1, 75, 75, 1.0, 5, 1),  # low prod, nearby
        Planet(2, -1, 75, 80, 2.6, 5, 5),  # high prod, nearby
    ]
    scored = evaluate_targets(my_planets, targets, [], "early", 0.03, {}, 10, 2, {})
    ids = [s["planet"].id for s in scored]
    assert ids[0] == 2  # high-prod planet ranked first

def test_evaluate_targets_accounts_for_garrison():
    my_planets = [Planet(0, 0, 80, 80, 2.0, 50, 3)]
    targets = [
        Planet(1, -1, 75, 75, 2.6, 5, 5),   # low garrison
        Planet(2, -1, 76, 76, 2.6, 80, 5),  # high garrison
    ]
    scored = evaluate_targets(my_planets, targets, [], "early", 0.03, {}, 10, 2, {})
    ids = [s["planet"].id for s in scored]
    assert ids[0] == 1  # cheaper to capture

def test_evaluate_targets_accounts_for_distance():
    my_planets = [Planet(0, 0, 80, 80, 2.0, 50, 3)]
    targets = [
        Planet(1, -1, 78, 78, 2.6, 5, 5),  # very close
        Planet(2, -1, 30, 30, 2.6, 5, 5),  # very far
    ]
    scored = evaluate_targets(my_planets, targets, [], "early", 0.03, {}, 10, 2, {})
    ids = [s["planet"].id for s in scored]
    assert ids[0] == 1  # closer planet ranked first

def test_evaluate_targets_empty():
    my_planets = [Planet(0, 0, 80, 80, 2.0, 50, 3)]
    scored = evaluate_targets(my_planets, [], [], "early", 0.03, {}, 10, 2, {})
    assert scored == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v -k "evaluate_targets"`
Expected: FAIL — `evaluate_targets` not defined

- [ ] **Step 3: Implement evaluate_targets**

Add to `main.py`:

```python
def evaluate_targets(my_planets, targets, enemy_fleets, phase, angular_velocity,
                     initial_planets, turn, num_players, enemy_strength):
    if not my_planets or not targets:
        return []

    scored = []
    for target in targets:
        init_p = initial_planets.get(target.id)
        if init_p is not None:
            t_init_x, t_init_y = init_p[2], init_p[3]
        else:
            t_init_x, t_init_y = target.x, target.y

        best_source = None
        best_dist = float("inf")
        for mp in my_planets:
            d = dist(mp.x, mp.y, target.x, target.y)
            if d < best_dist:
                best_dist = d
                best_source = mp

        ships_to_send = best_source.ships if best_source else 0
        tt = travel_time(best_dist, max(ships_to_send, 1))

        garrison_cost = target.ships
        if target.owner >= 0:
            garrison_cost += target.production * int(tt + 1)

        incoming_enemy = sum(
            ef.ships for ef in enemy_fleets
            if dist(ef.x, ef.y, target.x, target.y) < 30
            and abs(math.atan2(target.y - ef.y, target.x - ef.x) - ef.angle) < 0.5
        )

        prod_weight = target.production ** 1.5

        proximity_bonus = max(0.0, 1.0 - best_dist / 80.0)

        enemy_proximity = 0.0
        for ep_owner, strength in enemy_strength.items():
            if strength > 0:
                enemy_proximity += 1.0

        cluster_bonus = 0.0
        for other in targets:
            if other.id != target.id:
                d = dist(target.x, target.y, other.x, other.y)
                if d < 20:
                    cluster_bonus += other.production / (d + 1)

        neutral_bonus = 0.3 if target.owner == -1 and phase == "early" else 0.0

        effective_garrison = garrison_cost + incoming_enemy * 0.5

        score = (prod_weight * (1.0 + proximity_bonus + cluster_bonus * 0.1 + neutral_bonus)) / (tt + effective_garrison + 1)

        if num_players > 2 and target.owner >= 0:
            owner_str = enemy_strength.get(target.owner, 0)
            max_str = max(enemy_strength.values()) if enemy_strength else 1
            if owner_str >= max_str * 0.8:
                score *= 0.6

        scored.append({
            "planet": target,
            "score": score,
            "best_source": best_source,
            "garrison_cost": effective_garrison,
            "travel_time": tt,
        })

    scored.sort(key=lambda s: s["score"], reverse=True)
    return scored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add main.py test_agent.py
git commit -m "feat: add target evaluator — scores planets by production, garrison cost, distance"
```

---

### Task 5: Fleet Coordinator & Sun Avoidance

**Files:**
- Modify: `main.py`
- Modify: `test_agent.py`

Decide which planets send how many ships where. Handle sun avoidance, overkill prevention, garrison reserves, and evacuation.

- [ ] **Step 1: Write failing tests**

```python
# append to test_agent.py
from main import coordinate_fleets

def test_coordinate_fleets_basic_attack():
    my_planets = [Planet(0, 0, 80, 80, 2.0, 30, 3)]
    targets = [{
        "planet": Planet(1, -1, 75, 75, 1.5, 5, 2),
        "score": 1.0,
        "best_source": my_planets[0],
        "garrison_cost": 5,
        "travel_time": 1.5,
    }]
    moves = coordinate_fleets(my_planets, targets, {}, "early", 0.03, {}, 10, 2, 30)
    assert len(moves) >= 1
    assert moves[0][0] == 0  # from planet 0
    assert moves[0][2] > 5  # sends more than garrison
    assert moves[0][2] < 30  # doesn't send everything

def test_coordinate_fleets_respects_garrison_reserve():
    my_planets = [Planet(0, 0, 80, 80, 2.0, 10, 3)]
    targets = [{
        "planet": Planet(1, -1, 75, 75, 1.5, 5, 2),
        "score": 1.0,
        "best_source": my_planets[0],
        "garrison_cost": 5,
        "travel_time": 1.5,
    }]
    moves = coordinate_fleets(my_planets, targets, {}, "mid", 0.03, {}, 100, 2, 10)
    # With only 10 ships and needing a reserve, might not send at all or send few
    total_sent = sum(m[2] for m in moves)
    assert total_sent < 10  # can't send everything

def test_coordinate_fleets_sun_avoidance():
    # Planet at (20, 50) targeting (80, 50) — direct path crosses the sun
    my_planets = [Planet(0, 0, 20, 50, 2.0, 50, 3)]
    targets = [{
        "planet": Planet(1, -1, 80, 50, 1.5, 5, 2),
        "score": 1.0,
        "best_source": my_planets[0],
        "garrison_cost": 5,
        "travel_time": 10.0,
    }]
    moves = coordinate_fleets(my_planets, targets, {}, "early", 0.03, {}, 10, 2, 50)
    if len(moves) > 0:
        # The angle should NOT be direct (which would be 0.0)
        # It should be offset to go around the sun
        launch_angle = moves[0][1]
        end_x = 20 + math.cos(launch_angle) * 60
        end_y = 50 + math.sin(launch_angle) * 60
        # Check the adjusted path doesn't cross the sun
        assert not crosses_sun(20, 50, end_x, end_y)

def test_coordinate_fleets_evacuation():
    my_planets = [
        Planet(0, 0, 80, 80, 2.0, 10, 3),
        Planet(1, 0, 85, 85, 2.0, 30, 3),
    ]
    threats = {
        0: [{"fleet": Fleet(99, 1, 70, 80, 0.0, 50, 100), "ships": 100, "eta": 2, "urgency": "imminent"}],
        1: [],
    }
    moves = coordinate_fleets(my_planets, [], threats, "mid", 0.03, {}, 100, 2, 40)
    # Should evacuate planet 0 toward planet 1
    evac_moves = [m for m in moves if m[0] == 0]
    assert len(evac_moves) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v -k "coordinate_fleets"`
Expected: FAIL — `coordinate_fleets` not defined

- [ ] **Step 3: Implement coordinate_fleets**

Add to `main.py`:

```python
def coordinate_fleets(my_planets, scored_targets, threats, phase, angular_velocity,
                      initial_planets, turn, num_players, my_total):
    moves = []
    available = {p.id: p.ships for p in my_planets}
    planets_by_id = {p.id: p for p in my_planets}
    assigned_targets = set()

    garrison_factor = {"early": 0.2, "mid": 0.35, "late": 0.5}[phase]

    # 1. Handle evacuations for doomed planets
    for mp in my_planets:
        mp_threats = threats.get(mp.id, [])
        imminent = [t for t in mp_threats if t["urgency"] == "imminent"]
        if not imminent:
            continue
        total_incoming = sum(t["ships"] for t in imminent)
        if total_incoming > mp.ships * 2:
            nearest_friendly = None
            nearest_dist = float("inf")
            for other in my_planets:
                if other.id != mp.id:
                    d = dist(mp.x, mp.y, other.x, other.y)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_friendly = other
            if nearest_friendly:
                evac_angle = angle_to_target(mp.x, mp.y, nearest_friendly.x, nearest_friendly.y)
                evac_ships = available[mp.id]
                if evac_ships > 0:
                    moves.append([mp.id, evac_angle, evac_ships])
                    available[mp.id] = 0

    # 2. Handle reinforcements for threatened planets
    for mp in my_planets:
        mp_threats = threats.get(mp.id, [])
        for t in mp_threats:
            if t["urgency"] in ("imminent", "medium"):
                needed = t["ships"] - mp.ships
                if needed > 0:
                    for other in my_planets:
                        if other.id == mp.id or available[other.id] <= 0:
                            continue
                        d = dist(other.x, other.y, mp.x, mp.y)
                        eta = travel_time(d, min(available[other.id], needed))
                        if eta <= t["eta"] + 1:
                            reserve = max(other.production, int(available[other.id] * garrison_factor))
                            can_send = available[other.id] - reserve
                            send = min(can_send, needed)
                            if send > 0:
                                reinforce_angle = angle_to_target(other.x, other.y, mp.x, mp.y)
                                moves.append([other.id, reinforce_angle, send])
                                available[other.id] -= send
                                needed -= send

    # 3. Attack targets
    for entry in scored_targets:
        target = entry["planet"]
        if target.id in assigned_targets:
            continue

        ships_needed = int(entry["garrison_cost"]) + max(3, int(entry["garrison_cost"] * 0.2))

        best_source = None
        best_d = float("inf")
        for mp in my_planets:
            if available.get(mp.id, 0) <= 0:
                continue
            d = dist(mp.x, mp.y, target.x, target.y)
            if d < best_d:
                best_d = d
                best_source = mp

        if best_source is None:
            continue

        reserve = max(best_source.production, int(available[best_source.id] * garrison_factor))
        can_send = available[best_source.id] - reserve
        if can_send < ships_needed:
            if can_send >= target.ships + 2 and phase == "early":
                ships_needed = can_send
            else:
                continue

        init_p = initial_planets.get(target.id)
        if init_p is not None:
            t_init_x, t_init_y = init_p[2], init_p[3]
        else:
            t_init_x, t_init_y = target.x, target.y

        tt = travel_time(best_d, ships_needed)
        pred_x, pred_y = predict_planet_position(t_init_x, t_init_y, target.radius, angular_velocity, turn + int(tt) + 1)
        attack_angle = angle_to_target(best_source.x, best_source.y, pred_x, pred_y)

        launch_x = best_source.x + math.cos(attack_angle) * (best_source.radius + 0.2)
        launch_y = best_source.y + math.sin(attack_angle) * (best_source.radius + 0.2)
        end_x = launch_x + math.cos(attack_angle) * best_d
        end_y = launch_y + math.sin(attack_angle) * best_d

        if crosses_sun(launch_x, launch_y, end_x, end_y):
            offset = 0.3
            for sign in [1, -1]:
                alt_angle = attack_angle + sign * offset
                alt_end_x = launch_x + math.cos(alt_angle) * best_d
                alt_end_y = launch_y + math.sin(alt_angle) * best_d
                if not crosses_sun(launch_x, launch_y, alt_end_x, alt_end_y):
                    attack_angle = alt_angle
                    break
                offset += 0.1
                alt_angle = attack_angle + sign * offset
                alt_end_x = launch_x + math.cos(alt_angle) * best_d
                alt_end_y = launch_y + math.sin(alt_angle) * best_d
                if not crosses_sun(launch_x, launch_y, alt_end_x, alt_end_y):
                    attack_angle = alt_angle
                    break
            else:
                continue

        moves.append([best_source.id, attack_angle, ships_needed])
        available[best_source.id] -= ships_needed
        assigned_targets.add(target.id)

    return moves
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add main.py test_agent.py
git commit -m "feat: add fleet coordinator with sun avoidance, evacuation, and garrison reserves"
```

---

### Task 6: Main Agent Function & Integration

**Files:**
- Modify: `main.py`
- Modify: `test_agent.py`

Wire all components together into the `agent(obs)` entry point. Add comet handling. Test the complete agent against the kaggle environment.

- [ ] **Step 1: Write failing tests**

```python
# append to test_agent.py
from main import agent

def test_agent_returns_list():
    planets = [
        [0, 0, 80, 80, 2.0, 30, 3],
        [1, -1, 75, 75, 1.5, 5, 2],
        [2, 1, 20, 20, 2.0, 15, 3],
    ]
    obs = _make_obs(player=0, planets=planets, step=10)
    result = agent(obs)
    assert isinstance(result, list)

def test_agent_moves_are_valid_format():
    planets = [
        [0, 0, 80, 80, 2.0, 30, 3],
        [1, -1, 75, 75, 1.5, 5, 2],
    ]
    obs = _make_obs(player=0, planets=planets, step=10)
    result = agent(obs)
    for move in result:
        assert len(move) == 3
        assert isinstance(move[0], int)     # planet id
        assert isinstance(move[1], float)   # angle
        assert isinstance(move[2], int)     # ships

def test_agent_does_not_oversend():
    planets = [
        [0, 0, 80, 80, 2.0, 10, 3],
        [1, -1, 75, 75, 1.5, 5, 2],
        [2, -1, 70, 70, 1.5, 5, 2],
    ]
    obs = _make_obs(player=0, planets=planets, step=10)
    result = agent(obs)
    total_sent = sum(m[2] for m in result if m[0] == 0)
    assert total_sent <= 10

def test_agent_empty_when_no_planets():
    planets = [
        [0, 1, 80, 80, 2.0, 30, 3],
    ]
    obs = _make_obs(player=0, planets=planets, step=10)
    result = agent(obs)
    assert result == []

def test_agent_handles_no_targets():
    planets = [
        [0, 0, 80, 80, 2.0, 30, 3],
    ]
    obs = _make_obs(player=0, planets=planets, step=10)
    result = agent(obs)
    assert isinstance(result, list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v -k "test_agent"`
Expected: FAIL — `agent` not defined

- [ ] **Step 3: Implement agent function with comet handling**

Add to `main.py`:

```python
def should_capture_comet(comet_planet, my_planets, comets, comet_ids, turn):
    if comet_planet.id not in comet_ids:
        return False
    for group in comets:
        if comet_planet.id in group.get("planet_ids", []):
            idx = group.get("path_index", 0)
            pid_idx = group["planet_ids"].index(comet_planet.id)
            path = group["paths"][pid_idx]
            remaining_steps = len(path) - idx - 1
            if remaining_steps < 5:
                return False
            break
    if not my_planets:
        return False
    nearest_dist = min(dist(mp.x, mp.y, comet_planet.x, comet_planet.y) for mp in my_planets)
    return nearest_dist < 25 and comet_planet.ships < 30


def agent(obs):
    try:
        gs = parse_game_state(obs)
    except Exception:
        return []

    if not gs["my_planets"]:
        return []

    threats = analyze_threats(
        gs["my_planets"], gs["enemy_fleets"],
        gs["angular_velocity"], gs["initial_planets"], gs["turn"]
    )

    all_targets = gs["neutral_planets"] + gs["enemy_planets"]

    comet_targets = []
    regular_targets = []
    for t in all_targets:
        if t.id in gs["comet_ids"]:
            if should_capture_comet(t, gs["my_planets"], gs["comets"], gs["comet_ids"], gs["turn"]):
                comet_targets.append(t)
        else:
            regular_targets.append(t)

    scored = evaluate_targets(
        gs["my_planets"], regular_targets + comet_targets, gs["enemy_fleets"],
        gs["phase"], gs["angular_velocity"], gs["initial_planets"],
        gs["turn"], gs["num_players"], gs["enemy_strength"]
    )

    moves = coordinate_fleets(
        gs["my_planets"], scored, threats, gs["phase"],
        gs["angular_velocity"], gs["initial_planets"],
        gs["turn"], gs["num_players"], gs["my_total"]
    )

    validated = []
    available = {p.id: p.ships for p in gs["my_planets"]}
    for move in moves:
        pid, angle, ships = move
        if pid in available and ships > 0 and ships <= available[pid]:
            validated.append([int(pid), float(angle), int(ships)])
            available[pid] -= ships

    return validated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add main.py test_agent.py
git commit -m "feat: add main agent function with comet handling and move validation"
```

---

### Task 7: Integration Test Against Kaggle Environment

**Files:**
- Modify: `test_agent.py`

Run the agent against the actual kaggle environment to verify it doesn't crash and produces reasonable results over a full game.

- [ ] **Step 1: Write integration test**

```python
# append to test_agent.py
def test_agent_full_game_vs_random():
    """Run a full game against the random agent to verify no crashes."""
    try:
        from kaggle_environments import make
    except ImportError:
        pytest.skip("kaggle_environments not installed")

    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
    env.run(["main.py", "random"])
    final = env.steps[-1]

    # Agent should not error out
    assert final[0].status in ("DONE", "ACTIVE"), f"Agent errored: {final[0].status}"
    # Agent should have a score
    assert isinstance(final[0].reward, (int, float))

def test_agent_full_game_vs_starter():
    """Run a full game against the starter agent."""
    try:
        from kaggle_environments import make
    except ImportError:
        pytest.skip("kaggle_environments not installed")

    env = make("orbit_wars", configuration={"seed": 123}, debug=True)
    env.run(["main.py", "starter"])
    final = env.steps[-1]

    assert final[0].status in ("DONE", "ACTIVE"), f"Agent errored: {final[0].status}"

def test_agent_full_game_4p():
    """Run a 4-player game."""
    try:
        from kaggle_environments import make
    except ImportError:
        pytest.skip("kaggle_environments not installed")

    env = make("orbit_wars", configuration={"seed": 77}, debug=True)
    env.run(["main.py", "random", "random", "random"])
    final = env.steps[-1]

    assert final[0].status in ("DONE", "ACTIVE"), f"Agent errored: {final[0].status}"
```

- [ ] **Step 2: Run integration tests**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v -k "full_game" --timeout=120`
Expected: All 3 tests PASS. Agent completes full games without crashing.

- [ ] **Step 3: Run quick performance benchmark**

```bash
cd /home/jellyfish/Documents/GitHub/Orbit_wars && python3 -c "
from kaggle_environments import make
import time

wins = 0
for seed in range(10):
    env = make('orbit_wars', configuration={'seed': seed})
    env.run(['main.py', 'random'])
    final = env.steps[-1]
    if final[0].reward == 1:
        wins += 1
    print(f'Seed {seed}: reward={final[0].reward}')
print(f'Win rate vs random: {wins}/10')
"
```

Expected: Win rate >= 7/10 against random agent.

- [ ] **Step 4: Commit**

```bash
git add test_agent.py
git commit -m "test: add integration tests — full games vs random, starter, and 4-player"
```

---

### Task 8: Tuning & Edge Case Hardening

**Files:**
- Modify: `main.py`
- Modify: `test_agent.py`

Fix any issues found during integration testing. Tune garrison reserves, scoring weights, and phase thresholds based on game results.

- [ ] **Step 1: Write edge case tests**

```python
# append to test_agent.py
def test_agent_survives_all_planets_lost():
    """Agent should not crash if it only has fleets, no planets."""
    planets = [
        [0, 1, 80, 80, 2.0, 30, 3],
    ]
    fleets = [
        [0, 0, 50, 50, 0.0, 0, 10],
    ]
    obs = _make_obs(player=0, planets=planets, fleets=fleets, step=200)
    result = agent(obs)
    assert isinstance(result, list)
    assert result == []

def test_agent_handles_orbiting_target():
    """Agent should handle orbiting planets without crashing."""
    planets = [
        [0, 0, 80, 80, 2.0, 50, 3],
        [1, -1, 60, 50, 2.6, 5, 5],  # near center, will orbit
    ]
    initial = [
        [0, 0, 80, 80, 2.0, 50, 3],
        [1, -1, 60, 50, 2.6, 5, 5],
    ]
    obs = _make_obs(player=0, planets=planets, initial_planets=initial, step=50, angular_velocity=0.04)
    result = agent(obs)
    assert isinstance(result, list)

def test_agent_late_game_protects_lead():
    """In late game with a lead, agent should be more defensive."""
    planets = [
        [0, 0, 80, 80, 2.0, 100, 5],
        [1, 0, 75, 75, 2.0, 80, 4],
        [2, 1, 20, 20, 2.0, 30, 2],
    ]
    obs = _make_obs(player=0, planets=planets, step=450)
    result = agent(obs)
    # Should not send everything — need to keep reserves
    total_sent = sum(m[2] for m in result)
    total_available = 180
    assert total_sent < total_available * 0.7
```

- [ ] **Step 2: Run tests to verify they pass (or fix failures)**

Run: `cd /home/jellyfish/Documents/GitHub/Orbit_wars && python -m pytest test_agent.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run benchmark against starter agent**

```bash
cd /home/jellyfish/Documents/GitHub/Orbit_wars && python3 -c "
from kaggle_environments import make

wins = 0
for seed in range(10):
    env = make('orbit_wars', configuration={'seed': seed})
    env.run(['main.py', 'starter'])
    final = env.steps[-1]
    if final[0].reward == 1:
        wins += 1
    print(f'Seed {seed}: reward={final[0].reward}')
print(f'Win rate vs starter: {wins}/10')
"
```

Expected: Win rate >= 6/10 against starter agent.

- [ ] **Step 4: Apply any tuning adjustments discovered during benchmarks**

If win rates are low, adjust these parameters in `main.py`:
- `garrison_factor` values in `coordinate_fleets`
- `prod_weight` exponent in `evaluate_targets`
- Phase boundary thresholds (80, 300)
- Overkill buffer in `coordinate_fleets`

- [ ] **Step 5: Final commit**

```bash
git add main.py test_agent.py
git commit -m "feat: harden edge cases, tune scoring parameters based on benchmark results"
```
