# Orbit Wars Agent Design Spec

## Overview

A heuristic-based adaptive AI agent for the Orbit Wars Kaggle competition. Single-file `main.py` submission. No external dependencies beyond `math` and kaggle environment builtins. Must run within 1s per turn over 500-turn games. Handles both 1v1 and 4-player FFA.

## Architecture

Six logical components inside one `agent(obs)` function:

1. **GameState** — parse observation into Planet/Fleet named tuples, track turn number, player count, detect game phase
2. **OrbitalPredictor** — compute future positions of orbiting planets using `initial_planets` + `angular_velocity`
3. **ThreatAnalyzer** — scan enemy fleets, predict planet impacts, calculate reinforcement needs
4. **TargetEvaluator** — score every non-owned planet for attack priority
5. **FleetCoordinator** — globally assign which owned planets send how many ships where, prevent double-attacks
6. **PhaseManager** — determine early/mid/late phase, adjust aggression and garrison reserves

Execution order per turn: parse state -> detect threats -> evaluate targets -> coordinate fleets -> return moves.

## Target Scoring

```
score = (production * production_weight) / (travel_turns + garrison_to_overcome + 1)
```

Modifiers:
- **Proximity bonus**: planets near owned territory score higher
- **Enemy proximity penalty**: planets deep in enemy territory score lower
- **Cluster bonus**: planets near other valuable unowned planets score higher
- **Neutral vs enemy**: neutrals favored early; enemies favored mid/late
- **Production priority**: prod-5 planets heavily prioritized over prod-1

Garrison cost estimation includes:
- Current garrison ships
- Production during fleet transit (enemy planets keep producing)
- Known enemy fleets heading to the same target

4-player adjustment: penalize targets another enemy is already attacking.

## Threat Detection & Defense

**Fleet tracking**: for each enemy fleet, compute trajectory intersection with owned planets (accounting for planet radius + orbital movement).

**Response tiers**:
- Imminent (1-3 turns): reinforce or evacuate
- Medium (4-8 turns): increase garrison reserves, potentially redirect attacks
- Distant (8+ turns): monitor only

**Garrison policy**: `min_garrison = max(production, nearby_enemy_strength * 0.3)`. High-production planets keep larger reserves.

**Evacuation**: if a planet is certain to fall, launch all ships toward nearest friendly planet or valuable target.

**Sun avoidance**: check if fleet path crosses within sunRadius (10) of center (50,50). Skip or compute offset angle.

## Fleet Coordination

- Global assignment each turn: avoid multiple planets sending to the same target unintentionally
- **Pincer attacks**: when target is between two owned planets, coordinate simultaneous arrival
- **Overkill prevention**: send garrison + production_during_travel + small buffer, not everything
- **Batch vs split**: prefer one large fleet for distant targets (speed scaling benefit)

## Orbital Prediction

For orbiting planets, predict position at `current_turn + travel_time`:
- `angle = initial_angle + angular_velocity * target_turn`
- `x = 50 + orbital_radius * cos(angle)`, `y = 50 + orbital_radius * sin(angle)`
- Iterative convergence (2-3 iterations): estimate travel -> predict position -> re-estimate travel

## Phase Management

- **Early (turns 0-80)**: maximize expansion to cheap nearby neutrals. Low garrison reserves. Prioritize high-production planets.
- **Mid (turns 80-300)**: consolidate, attack enemy weak points, defend high-value planets, consider fleet economics.
- **Late (turns 300-500)**: protect lead (turtle) or go all-in if losing. Ship count at turn 500 determines winner.

## Comet Handling

- Evaluate quadrant comet at spawn turns (50, 150, 250, 350, 450)
- Capture if: nearby, will stay on board long enough, garrison cost is low
- Use `comets.paths` + `path_index` to predict trajectory and remaining board time
- Can launch from owned comets as forward bases

## 4-Player Adaptations

- Track relative strength of each opponent (total ships + planets)
- Avoid attacking strongest player; let others weaken them
- Opportunistically target weakest player's planets
- If two enemies are fighting, expand elsewhere

## Constraints

- 1 second per turn time limit
- Single `main.py` file
- Only `math` module + kaggle environment imports
- Must handle both 2-player and 4-player games
- 500-turn maximum game length
