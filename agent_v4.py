import math
from collections import namedtuple

CENTER = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet = namedtuple("Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])


def dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(s, MAX_SPEED)


def travel_turns(distance, ships):
    if distance <= 0 or ships <= 0:
        return 0.0
    return distance / fleet_speed(ships)


def point_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return dist(px, py, ax, ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return dist(px, py, ax + t * dx, ay + t * dy)


def crosses_sun(x1, y1, x2, y2):
    return point_seg_dist(CENTER, CENTER, x1, y1, x2, y2) < SUN_RADIUS


def predict_pos(init_x, init_y, p_radius, ang_vel, turn):
    dx, dy = init_x - CENTER, init_y - CENTER
    orb_r = math.hypot(dx, dy)
    if orb_r + p_radius >= ROTATION_RADIUS_LIMIT:
        return init_x, init_y
    a = math.atan2(dy, dx) + ang_vel * turn
    return CENTER + orb_r * math.cos(a), CENTER + orb_r * math.sin(a)


def is_static(px, py, p_radius):
    return math.hypot(px - CENTER, py - CENTER) + p_radius >= ROTATION_RADIUS_LIMIT


def safe_angle(sx, sy, sr, tx, ty):
    angle = math.atan2(ty - sy, tx - sx)
    d = dist(sx, sy, tx, ty)
    lx = sx + math.cos(angle) * (sr + 0.2)
    ly = sy + math.sin(angle) * (sr + 0.2)
    ex = lx + math.cos(angle) * (d + 5)
    ey = ly + math.sin(angle) * (d + 5)
    if not crosses_sun(lx, ly, ex, ey):
        return angle
    for offset in [0.25, 0.45, 0.65, 0.85, 1.05, 1.3]:
        for sign in [1, -1]:
            alt = angle + sign * offset
            ax = lx + math.cos(alt) * (d + 5)
            ay = ly + math.sin(alt) * (d + 5)
            if not crosses_sun(lx, ly, ax, ay):
                return alt
    return None


def fleet_target(fx, fy, angle, ships, planets, max_t=60):
    spd = fleet_speed(ships)
    ca, sa = math.cos(angle), math.sin(angle)
    best_proj = float('inf')
    best_pid = None
    for p in planets:
        dx, dy = p.x - fx, p.y - fy
        proj = dx * ca + dy * sa
        if proj <= 0 or proj >= best_proj:
            continue
        perp = abs(dx * sa - dy * ca)
        if perp < p.radius + 1.0:
            eta = max(1, int(proj / spd))
            if eta <= max_t:
                best_proj = proj
                best_pid = p.id
                best_eta = eta
    if best_pid is not None:
        return best_pid, best_eta
    return None, None


def fleets_en_route(my_fleets, planets):
    targeted = {}
    for f in my_fleets:
        pid, _ = fleet_target(f.x, f.y, f.angle, f.ships, planets)
        if pid is not None:
            targeted[pid] = targeted.get(pid, 0) + f.ships
    return targeted


def incoming_threats(enemy_fleets, my_planets):
    threats = {}
    for ef in enemy_fleets:
        pid, eta = fleet_target(ef.x, ef.y, ef.angle, ef.ships, my_planets, max_t=30)
        if pid is not None:
            if pid not in threats:
                threats[pid] = []
            threats[pid].append((ef.ships, eta))
    return threats


def agent(obs):
    try:
        g = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
        player = g("player", 0)
        raw_planets = g("planets", [])
        raw_fleets = g("fleets", [])
        ang_vel = g("angular_velocity", 0.03)
        raw_initial = g("initial_planets", raw_planets)
        raw_comets = g("comets", [])
        comet_ids = set(g("comet_planet_ids", []) or [])
        step = g("step", 0) or 0

        planets = [Planet(*p) for p in raw_planets]
        fleets = [Fleet(*f) for f in raw_fleets]
        init_map = {p[0]: p for p in raw_initial}

        my_planets = [p for p in planets if p.owner == player]
        if not my_planets:
            return []

        my_fleets = [f for f in fleets if f.owner == player]
        enemy_fleets = [f for f in fleets if f.owner != player]
        targets = [p for p in planets if p.owner != player]

        en_route = fleets_en_route(my_fleets, planets)
        threats = incoming_threats(enemy_fleets, my_planets)

        my_total = sum(p.ships for p in my_planets) + sum(f.ships for f in my_fleets)
        my_prod = sum(p.production for p in my_planets)
        enemy_total = {}
        enemy_prod = {}
        for p in planets:
            if p.owner >= 0 and p.owner != player:
                enemy_total[p.owner] = enemy_total.get(p.owner, 0) + p.ships
                enemy_prod[p.owner] = enemy_prod.get(p.owner, 0) + p.production
        for f in fleets:
            if f.owner != player:
                enemy_total[f.owner] = enemy_total.get(f.owner, 0) + f.ships
        num_players = max(len(set(p.owner for p in planets if p.owner >= 0) | set(f.owner for f in fleets)), 2)
        max_enemy_total = max(enemy_total.values()) if enemy_total else 0
        advantage = my_total / max(max_enemy_total, 1)

        moves = []
        available = {p.id: p.ships for p in my_planets}

        # === EVACUATE doomed planets ===
        for mp in my_planets:
            th = threats.get(mp.id, [])
            imminent = sum(s for s, eta in th if eta <= 3)
            if imminent > mp.ships * 1.5 and len(my_planets) > 1:
                best = min((o for o in my_planets if o.id != mp.id),
                           key=lambda o: dist(mp.x, mp.y, o.x, o.y), default=None)
                if best and available[mp.id] > 0:
                    a = safe_angle(mp.x, mp.y, mp.radius, best.x, best.y)
                    if a is not None:
                        moves.append([mp.id, a, available[mp.id]])
                        available[mp.id] = 0

        # === REINFORCE threatened planets ===
        for mp in my_planets:
            th = threats.get(mp.id, [])
            for inc_ships, eta in sorted(th, key=lambda x: x[1]):
                if eta > 8:
                    continue
                deficit = inc_ships - mp.ships
                if deficit <= 0:
                    continue
                for other in sorted(my_planets, key=lambda o: dist(o.x, o.y, mp.x, mp.y)):
                    if other.id == mp.id:
                        continue
                    d = dist(other.x, other.y, mp.x, mp.y)
                    tt = travel_turns(d, available.get(other.id, 0))
                    if tt > eta + 1 or available.get(other.id, 0) <= 0:
                        continue
                    reserve = max(other.production, 3)
                    can_send = available[other.id] - reserve
                    send = min(can_send, deficit)
                    if send >= 3:
                        a = safe_angle(other.x, other.y, other.radius, mp.x, mp.y)
                        if a is not None:
                            moves.append([other.id, a, send])
                            available[other.id] -= send
                            deficit -= send
                    if deficit <= 0:
                        break

        # === ATTACK: coordinated target-centric allocation ===
        skip_comets = set()
        for t in targets:
            if t.id in comet_ids:
                for grp in raw_comets:
                    pids = grp.get("planet_ids", []) if isinstance(grp, dict) else getattr(grp, "planet_ids", [])
                    if t.id in pids:
                        idx = grp.get("path_index", 0) if isinstance(grp, dict) else getattr(grp, "path_index", 0)
                        paths = grp.get("paths", []) if isinstance(grp, dict) else getattr(grp, "paths", [])
                        pi = pids.index(t.id)
                        if pi < len(paths) and len(paths[pi]) - idx - 1 < 10:
                            skip_comets.add(t.id)
                        break

        target_info = {}
        for t in targets:
            if t.id in skip_comets:
                continue
            init_p = init_map.get(t.id)
            static = is_static(init_p[2], init_p[3], t.radius) if init_p else True
            target_info[t.id] = (t, static, init_p)

        max_range = 55 if advantage < 1.5 else 70
        all_options = []
        for mp in my_planets:
            if available.get(mp.id, 0) < 5:
                continue

            mp_init = init_map.get(mp.id)
            mp_static = is_static(mp_init[2], mp_init[3], mp.radius) if mp_init else True

            for tid, (t, static, init_p) in target_info.items():
                d = dist(mp.x, mp.y, t.x, t.y)
                if d > max_range:
                    continue

                tgt_x, tgt_y = t.x, t.y
                if not static and init_p and step > 0:
                    tt_est = travel_turns(d, max(t.ships + 3, 10))
                    for _ in range(3):
                        px, py = predict_pos(init_p[2], init_p[3], t.radius, ang_vel, step + int(tt_est) + 1)
                        src_x, src_y = mp.x, mp.y
                        if not mp_static and mp_init:
                            src_x, src_y = predict_pos(mp_init[2], mp_init[3], mp.radius, ang_vel, step + 1)
                        d2 = dist(src_x, src_y, px, py)
                        tt_est = travel_turns(d2, max(t.ships + 3, 10))
                    tgt_x, tgt_y = px, py
                    d = dist(mp.x, mp.y, tgt_x, tgt_y)

                garrison = t.ships
                tt = travel_turns(d, max(garrison + 3, 10))
                if t.owner >= 0:
                    garrison += int(t.production * (tt + 1))

                already = en_route.get(t.id, 0)
                net_garrison = max(0, garrison - already)
                ships_needed = net_garrison + max(2, int(net_garrison * 0.15))
                if ships_needed <= 0:
                    continue

                prod_val = t.production ** 1.8
                if t.owner >= 0:
                    prod_val *= 1.8
                roi = prod_val / (ships_needed + d / 6.0 + 1.0)
                if not static:
                    roi *= 0.65
                if t.owner == -1:
                    roi *= 1.5
                    if step < 60:
                        roi *= 1.5
                if advantage > 2.5 and t.owner >= 0:
                    roi *= 1.5
                if num_players > 2 and t.owner >= 0:
                    owner_str = enemy_total.get(t.owner, 0)
                    max_str = max(enemy_total.values()) if enemy_total else 1
                    if max_str > 0 and owner_str >= max_str * 0.8:
                        roi *= 0.5

                all_options.append((roi, mp, t, d, ships_needed, tgt_x, tgt_y))

        all_options.sort(key=lambda x: x[0], reverse=True)

        committed = {}
        if step < 60:
            max_targets = max(4, len(my_planets))
        else:
            max_targets = max(4, len(my_planets) * 2 // 3)
        distinct_targets = set()

        for roi, mp, target, d, needed, tgt_x, tgt_y in all_options:
            already_committed = committed.get(target.id, 0)
            still_needed = needed - already_committed
            if still_needed <= 0:
                continue

            if target.id not in distinct_targets and len(distinct_targets) >= max_targets:
                continue

            avail = available.get(mp.id, 0)
            th = threats.get(mp.id, [])
            under_threat = any(eta <= 5 for _, eta in th)

            if step < 80:
                reserve = max(mp.production, 2)
            elif advantage > 2.0:
                reserve = max(mp.production, 2)
            elif step < 300:
                reserve = max(mp.production * 2, int(avail * 0.15), 4)
            else:
                reserve = max(mp.production * 3, int(avail * 0.3), 8)

            if under_threat:
                reserve = max(reserve, int(avail * 0.5))

            can_send = avail - reserve
            if step < 60 and target.owner == -1:
                min_fleet = 3
            else:
                min_fleet = max(5, int(d * 0.15))
            if can_send < min_fleet:
                continue

            send = min(can_send, max(still_needed, min_fleet))
            if already_committed == 0 and send < still_needed * 0.4 and still_needed > 10:
                continue

            angle = safe_angle(mp.x, mp.y, mp.radius, tgt_x, tgt_y)
            if angle is None:
                continue

            moves.append([mp.id, angle, send])
            available[mp.id] -= send
            committed[target.id] = already_committed + send
            distinct_targets.add(target.id)

        # === MOP-UP: idle planets attack nearest cheap target ===
        for mp in my_planets:
            avail = available.get(mp.id, 0)
            if avail < 8:
                continue
            reserve = max(mp.production * 2, 5)
            can_send = avail - reserve
            if can_send < 5:
                continue
            best_target = None
            best_cost = float('inf')
            for tid, (t, static, init_p) in target_info.items():
                d = dist(mp.x, mp.y, t.x, t.y)
                if d > 40:
                    continue
                garrison = t.ships
                if t.owner >= 0:
                    tt = travel_turns(d, can_send)
                    garrison += int(t.production * (tt + 1))
                already_there = en_route.get(t.id, 0) + committed.get(t.id, 0)
                needed = max(0, garrison - already_there) + 2
                if needed <= 0 or needed > can_send:
                    continue
                cost = needed + d * 0.5
                if cost < best_cost:
                    best_cost = cost
                    best_target = (t, d, needed)
            if best_target:
                t, d, needed = best_target
                a = safe_angle(mp.x, mp.y, mp.radius, t.x, t.y)
                if a is not None:
                    send = min(can_send, max(needed, 5))
                    moves.append([mp.id, a, send])
                    available[mp.id] -= send
                    committed[t.id] = committed.get(t.id, 0) + send

        # === REDISTRIBUTE: idle planets forward ships to front-line ===
        if step > 40 and len(my_planets) > 3:
            enemy_cx = 0.0
            enemy_cy = 0.0
            enemy_count = 0
            for p in planets:
                if p.owner >= 0 and p.owner != player:
                    enemy_cx += p.x
                    enemy_cy += p.y
                    enemy_count += 1
            if enemy_count > 0:
                enemy_cx /= enemy_count
                enemy_cy /= enemy_count

                for mp in my_planets:
                    avail = available.get(mp.id, 0)
                    if avail < 15:
                        continue
                    my_d_to_enemy = dist(mp.x, mp.y, enemy_cx, enemy_cy)
                    closer = [o for o in my_planets if o.id != mp.id
                              and dist(o.x, o.y, enemy_cx, enemy_cy) < my_d_to_enemy - 10]
                    if not closer:
                        continue
                    best_fwd = min(closer, key=lambda o: dist(o.x, o.y, enemy_cx, enemy_cy))
                    reserve = max(mp.production * 3, 10)
                    send = avail - reserve
                    if send < 10:
                        continue
                    a = safe_angle(mp.x, mp.y, mp.radius, best_fwd.x, best_fwd.y)
                    if a is not None:
                        moves.append([mp.id, a, send])
                        available[mp.id] -= send

        # Validate moves
        result = []
        used = {}
        for pid, angle, ships in moves:
            ships = int(ships)
            mp = next((p for p in my_planets if p.id == pid), None)
            if mp and ships > 0:
                u = used.get(pid, 0)
                if u + ships <= mp.ships:
                    result.append([int(pid), float(angle), ships])
                    used[pid] = u + ships

        return result
    except Exception:
        return []
