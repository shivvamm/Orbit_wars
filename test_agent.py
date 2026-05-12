# test_agent.py
import math
import pytest

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
    assert crosses_sun(20, 50, 80, 50) is True


def test_crosses_sun_safe_path():
    assert crosses_sun(10, 10, 90, 10) is False


def test_crosses_sun_tangent():
    assert crosses_sun(10, 39, 90, 39) is False


def test_predict_planet_position_static():
    ix, iy = 80.0, 80.0
    x, y = predict_planet_position(ix, iy, 2.0, 0.03, 100)
    assert x == pytest.approx(80.0)
    assert y == pytest.approx(80.0)


def test_predict_planet_position_orbiting():
    ix, iy = 65.0, 50.0
    radius = 1 + math.log(5)
    x, y = predict_planet_position(ix, iy, radius, 0.04, 10)
    orbital_r = math.hypot(ix - 50, iy - 50)
    init_angle = math.atan2(iy - 50, ix - 50)
    new_angle = init_angle + 0.04 * 10
    assert x == pytest.approx(50 + orbital_r * math.cos(new_angle))
    assert y == pytest.approx(50 + orbital_r * math.sin(new_angle))


def test_angle_to_target():
    assert angle_to_target(0, 0, 1, 0) == pytest.approx(0.0)
    assert angle_to_target(0, 0, 0, 1) == pytest.approx(math.pi / 2)
