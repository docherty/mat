from connectors.loader import load_connectors_dir
from connectors.local_affinity import resolve_local_pins, routing_pool
from connectors.paths import EXAMPLES_DIR


def test_routing_pool_pins_one_local_per_endpoint():
    pool = load_connectors_dir(EXAMPLES_DIR)
    locals_ = [c for c in pool if c.locality == "local"]
    if len(locals_) < 2:
        return
    routed = routing_pool(pool, primary_id=locals_[0].id)
    routed_locals = [c for c in routed if c.locality == "local"]
    assert len(routed_locals) == 1
    assert routed_locals[0].id == locals_[0].id


def test_resolve_local_pins_prefers_primary():
    pool = load_connectors_dir(EXAMPLES_DIR)
    locals_ = [c for c in pool if c.locality == "local"]
    if not locals_:
        return
    primary = locals_[0]
    pins = resolve_local_pins(pool, primary_id=primary.id)
    for pin in pins.values():
        assert pin.id == primary.id


def test_role_coordinator_pins_across_roles():
    from eval.live_loop import RoleCoordinator
    from eval.oracle import Task

    pool = load_connectors_dir(EXAMPLES_DIR)
    locals_ = [c for c in pool if c.locality == "local"]
    if len(locals_) < 2:
        return
    routed = routing_pool(pool, primary_id=locals_[0].id)
    coord = RoleCoordinator()
    coord.bind_pool(routed)
    task = Task(
        id="t",
        prompt="def f(): pass",
        tests="",
        difficulty=0.3,
        required_tags={"coding": 1.0},
    )
    picks = {coord.pick(task, routed, role=r).id for r in ("thinker", "worker", "verifier")}
    assert len(picks) == 1
