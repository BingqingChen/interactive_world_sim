"""Gripper<->T-block contact detection for the MuJoCo/ALOHA pusht scene."""


def build_contact_sets(env):
    """Geom-id sets for gripper<->T contact counting: the T block is body 'box'
    (geom 'pusht' + its convex-hull geoms); the touching surfaces of each arm are
    its two *finger_link bodies."""
    m = env._env.physics.model
    t_geoms, left, right = set(), set(), set()
    for g in range(m.ngeom):
        body = m.id2name(m.geom_bodyid[g], "body") or ""
        if body == "box":
            t_geoms.add(g)
        elif "finger_link" in body:
            (left if body.startswith("left/") else right).add(g)
    assert t_geoms and left and right, "contact geom sets not found"
    return t_geoms, left, right


def gripper_t_contact(env, contact_sets):
    """(left_touching, right_touching) at the current physics state."""
    t_geoms, left, right = contact_sets
    d = env._env.physics.data
    in_l = in_r = False
    for i in range(d.ncon):
        g1, g2 = int(d.contact.geom1[i]), int(d.contact.geom2[i])
        if g1 in t_geoms:
            g1, g2 = g2, g1
        if g2 not in t_geoms:
            continue
        if g1 in left:
            in_l = True
        elif g1 in right:
            in_r = True
    return in_l, in_r
