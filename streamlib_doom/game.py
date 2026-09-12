"""E1M1 as a game, 35 tics a second: movement against the real linedefs,
doors, the lift, the secret floor, the exit switch, pickups, hitscan weapons,
zombiemen and imps with Doom's own states and sounds, damage and death.

Pure Python and numpy on purpose: it is the model, and the graph around it
is the renderer, the status bar and the browser. Numbers that look arbitrary
are Doom's (speeds in map units per tic, tics per state, damage dice).
"""
from __future__ import annotations

import math
import random

import numpy

from .wad import Level, Wad

TICRATE = 35
PLAYER_RADIUS = 16
PLAYER_HEIGHT = 56
STEP_HEIGHT = 24
VIEW_HEIGHT = 41
FRICTION = 0.90625
RUN_THRUST = 1.6
MAX_BOB = 16.0
USE_RANGE = 64
HITSCAN_RANGE = 2048
MELEE_RANGE = 64
DOOR_SPEED = 2.0
DOOR_WAIT = 150
PLAT_SPEED = 4.0
PLAT_WAIT = 105
MESSAGE_TICS = 4 * TICRATE

DOOR_SPECIALS = {1, 26, 27, 28, 31, 32, 33, 34, 117, 118}
DOOR_STAYS_OPEN = {31, 32, 33, 34}

ZOMBIEMAN, IMP = 3004, 3001
MONSTER = {
    ZOMBIEMAN: dict(prefix="POSS", health=20, radius=20, speed=8, pain_chance=200, pain_tics=3, walk="ABCD",
                    attack=("E", 10, "F", 8), death="HIJKL", death_tics=5, sight=("DSPOSIT1", "DSPOSIT2", "DSPOSIT3"),
                    pain="DSPOPAIN", die=("DSPODTH1", "DSPODTH2", "DSPODTH3"), missile="hitscan"),
    IMP: dict(prefix="TROO", health=60, radius=20, speed=8, pain_chance=200, pain_tics=2, walk="ABCD",
              attack=("E", 8, "F", 8, "G", 6), death="IJKLM", death_tics=8, sight=("DSBGSIT1", "DSBGSIT2"),
              pain="DSPOPAIN", die=("DSBGDTH1", "DSBGDTH2"), missile="fireball"),
}
ITEM = {  # type: (sprite prefix, frames, fullbright)
    2011: ("STIM", "A", False), 2012: ("MEDI", "A", False), 2014: ("BON1", "ABCDCB", True), 2015: ("BON2", "ABCDCB", True),
    2018: ("ARM1", "AB", False), 2019: ("ARM2", "AB", False), 2007: ("CLIP", "A", False), 2048: ("AMMO", "A", False),
    2008: ("SHOT", "A", False), 2049: ("SBOX", "A", False), 2046: ("BROK", "A", False), 2010: ("ROCK", "A", False),
    2001: ("SHOT", "A", False),
}
ITEM_SPRITE_OVERRIDE = {2001: "SHOT"}  # the shotgun weapon's sprite prefix is SHOT too, frame A
SOLID_DECORATION = {2028: ("COLU", "A", True), 48: ("ELEC", "A", False), 2035: ("BAR1", "AB", False), 35: ("CBRA", "A", True), 34: ("CAND", "A", True)}
PASSABLE_DECORATION = {24: ("POL5", "A", False), 10: ("PLAY", "W", False), 12: ("PLAY", "W", False), 15: ("PLAY", "N", False), 18: ("POSS", "L", False)}
MULTIPLAYER_ONLY = 16
MEDIUM_SKILL = 2

WEAPON_PISTOL, WEAPON_SHOTGUN = 2, 3


def _dice(count: int, sides: int) -> int:
    return sum(random.randint(1, sides) for _ in range(count))


class Game:
    def __init__(self, wad: Wad, level: Level) -> None:
        self.wad = wad
        self.level = level
        lv = level
        n = len(lv.linedefs)
        self.l1 = numpy.array([lv.vertexes[l[0]] for l in lv.linedefs], dtype=numpy.float64)
        self.l2 = numpy.array([lv.vertexes[l[1]] for l in lv.linedefs], dtype=numpy.float64)
        self.ld = self.l2 - self.l1
        self.ld_len2 = numpy.maximum((self.ld ** 2).sum(axis=1), 1e-9)
        self.flags = numpy.array([int(l[2]) for l in lv.linedefs])
        self.special = numpy.array([int(l[3]) for l in lv.linedefs])
        self.tag = numpy.array([int(l[4]) for l in lv.linedefs])
        self.right_sector = numpy.array([lv.sidedefs[int(l[5])][5] if int(l[5]) >= 0 else -1 for l in lv.linedefs])
        self.left_sector = numpy.array([lv.sidedefs[int(l[6])][5] if int(l[6]) >= 0 else -1 for l in lv.linedefs])
        self.two_sided = (self.flags & 4 != 0) & (self.left_sector >= 0) & (self.right_sector >= 0)
        self.floors = numpy.array([s[0] for s in lv.sectors], dtype=numpy.float64)
        self.ceilings = numpy.array([s[1] for s in lv.sectors], dtype=numpy.float64)
        self.base_floors, self.base_ceilings = self.floors.copy(), self.ceilings.copy()
        self.lights = [float(s[4]) for s in lv.sectors]
        self.neighbours: dict[int, set[int]] = {}
        for i in range(n):
            if self.two_sided[i]:
                a, b = int(self.right_sector[i]), int(self.left_sector[i])
                self.neighbours.setdefault(a, set()).add(b)
                self.neighbours.setdefault(b, set()).add(a)
        self.min_light = [min([lv.sectors[m][4] for m in self.neighbours.get(k, ())] + [s[4]]) for k, s in enumerate(lv.sectors)]
        base = wad.index["E1M1"]
        self.segs = numpy.frombuffer(wad.lump_at(base + 5), dtype="<i2").reshape(-1, 6)
        self.subsectors = numpy.frombuffer(wad.lump_at(base + 6), dtype="<i2").reshape(-1, 2)
        raw_nodes = wad.lump_at(base + 7)
        self.nodes = numpy.frombuffer(raw_nodes, dtype="<i2").reshape(-1, 14)
        self.node_children = numpy.frombuffer(raw_nodes, dtype="<u2").reshape(-1, 14)[:, 12:14]
        self.random = random.Random()
        self.flicker: dict[int, tuple[int, float]] = {}
        self.reset()

    # -- world queries ------------------------------------------------------------
    def sector_at(self, x: float, y: float) -> int:
        n = len(self.nodes) - 1
        while True:
            nx, ny, ndx, ndy = (int(v) for v in self.nodes[n][:4])
            side = 0 if (y - ny) * ndx < ndy * (x - nx) else 1
            child = int(self.node_children[n][side])
            if child & 0x8000:
                seg = self.segs[int(self.subsectors[child & 0x7FFF][1])]
                line = self.level.linedefs[int(seg[3])]
                sidedef = int(line[5]) if int(seg[4]) == 0 else int(line[6])
                return self.level.sidedefs[sidedef][5]
            n = child

    def _crossings(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        """(t along a->b in 0..1, or inf) for every linedef the segment a->b crosses."""
        d = b - a
        w = self.l1 - a
        denom = d[0] * self.ld[:, 1] - d[1] * self.ld[:, 0]
        with numpy.errstate(divide="ignore", invalid="ignore"):
            t = (w[:, 0] * self.ld[:, 1] - w[:, 1] * self.ld[:, 0]) / denom
            u = (w[:, 0] * d[1] - w[:, 1] * d[0]) / denom
        hit = (numpy.abs(denom) > 1e-9) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
        return numpy.where(hit, t, numpy.inf)

    def _opening(self, line: int) -> tuple[float, float]:
        a, b = int(self.right_sector[line]), int(self.left_sector[line])
        return max(self.floors[a], self.floors[b]), min(self.ceilings[a], self.ceilings[b])

    def line_of_sight(self, ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> bool:
        a, b = numpy.array((ax, ay)), numpy.array((bx, by))
        t = self._crossings(a, b)
        for line in numpy.nonzero(numpy.isfinite(t))[0]:
            if not self.two_sided[line]:
                return False
            floor, ceiling = self._opening(int(line))
            z = az + (bz - az) * t[line]
            if z < floor or z > ceiling or ceiling - floor <= 0:
                return False
        return True

    def wall_hit(self, ax: float, ay: float, az: float, angle: float, pitch_z_per_unit: float, range_: float) -> tuple[float, int]:
        """Distance to the first wall a shot along `angle` cannot pass, and the line."""
        a = numpy.array((ax, ay))
        b = a + numpy.array((math.cos(angle), math.sin(angle))) * range_
        t = self._crossings(a, b)
        for line in numpy.argsort(t):
            if not numpy.isfinite(t[line]):
                break
            if not self.two_sided[line]:
                return float(t[line] * range_), int(line)
            floor, ceiling = self._opening(int(line))
            z = az + pitch_z_per_unit * t[line] * range_
            if z < floor or z > ceiling:
                return float(t[line] * range_), int(line)
        return range_, -1

    def move_actor(self, x: float, y: float, nx: float, ny: float, radius: float, sector: int, is_player: bool) -> tuple[float, float]:
        """Slide (nx, ny) out of every wall it would overlap, Doom's step rules applied."""
        q = numpy.array((nx, ny), dtype=numpy.float64)
        floor_here = self.floors[sector]
        for _ in range(4):
            rel = q - self.l1
            t = numpy.clip((rel * self.ld).sum(axis=1) / self.ld_len2, 0.0, 1.0)
            closest = self.l1 + self.ld * t[:, None]
            offset = q - closest
            dist = numpy.sqrt((offset ** 2).sum(axis=1))
            near = numpy.nonzero(dist < radius)[0]
            if len(near) == 0:
                break
            pushed = False
            for line in near[numpy.argsort(dist[near])]:
                line = int(line)
                blocking = not self.two_sided[line] or (self.flags[line] & 1)
                if not blocking:
                    floor, ceiling = self._opening(line)
                    a, b = int(self.right_sector[line]), int(self.left_sector[line])
                    blocking = (ceiling - floor < PLAYER_HEIGHT) or (floor - floor_here > STEP_HEIGHT)
                    if not blocking and not is_player and abs(self.floors[a] - self.floors[b]) > STEP_HEIGHT:
                        blocking = True
                if not blocking:
                    continue
                d = dist[line]
                if d < 1e-6:
                    normal = numpy.array((-self.ld[line][1], self.ld[line][0]))
                    normal /= max(numpy.linalg.norm(normal), 1e-9)
                else:
                    normal = offset[line] / d
                q = q + normal * (radius - d + 0.01)
                pushed = True
                break
            if not pushed:
                break
        return float(q[0]), float(q[1])

    # -- state ----------------------------------------------------------------------
    def reset(self) -> None:
        x, y, angle = self.level.player_start()
        self.tick_count = 0
        self.floors[:] = self.base_floors
        self.ceilings[:] = self.base_ceilings
        self.player = dict(x=float(x), y=float(y), angle=math.radians(angle), momx=0.0, momy=0.0, health=100, armor=0,
                           armor_type=0, bullets=50, shells=0, rockets=0, weapons={WEAPON_PISTOL}, ready=WEAPON_PISTOL,
                           pending=None, weapon_tics=0, weapon_state="ready", weapon_raise=0, flash=0, refire=0,
                           damage_count=0, bonus_count=0, message="", message_tics=0, face="STFST01", face_tics=0,
                           face_priority=0, dead=False, dead_tics=0, attacker=None, fire_held_tics=0, bob=0.0,
                           sector=self.sector_at(x, y), kills=0, nukage_tics=0, complete=0)
        self.player["z"] = self.floors[self.player["sector"]]
        self.monsters, self.items, self.decorations, self.projectiles, self.effects = [], [], [], [], []
        for tx, ty, tangle, kind, flags in ((int(v) for v in t) for t in self.level.things):
            if not (flags & MEDIUM_SKILL) or (flags & MULTIPLAYER_ONLY):
                continue
            sector = self.sector_at(tx, ty)
            if kind in MONSTER:
                spec = MONSTER[kind]
                self.monsters.append(dict(kind=kind, x=float(tx), y=float(ty), angle=math.radians(tangle), sector=sector,
                                          health=spec["health"], state="idle", frame="A", state_tics=0, move_tics=0,
                                          move_angle=0.0, alive=True, ambush=bool(flags & 8), attack_step=0, radius=spec["radius"]))
            elif kind in ITEM:
                self.items.append(dict(kind=kind, x=float(tx), y=float(ty), sector=sector))
            elif kind in SOLID_DECORATION or kind in PASSABLE_DECORATION:
                self.decorations.append(dict(kind=kind, x=float(tx), y=float(ty), sector=sector, health=20 if kind == 2035 else None, exploding=-1))
        self.doors: dict[int, dict] = {}
        self.plats: dict[int, dict] = {}
        self.used_once: set[int] = set()
        self.events: list[str] = []
        self.random = random.Random(self.tick_count)
        self.level_complete = 0

    # -- tic -------------------------------------------------------------------------
    def tick(self, controls: dict) -> None:
        self.events = []
        self.tick_count += 1
        p = self.player
        if self.level_complete:
            self.level_complete -= 1
            if self.level_complete == 0:
                self.reset()
            return
        if p["dead"]:
            p["dead_tics"] += 1
            if controls.get("fire") and p["dead_tics"] > TICRATE:
                self.reset()
            self._tick_monsters()
            self._tick_projectiles()
            self._tick_sectors()
            self._decay_counters()
            return
        self._tick_player(controls)
        self._tick_weapon(controls)
        self._tick_monsters()
        self._tick_projectiles()
        self._tick_sectors()
        self._tick_effects()
        self._decay_counters()

    def _decay_counters(self) -> None:
        p = self.player
        if p["damage_count"] > 0:
            p["damage_count"] -= 1
        if p["bonus_count"] > 0:
            p["bonus_count"] -= 1
        if p["message_tics"] > 0:
            p["message_tics"] -= 1
            if p["message_tics"] == 0:
                p["message"] = ""
        if p["face_tics"] > 0:
            p["face_tics"] -= 1
            if p["face_tics"] == 0:
                p["face_priority"] = 0
        for d in self.decorations:
            if d["exploding"] >= 0:
                d["exploding"] += 1

    def _say(self, message: str) -> None:
        self.player["message"] = message
        self.player["message_tics"] = MESSAGE_TICS

    def _tick_player(self, controls: dict) -> None:
        p = self.player
        p["angle"] = (p["angle"] - math.radians(float(controls.get("turn", 0.0)))) % (2 * math.pi)
        forward = max(-1.0, min(1.0, float(controls.get("forward", 0.0))))
        strafe = max(-1.0, min(1.0, float(controls.get("strafe", 0.0))))
        fx, fy = math.cos(p["angle"]), math.sin(p["angle"])
        rx, ry = fy, -fx
        p["momx"] += (fx * forward + rx * strafe) * RUN_THRUST
        p["momy"] += (fy * forward + ry * strafe) * RUN_THRUST
        p["momx"] *= FRICTION
        p["momy"] *= FRICTION
        if abs(p["momx"]) < 0.02:
            p["momx"] = 0.0
        if abs(p["momy"]) < 0.02:
            p["momy"] = 0.0
        old_x, old_y = p["x"], p["y"]
        nx, ny = self.move_actor(p["x"], p["y"], p["x"] + p["momx"], p["y"] + p["momy"], PLAYER_RADIUS, p["sector"], True)
        p["x"], p["y"] = nx, ny
        p["sector"] = self.sector_at(nx, ny)
        p["z"] = self.floors[p["sector"]]
        speed2 = p["momx"] ** 2 + p["momy"] ** 2
        p["bob"] = min(MAX_BOB, speed2 / 4.0)
        self._cross_lines(old_x, old_y, nx, ny)
        self._pick_up_items()
        if controls.get("use"):
            if not p.get("use_held"):
                self._use_line()
            p["use_held"] = True
        else:
            p["use_held"] = False
        pending = controls.get("weapon")
        if pending in (WEAPON_PISTOL, WEAPON_SHOTGUN) and pending in p["weapons"] and pending != p["ready"] and p["pending"] is None:
            p["pending"] = pending
        # nukage
        special = self.level.sectors[p["sector"]][5]
        if special in (7, 5, 16, 4):
            p["nukage_tics"] += 1
            if p["nukage_tics"] % 32 == 0:
                self._damage_player({7: 5, 5: 10, 16: 20, 4: 20}[special], None)
        # the face looks around
        if p["face_priority"] == 0 and self.tick_count % 17 == 0:
            row = self._face_row()
            p["face"] = f"STFST{row}{self.random.choice((0, 1, 2))}"

    def _face_row(self) -> int:
        health = self.player["health"]
        return 0 if health >= 80 else 1 if health >= 60 else 2 if health >= 40 else 3 if health >= 20 else 4

    def _set_face(self, name: str, priority: int, tics: int) -> None:
        p = self.player
        if priority >= p["face_priority"]:
            p["face"], p["face_priority"], p["face_tics"] = name, priority, tics

    def _cross_lines(self, ox: float, oy: float, nx: float, ny: float) -> None:
        if (ox, oy) == (nx, ny):
            return
        t = self._crossings(numpy.array((ox, oy)), numpy.array((nx, ny)))
        for line in numpy.nonzero(numpy.isfinite(t))[0]:
            line = int(line)
            special = int(self.special[line])
            if special == 88:
                self._start_plat(int(self.tag[line]), "lift")
            elif special == 36 and line not in self.used_once:
                self.used_once.add(line)
                self._start_plat(int(self.tag[line]), "lower_fast")

    def _use_line(self) -> None:
        p = self.player
        a = numpy.array((p["x"], p["y"]))
        b = a + numpy.array((math.cos(p["angle"]), math.sin(p["angle"]))) * USE_RANGE
        t = self._crossings(a, b)
        for line in numpy.argsort(t):
            line = int(line)
            if not numpy.isfinite(t[line]):
                break
            special = int(self.special[line])
            if special in DOOR_SPECIALS:
                self._open_door(int(self.left_sector[line]), special in DOOR_STAYS_OPEN)
                return
            if special == 11:
                self.events.append("DSSWTCHN")
                self._say("Level completed: E1M1 Hangar. Restarting in three seconds.")
                self.level_complete = 3 * TICRATE
                return
            if not self.two_sided[line]:
                self.events.append("DSNOWAY")
                return

    def _open_door(self, sector: int, stays_open: bool) -> None:
        door = self.doors.get(sector)
        if door and door["state"] in ("opening", "open"):
            if not stays_open:
                door["state"], door["wait"] = "closing", 0
                self.events.append("DSDORCLS")
            return
        top = min(self.ceilings[m] for m in self.neighbours.get(sector, ())) - 4
        self.doors[sector] = dict(state="opening", top=top, wait=0, stays=stays_open)
        self.events.append("DSDOROPN")

    def _start_plat(self, tag: int, kind: str) -> None:
        for sector, s in enumerate(self.level.sectors):
            if s[6] != tag or sector in self.plats:
                continue
            neighbours = self.neighbours.get(sector, ())
            if kind == "lift":
                target = min(self.floors[m] for m in neighbours)
                self.plats[sector] = dict(kind="lift", state="down", target=target, home=self.floors[sector], wait=0)
            else:
                target = max(self.floors[m] for m in neighbours) + 8
                self.plats[sector] = dict(kind="lower", state="down", target=target, home=target, wait=0)
            self.events.append("DSPSTART")

    def _tick_sectors(self) -> None:
        for sector, door in list(self.doors.items()):
            if door["state"] == "opening":
                self.ceilings[sector] = min(door["top"], self.ceilings[sector] + DOOR_SPEED)
                if self.ceilings[sector] >= door["top"]:
                    door["state"] = "stay" if door["stays"] else "open"
            elif door["state"] == "open":
                door["wait"] += 1
                if door["wait"] >= DOOR_WAIT:
                    door["state"] = "closing"
                    self.events.append("DSDORCLS")
            elif door["state"] == "closing":
                occupied = self.player["sector"] == sector or any(m["alive"] and m["sector"] == sector for m in self.monsters)
                if occupied:
                    door["state"], door["wait"] = "opening", 0
                    self.events.append("DSDOROPN")
                    continue
                self.ceilings[sector] = max(self.floors[sector], self.ceilings[sector] - DOOR_SPEED)
                if self.ceilings[sector] <= self.floors[sector]:
                    del self.doors[sector]
        for sector, plat in list(self.plats.items()):
            if plat["state"] == "down":
                self.floors[sector] = max(plat["target"], self.floors[sector] - PLAT_SPEED)
                if self.floors[sector] <= plat["target"]:
                    self.events.append("DSPSTOP")
                    if plat["kind"] == "lift":
                        plat["state"] = "wait"
                    else:
                        del self.plats[sector]
            elif plat["state"] == "wait":
                plat["wait"] += 1
                if plat["wait"] >= PLAT_WAIT:
                    plat["state"] = "up"
                    self.events.append("DSPSTART")
            elif plat["state"] == "up":
                self.floors[sector] = min(plat["home"], self.floors[sector] + PLAT_SPEED)
                if self.floors[sector] >= plat["home"]:
                    self.events.append("DSPSTOP")
                    del self.plats[sector]
        p = self.player
        p["z"] = self.floors[p["sector"]]
        for m in self.monsters:
            m["z"] = self.floors[m["sector"]]

    # -- items ---------------------------------------------------------------------
    def _pick_up_items(self) -> None:
        p = self.player
        keep = []
        for item in self.items:
            if (item["x"] - p["x"]) ** 2 + (item["y"] - p["y"]) ** 2 > 36.0 ** 2 or abs(self.floors[item["sector"]] - p["z"]) > 24:
                keep.append(item)
                continue
            if not self._give(item["kind"]):
                keep.append(item)
        self.items = keep

    def _give(self, kind: int) -> bool:
        p = self.player
        sound = "DSITEMUP"
        if kind == 2011:
            if p["health"] >= 100:
                return False
            p["health"] = min(100, p["health"] + 10)
            self._say("Picked up a stimpack.")
        elif kind == 2012:
            if p["health"] >= 100:
                return False
            self._say("Picked up a medikit that you REALLY need!" if p["health"] < 25 else "Picked up a medikit.")
            p["health"] = min(100, p["health"] + 25)
        elif kind == 2014:
            p["health"] = min(200, p["health"] + 1)
            self._say("Picked up a health bonus.")
        elif kind == 2015:
            p["armor"] = min(200, p["armor"] + 1)
            p["armor_type"] = max(p["armor_type"], 1)
            self._say("Picked up an armor bonus.")
        elif kind == 2018:
            if p["armor"] >= 100:
                return False
            p["armor"], p["armor_type"] = 100, 1
            self._say("Picked up the armor.")
        elif kind == 2019:
            if p["armor"] >= 200:
                return False
            p["armor"], p["armor_type"] = 200, 2
            self._say("Picked up the MegaArmor!")
        elif kind in (2007, 2048):
            if p["bullets"] >= 200:
                return False
            p["bullets"] = min(200, p["bullets"] + (10 if kind == 2007 else 50))
            self._say("Picked up a clip." if kind == 2007 else "Picked up a box of bullets.")
        elif kind in (2008, 2049):
            if p["shells"] >= 50:
                return False
            p["shells"] = min(50, p["shells"] + (4 if kind == 2008 else 20))
            self._say("Picked up 4 shotgun shells." if kind == 2008 else "Picked up a box of shotgun shells.")
        elif kind in (2010, 2046):
            if p["rockets"] >= 50:
                return False
            p["rockets"] = min(50, p["rockets"] + (1 if kind == 2010 else 5))
            self._say("Picked up a rocket." if kind == 2010 else "Picked up a box of rockets.")
        elif kind == 2001:
            if WEAPON_SHOTGUN in p["weapons"]:
                if p["shells"] >= 50:
                    return False
                p["shells"] = min(50, p["shells"] + 8)
            else:
                p["weapons"].add(WEAPON_SHOTGUN)
                p["shells"] = min(50, p["shells"] + 8)
                p["pending"] = WEAPON_SHOTGUN
                self._set_face(f"STFEVL{self._face_row()}", 8, 2 * TICRATE)
            self._say("You got the shotgun!")
            sound = "DSWPNUP"
        else:
            return False
        p["bonus_count"] += 6
        self.events.append(sound)
        return True

    # -- weapons ---------------------------------------------------------------------
    def _tick_weapon(self, controls: dict) -> None:
        p = self.player
        fire = bool(controls.get("fire"))
        p["fire_held_tics"] = p["fire_held_tics"] + 1 if fire else 0
        if p["flash"] > 0:
            p["flash"] -= 1
        if p["weapon_state"] == "lower":
            p["weapon_raise"] += 8
            if p["weapon_raise"] >= 128:
                p["ready"], p["pending"] = p["pending"], None
                p["weapon_state"] = "raise"
            return
        if p["weapon_state"] == "raise":
            p["weapon_raise"] -= 8
            if p["weapon_raise"] <= 0:
                p["weapon_raise"], p["weapon_state"] = 0, "ready"
            return
        if p["weapon_state"] == "ready":
            if p["pending"] is not None:
                p["weapon_state"], p["weapon_raise"] = "lower", 0
                return
            if fire and self._can_fire():
                self._fire()
            elif fire:
                p["pending"] = WEAPON_PISTOL if p["ready"] == WEAPON_SHOTGUN and p["bullets"] > 0 else None
            return
        # in a fire sequence
        p["weapon_tics"] -= 1
        if p["weapon_tics"] <= 0:
            sequence = self._sequence()
            p["seq_index"] += 1
            if p["seq_index"] < len(sequence):
                frame, tics, sound = sequence[p["seq_index"]]
                p["weapon_tics"] = tics
                if sound:
                    self.events.append(sound)
            else:
                p["weapon_state"] = "ready"
                if fire and self._can_fire():
                    self._fire()

    def _sequence(self) -> list[tuple[str, int, str | None]]:
        if self.player["ready"] == WEAPON_SHOTGUN:
            return [("A", 3, None), ("B", 7, None), ("C", 5, None), ("D", 5, "DSSGCOCK"), ("C", 4, None), ("B", 5, None), ("A", 3, None)]
        return [("B", 4, None), ("C", 6, None), ("D", 4, None)]

    def _can_fire(self) -> bool:
        p = self.player
        return (p["bullets"] > 0) if p["ready"] == WEAPON_PISTOL else (p["shells"] > 0)

    def _fire(self) -> None:
        p = self.player
        p["weapon_state"], p["seq_index"], p["weapon_tics"] = "fire", 0, self._sequence()[0][1]
        if p["fire_held_tics"] > 2 * TICRATE:
            self._set_face(f"STFKILL{self._face_row()}", 5, TICRATE)
        if p["ready"] == WEAPON_PISTOL:
            p["bullets"] -= 1
            p["flash"] = 7
            self.events.append("DSPISTOL")
            spread = 0.0 if p["refire"] == 0 else self.random.uniform(-5.6, 5.6)
            self._hitscan(p["angle"] + math.radians(spread), 5 * self.random.randint(1, 3))
            p["refire"] += 1
        else:
            p["shells"] -= 1
            p["flash"] = 7
            self.events.append("DSSHOTGN")
            for _ in range(7):
                self._hitscan(p["angle"] + math.radians(self.random.uniform(-5.6, 5.6)), 5 * self.random.randint(1, 3))
        if p["fire_held_tics"] == 0:
            p["refire"] = 0

    def _hitscan(self, angle: float, damage: int, from_monster: dict | None = None) -> None:
        if from_monster is None:
            sx, sy, sz = self.player["x"], self.player["y"], self.player["z"] + VIEW_HEIGHT
        else:
            sx, sy, sz = from_monster["x"], from_monster["y"], from_monster["z"] + 32
        wall_distance, _line = self.wall_hit(sx, sy, sz, angle, 0.0, HITSCAN_RANGE)
        dx, dy = math.cos(angle), math.sin(angle)
        best, best_target = wall_distance, None
        targets = [self.player] if from_monster is not None else [m for m in self.monsters if m["alive"]] + [d for d in self.decorations if d["kind"] == 2035 and d["exploding"] < 0]
        for target in targets:
            rx, ry = target["x"] - sx, target["y"] - sy
            along = rx * dx + ry * dy
            if along <= 0 or along >= best:
                continue
            radius = target.get("radius", PLAYER_RADIUS if target is self.player else 10)
            if abs(rx * dy - ry * dx) <= radius:
                best, best_target = along, target
        if best_target is None:
            if wall_distance < HITSCAN_RANGE:
                self.effects.append(dict(prefix="PUFF", frames="ABCD", tics=4, x=sx + dx * (wall_distance - 8), y=sy + dy * (wall_distance - 8), z=sz + self.random.uniform(-8, 8), age=0, fullbright=True))
            return
        if best_target is self.player:
            self._damage_player(damage, from_monster)
        elif best_target.get("kind") == 2035:
            best_target["health"] -= damage
            if best_target["health"] <= 0:
                self._explode_barrel(best_target)
        else:
            self._damage_monster(best_target, damage)
            self.effects.append(dict(prefix="BLUD", frames="CBA", tics=8, x=best_target["x"] - dx * 8, y=best_target["y"] - dy * 8, z=best_target["z"] + 32 + self.random.uniform(-8, 8), age=0, fullbright=False))

    def _explode_barrel(self, barrel: dict) -> None:
        barrel["exploding"] = 0
        self.events.append("DSBAREXP")
        for target in [self.player] + [m for m in self.monsters if m["alive"]]:
            dist = math.hypot(target["x"] - barrel["x"], target["y"] - barrel["y"])
            if dist < 128 + 20:
                damage = max(0, int(128 - dist))
                if damage <= 0:
                    continue
                if target is self.player:
                    self._damage_player(damage, None)
                else:
                    self._damage_monster(target, damage)

    def _damage_player(self, damage: int, attacker: dict | None) -> None:
        p = self.player
        if p["dead"]:
            return
        if p["armor"] > 0:
            saved = damage // (3 if p["armor_type"] < 2 else 2)
            if p["armor"] <= saved:
                saved = p["armor"]
                p["armor_type"] = 0
            p["armor"] -= saved
            damage -= saved
        p["health"] -= damage
        p["damage_count"] = min(100, p["damage_count"] + damage)
        p["attacker"] = attacker
        if attacker is not None and damage < 20:
            rel = math.atan2(attacker["y"] - p["y"], attacker["x"] - p["x"]) - p["angle"]
            rel = (rel + math.pi) % (2 * math.pi) - math.pi
            face = "STFST%d1" % self._face_row() if abs(rel) < math.radians(30) else ("STFTL%d0" if rel > 0 else "STFTR%d0") % self._face_row()
        else:
            face = f"STFOUCH{self._face_row()}"
        self._set_face(face, 7, TICRATE)
        if p["health"] <= 0:
            p["health"], p["dead"], p["dead_tics"] = 0, True, 0
            p["face"], p["face_priority"], p["face_tics"] = "STFDEAD0", 10, 10_000
            self.events.append("DSPLDETH")
        else:
            self.events.append("DSPLPAIN")

    def _damage_monster(self, m: dict, damage: int) -> None:
        spec = MONSTER[m["kind"]]
        m["health"] -= damage
        if m["state"] == "idle":
            self._wake(m)
        if m["health"] <= 0:
            m["alive"], m["state"], m["state_tics"], m["frame"] = False, "die", 0, spec["death"][0]
            self.events.append(self.random.choice(spec["die"]))
            self.player["kills"] += 1
            return
        if self.random.randint(0, 255) < spec["pain_chance"]:
            m["state"], m["state_tics"], m["frame"] = "pain", spec["pain_tics"], "G" if m["kind"] == ZOMBIEMAN else "H"
            self.events.append(spec["pain"])

    # -- monsters ----------------------------------------------------------------------
    def _wake(self, m: dict) -> None:
        spec = MONSTER[m["kind"]]
        m["state"], m["state_tics"], m["move_tics"] = "chase", 0, 0
        self.events.append(self.random.choice(spec["sight"]))

    def _tick_monsters(self) -> None:
        p = self.player
        for m in self.monsters:
            spec = MONSTER[m["kind"]]
            if m["state"] == "die":
                m["state_tics"] += 1
                index = min(len(spec["death"]) - 1, m["state_tics"] // spec["death_tics"])
                m["frame"] = spec["death"][index]
                continue
            if not m["alive"]:
                continue
            dx, dy = p["x"] - m["x"], p["y"] - m["y"]
            dist = math.hypot(dx, dy)
            if m["state"] == "idle":
                if self.tick_count % 8 == 0 and not p["dead"] and dist < 1200:
                    facing = math.cos(math.atan2(dy, dx) - m["angle"]) > -0.2 or m["ambush"] is False and dist < 256
                    if facing and self.line_of_sight(m["x"], m["y"], m["z"] + 32, p["x"], p["y"], p["z"] + VIEW_HEIGHT):
                        self._wake(m)
                continue
            if m["state"] == "pain":
                m["state_tics"] -= 1
                if m["state_tics"] <= 0:
                    m["state"] = "chase"
                continue
            if m["state"] == "attack":
                self._tick_attack(m, spec, dist)
                continue
            # chase
            m["angle"] = math.atan2(dy, dx) if m["move_tics"] <= 0 else m["angle"]
            if m["move_tics"] <= 0:
                jitter = self.random.uniform(-0.8, 0.8) if dist > 96 else 0.0
                m["move_angle"] = math.atan2(dy, dx) + jitter
                m["move_tics"] = self.random.randint(8, 24)
                sees = self.line_of_sight(m["x"], m["y"], m["z"] + 32, p["x"], p["y"], p["z"] + VIEW_HEIGHT) and not p["dead"]
                if sees and spec["missile"] == "fireball" and dist < MELEE_RANGE + m["radius"] + PLAYER_RADIUS:
                    m["state"], m["state_tics"], m["attack_step"], m["melee"] = "attack", 0, 0, True
                    continue
                if sees and self.random.random() < max(0.05, 1.0 - dist / 1400.0) * 0.28:
                    m["state"], m["state_tics"], m["attack_step"], m["melee"] = "attack", 0, 0, False
                    m["angle"] = math.atan2(dy, dx)
                    continue
            m["move_tics"] -= 1
            nx = m["x"] + math.cos(m["move_angle"]) * spec["speed"]
            ny = m["y"] + math.sin(m["move_angle"]) * spec["speed"]
            nx, ny = self.move_actor(m["x"], m["y"], nx, ny, m["radius"], m["sector"], False)
            for other in self.monsters:
                if other is not m and other["alive"] and math.hypot(other["x"] - nx, other["y"] - ny) < m["radius"] + other["radius"]:
                    nx, ny = m["x"], m["y"]
                    m["move_tics"] = 0
                    break
            if math.hypot(nx - p["x"], ny - p["y"]) < m["radius"] + PLAYER_RADIUS:
                nx, ny = m["x"], m["y"]
            if (abs(nx - m["x"]) + abs(ny - m["y"])) < 0.5:
                m["move_tics"] = 0
            m["x"], m["y"] = nx, ny
            m["sector"] = self.sector_at(nx, ny)
            m["z"] = self.floors[m["sector"]]
            m["angle"] = m["move_angle"]
            m["frame"] = spec["walk"][(self.tick_count // 4) % 4]

    def _tick_attack(self, m: dict, spec: dict, dist: float) -> None:
        p = self.player
        steps = spec["attack"]
        frame, tics = steps[m["attack_step"] * 2], steps[m["attack_step"] * 2 + 1]
        m["frame"] = frame
        m["angle"] = math.atan2(p["y"] - m["y"], p["x"] - m["x"])
        m["state_tics"] += 1
        if m["state_tics"] >= tics:
            m["state_tics"] = 0
            m["attack_step"] += 1
            if m["attack_step"] * 2 >= len(steps):
                m["state"], m["move_tics"] = "chase", 0
                return
            if m["attack_step"] * 2 == len(steps) - 2:  # the firing frame
                if m["kind"] == ZOMBIEMAN:
                    self.events.append("DSPISTOL")
                    self._hitscan(m["angle"] + math.radians(self.random.uniform(-16, 16)), 3 * self.random.randint(1, 5), from_monster=m)
                elif m.get("melee") and dist < MELEE_RANGE + m["radius"] + PLAYER_RADIUS:
                    self.events.append("DSCLAW")
                    self._damage_player(3 * self.random.randint(1, 8), m)
                else:
                    self.events.append("DSFIRSHT")
                    self.projectiles.append(dict(x=m["x"] + math.cos(m["angle"]) * 24, y=m["y"] + math.sin(m["angle"]) * 24, z=m["z"] + 32,
                                                 angle=m["angle"], dz=((p["z"] + VIEW_HEIGHT) - (m["z"] + 32)) / max(dist, 1.0) * 10.0,
                                                 owner=m, age=0, exploding=-1))

    def _tick_projectiles(self) -> None:
        p = self.player
        keep = []
        for ball in self.projectiles:
            ball["age"] += 1
            if ball["exploding"] >= 0:
                ball["exploding"] += 1
                if ball["exploding"] < 18:
                    keep.append(ball)
                continue
            nx, ny, nz = ball["x"] + math.cos(ball["angle"]) * 10.0, ball["y"] + math.sin(ball["angle"]) * 10.0, ball["z"] + ball["dz"]
            wall_distance, _line = self.wall_hit(ball["x"], ball["y"], ball["z"], ball["angle"], ball["dz"] / 10.0, 10.0)
            sector = self.sector_at(nx, ny)
            hit_wall = wall_distance < 10.0 or nz < self.floors[sector] or nz > self.ceilings[sector]
            hit_player = not p["dead"] and math.hypot(nx - p["x"], ny - p["y"]) < PLAYER_RADIUS + 6 and p["z"] <= nz <= p["z"] + PLAYER_HEIGHT
            if hit_player:
                self._damage_player(3 * self.random.randint(1, 8), ball["owner"])
            if hit_wall or hit_player:
                ball["exploding"] = 0
                self.events.append("DSFIRXPL")
            else:
                ball["x"], ball["y"], ball["z"] = nx, ny, nz
            keep.append(ball)
        self.projectiles = keep

    def _tick_effects(self) -> None:
        for e in self.effects:
            e["age"] += 1
        self.effects = [e for e in self.effects if e["age"] < e["tics"] * len(e["frames"])]
        self.decorations = [d for d in self.decorations if d["exploding"] < 5 * 5]

    # -- lighting --------------------------------------------------------------------
    def sector_lights(self) -> list[float]:
        lights = []
        tick = self.tick_count
        for k, (floor, ceiling, ff, cf, light, special, tag) in enumerate(self.level.sectors):
            low = self.min_light[k]
            if special == 1:
                until, level = self.flicker.get(k, (0, light))
                if tick >= until:
                    level = low if level == light else light
                    until = tick + (self.random.randint(1, 7) if level == light else self.random.randint(1, 4)) * 4
                    self.flicker[k] = (until, level)
                lights.append(level)
            elif special in (2, 12):
                lights.append(light if (tick % 20) < 5 else low)
            elif special in (3, 13):
                lights.append(light if (tick % 40) < 5 else low)
            elif special == 8:
                span = max(light - low, 0)
                lights.append(low + span * (0.5 + 0.5 * math.sin(tick * 8 / max(span, 1) * 0.5)))
            elif special == 17:
                lights.append(max(low, light - self.random.randint(0, 3) * 16))
            else:
                lights.append(light)
        return lights

    # -- the bag every frame is drawn from --------------------------------------------
    def snapshot(self) -> dict:
        p = self.player
        tick = self.tick_count
        bob_angle = (tick % 64) / 64.0 * 2 * math.pi
        view_z = p["z"] + (VIEW_HEIGHT if not p["dead"] else max(6.0, VIEW_HEIGHT - p["dead_tics"] * 1.5)) + p["bob"] / 2.0 * math.sin(bob_angle)
        things = []
        for m in self.monsters:
            spec = MONSTER[m["kind"]]
            light = self.lights[m["sector"]]
            things.append([m["x"], m["y"], float(self.floors[m["sector"]]), spec["prefix"], m["frame"], math.degrees(m["angle"]), 0, light])
        for item in self.items:
            prefix, frames, fullbright = ITEM[item["kind"]]
            things.append([item["x"], item["y"], float(self.floors[item["sector"]]), prefix, frames[(tick // 6) % len(frames)], 0.0, int(fullbright), self.lights[item["sector"]]])
        for d in self.decorations:
            if d["exploding"] >= 0:
                things.append([d["x"], d["y"], float(self.floors[d["sector"]]), "BEXP", "ABCDE"[min(4, d["exploding"] // 5)], 0.0, 1, 255.0])
                continue
            prefix, frames, fullbright = (SOLID_DECORATION.get(d["kind"]) or PASSABLE_DECORATION[d["kind"]])
            things.append([d["x"], d["y"], float(self.floors[d["sector"]]), prefix, frames[(tick // 6) % len(frames)], 0.0, int(fullbright), self.lights[d["sector"]]])
        for ball in self.projectiles:
            frame = "AB"[(tick // 4) % 2] if ball["exploding"] < 0 else "CDE"[min(2, ball["exploding"] // 6)]
            things.append([ball["x"], ball["y"], ball["z"] - 8.0, "BAL1", frame, 0.0, 1, 255.0])
        for e in self.effects:
            things.append([e["x"], e["y"], e["z"] - 8.0, e["prefix"], e["frames"][min(len(e["frames"]) - 1, e["age"] // e["tics"])], 0.0, int(e["fullbright"]), 192.0])
        if p["damage_count"] > 0:
            palette = min(8, 1 + (p["damage_count"] + 7) // 8)
        elif p["bonus_count"] > 0:
            palette = min(12, 9 + (p["bonus_count"] + 7) // 8)
        else:
            palette = 0
        weapon_frame = "A"
        if p["weapon_state"] == "fire":
            weapon_frame = self._sequence()[p["seq_index"]][0]
        raise_offset = p["weapon_raise"] if p["weapon_state"] in ("lower", "raise") else 0
        return {
            "tick": tick, "x": p["x"], "y": p["y"], "z": view_z, "angle": p["angle"], "sector": p["sector"],
            "extralight": 2 if p["flash"] > 0 else 0, "lights": self.sector_lights(),
            "floors": self.floors.tolist(), "ceilings": self.ceilings.tolist(), "things": things,
            "hud": {
                "bullets": p["bullets"], "shells": p["shells"], "rockets": p["rockets"], "cells": 0, "health": p["health"], "armor": p["armor"],
                "weapons": sorted(p["weapons"]), "ready": p["ready"], "face": p["face"], "message": p["message"], "dead": p["dead"],
                "weapon_frame": weapon_frame, "flash": p["flash"], "raise": raise_offset,
                "bob_x": p["bob"] * math.cos(bob_angle), "bob_y": p["bob"] * abs(math.sin(bob_angle)),
                "kills": p["kills"], "monsters": len(self.monsters), "complete": self.level_complete > 0,
            },
            "palette": palette, "events": list(self.events),
        }
