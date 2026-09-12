"""One instruction to the running game, as a processor an agent adds over MCP.

The catalog serves this class's config schema, so an agent holding only the
node's URL learns the verbs from `/api/registry`: add a `DirectorCommand` with
the command in its config, connect its `command` output to the game's
`director_from_upstream`, and the game does it. It publishes its instruction
for two seconds — long enough to cover a link wired after its setup — then
idles, and can be removed.
"""
from __future__ import annotations

import dataclasses
import json
import os
import urllib.request
from typing import Annotated, Literal

from streamlib import RuntimeContextFullAccess, RuntimeContextLimitedAccess, log, output, processor

DIRECTOR_ENDPOINT = os.environ.get("STREAMLIB_DOOM_DIRECTOR_URL", "http://127.0.0.1:8668/director")


@dataclasses.dataclass
class DirectorCommandConfig:
    command: Annotated[
        Literal["spawn", "give", "lights", "effect", "message", "god", "heal", "autopilot", "mission", "style", "repaint", "control", "face", "clear"],
        "What to do: spawn monsters, give the player things, set the lights, set a screen effect, post a message, god mode, heal, hand the marine to the autopilot, give the planner a mission, set the diffusion re-render's style, or repaint the level's textures.",
    ]
    style: Annotated[str, "For style: a preset (lego, cyberpunk, bladerunner, night_city, photoreal, anime, claymation, watercolor, alien, none) or any free-text prompt for the diffusion re-render. For repaint: the material or look the generated wall and floor textures should have, e.g. marble, rusted copper, candy."] = "photoreal"
    goal: Annotated[Literal["patrol", "courtyard", "hangar", "hold"], "For mission: where the planner takes the robot. patrol loops the level; courtyard and hangar go there and hold; hold stops."] = "patrol"
    mode: Annotated[Literal["stop", "manual", "auto"], "For control: who drives. stop and manual switch every machine driver off and leave the controls to whoever is playing; auto gives them back to the planner."] = "auto"
    kind: Annotated[Literal["imp", "zombieman"], "For spawn: which monster."] = "imp"
    count: Annotated[int, "For spawn: how many, 1 to 8."] = 1
    where: Annotated[Literal["behind", "ahead", "left", "right"], "For spawn: where, relative to the way the player faces."] = "behind"
    distance: Annotated[int, "For spawn: how far away in map units, 64 to 900. 128 is right in front; 400 is across the room, in shot but not in your face."] = 128
    item: Annotated[Literal["shotgun", "health", "armor", "ammo", "everything", "disarm"], "For give: what the player receives; disarm takes all ammunition away so the planner keeps moving instead of fighting."] = "shotgun"
    effect: Annotated[Literal["none", "crt", "night_vision", "invulnerable", "thermal"], "For effect: the screen filter the compositor bakes into every frame."] = "night_vision"
    factor: Annotated[float, "For lights: 0.1 is nearly dark, 1.0 is the level as authored, 1.5 is overlit."] = 1.0
    text: Annotated[str, "For message: shown on the HUD's message line in Doom's font, up to 60 characters."] = ""
    on: Annotated[bool, "For god and autopilot: on or off."] = True


@processor(
    execution="manual",
    description="Fires one instruction into the running DOOM game the moment it is added: spawn monsters behind the player, give them the shotgun, dim the lights, set a screen effect, post a message, god mode, heal, autopilot. Just `add_processor` it with a config; it delivers over the game's control endpoint and needs no link. Remove it afterwards.",
)
class DirectorCommand:
    def __init__(self, config: DirectorCommandConfig) -> None:
        self.command = dataclasses.asdict(config)

    @output()
    def command_to_game(self) -> None: ...

    def setup(self, ctx: RuntimeContextFullAccess) -> None:
        # Delivered from this helper process over HTTP, so it never depends on an
        # engine link wired after the game's setup — it lands whether or not this
        # processor is ever connected to anything.
        import time
        for attempt in range(20):
            try:
                request = urllib.request.Request(DIRECTOR_ENDPOINT, data=json.dumps(self.command).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=5) as response:
                    log.info(f"MARKER:DIRECTOR_SENT {response.status} {json.dumps(self.command)}")
                return
            except Exception as failure:
                last = failure
                time.sleep(0.5)
        log.info(f"MARKER:DIRECTOR_FAILED {last!r}")

    def process(self, ctx: RuntimeContextLimitedAccess) -> None:
        pass
