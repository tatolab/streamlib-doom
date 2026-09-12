#!/bin/bash
# A cinematic sequence of director commands over the game's control endpoint,
# for recording. Runs alongside the autopilot. Everything here is exactly what
# Claude issues when it directs; scripting it makes the recording repeatable.
set -u
D=http://127.0.0.1:8668/director
T="${STREAMLIB_DOOM_TRANSCRIPT:-/tmp/streamlib-doom-transcript.log}"
say() { printf '> %s\n%s\n\n' "$1" "$2" >> "$T"; }
fire() { curl -s -X POST "$D" -d "$2" >/dev/null; say "$1" "$3"; }

sleep 6
say "look at the game and tell me what is happening" "The marine is walking out of the hangar toward the slime courtyard on autopilot, 100% health, pistol only. Quiet so far. Let me change that."
sleep 5
fire "give them a fighting chance first" '{"command":"give","item":"everything"}' "Added a DirectorCommand to the graph: full health, armor, shotgun and ammo. Now it is a fair fight."
sleep 6
fire "make it tense — kill the lights" '{"command":"lights","factor":0.28}' "Dropped the sector lights to 0.28. The hangar just got a lot darker."
sleep 7
fire "ambush them" '{"command":"spawn","kind":"imp","count":4,"where":"behind"}' "Teleported four imps in behind the player. Monsters alive jumped by four; watch the marine turn and fight."
sleep 9
fire "give me a horror filter" '{"command":"effect","effect":"night_vision"}' "Spliced the compositor to night vision — every frame the phone and the recorder see is now green-on-black."
sleep 9
fire "post a taunt on the HUD" '{"command":"message","text":"THEY KNOW YOU ARE HERE"}' "Wrote a message onto the HUD in Doom's own font."
sleep 8
fire "now go thermal" '{"command":"effect","effect":"thermal"}' "Switched the filter to thermal — cold blues for the walls, hot orange for anything alive."
sleep 9
fire "one more wave, zombiemen this time" '{"command":"spawn","kind":"zombieman","count":4,"where":"ahead"}' "Four zombiemen ahead now too. The courtyard is a crossfire."
sleep 9
fire "back to normal, let them breathe" '{"command":"effect","effect":"none"}' "Cleared the effect and let the level look like 1993 again. Good run."
sleep 6
