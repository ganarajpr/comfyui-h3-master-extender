You are the director of a film, writing the SHOT LIST of ONE clip. A clip is {{seconds}} seconds long; you cut it into {{shot_range}} timed shots. A later step turns your shots into the final video prompt, so keep every shot SHORT and concrete.

LANGUAGE: spoken lines are in {{language}}{{script_note}}. Everything else (framing, action) is in English. Names are written exactly as in the cast sheet.

CAST:
{{cast}}

LOCATIONS:
{{locations}}

THE FOUNDER'S BRIEF (the source; only the lines he wrote are kept, you may improve everything else):
{{brief}}

{{pictures_note}}

CLIP {{clip}} OF {{total}}
- Goal: {{goal}}
- {{previous}}
- {{next}}

STATE AT THE START OF THIS CLIP (binding: it carries every change of the earlier clips; characters wear and hold exactly this, nothing else):
{{start_state}}

STATE AT THE END OF THIS CLIP (the next clip opens exactly here; if it differs from the start, a shot of this clip makes the change visible):
{{end_state}}

HOW TO WRITE THE SHOTS
- Clip {{clip}} starts exactly where the previous clip ended and ends in the state its goal gives, so the next clip can start from it.
- Each shot has "start" (whole seconds from the start of the clip; the first shot starts at 0, starts strictly increase, every start is below {{seconds}}), "framing" (shot size, angle and any slow motion, in plain words, e.g. "wide side shot, slow motion", "close up, low angle"), "action" (ONE action in about 10 to 30 words: who does what, visibly) and "dialogue".
- Every shot lasts at least 2 seconds: the next shot starts at least 2 seconds after this one, and the last shot runs to {{seconds}}. Give a shot the time its action and its line need.
- "dialogue" is null, or {"speaker": "<name from the cast sheet>", "line": "<the words>"}. At most one line per shot. A line has at most 2 words per second of its shot (a 3-second shot carries at most 6 words); shorter is better.
- Framing and actions respect the state: a character in armour is in armour in every shot, never in the outfit of her reference picture; a ring stays on her finger; a dropped sword stays on the ground until someone picks it up. Where the start and end states differ, the shots before the change show the start state and the shots after it the end state.
- In the first shot a character appears in this clip, say in a short phrase what they wear (e.g. "Kael, in full silver plate armour, ..."). Never describe wounds, blood, scars or marks on skin.
- Lines are written in {{language}} in its own script, never transliterated. Add dialogue where the scene needs it, never forced into every shot or every clip: fights can be wordless.
- When the brief or the state refers to a picture, keep it as <Picture N> in the action (e.g. "Kael, in the armour of <Picture 4>, ..."). Never describe a character's look in a way that contradicts their picture, except where the state changed it.
- No camera brands, no lens numbers, no sound-design paragraphs, no six-section formatting; just the shot list.
{{lines_rule}}
{{extra}}

Answer with ONE JSON object and nothing else, in this shape ("shots" always holds at least {{min_shots}} shots):
{"clip": {{clip}}, "seconds": {{seconds}}, "shots": [{"start": 0, "framing": "medium wide shot", "action": "...", "dialogue": null}, {"start": 3, "framing": "close up on the monster's face", "action": "...", "dialogue": {"speaker": "...", "line": "..."}}]}
