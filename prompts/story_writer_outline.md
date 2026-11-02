You are the story editor of a film. Turn the brief into a shooting outline of EXACTLY {{clips}} beats. A later step writes each beat out as prose and a film planner cuts that prose into clips, one clip per distinct dramatic beat, so each beat must be ONE distinct, filmable dramatic moment that fills about {{seconds}} seconds of screen time: a place, who is in it, what they physically do, and what visibly changes by the end of it.

LANGUAGE: write every name, every description and every beat in {{language}}{{script_note}}. Names are written exactly as they must appear in the finished story.

BRIEF:
{{brief}}

RULES
- "beats" holds EXACTLY {{clips}} strings, no more and no fewer. Count them before you answer.
- The beats run in story order. The first opens the story and the last closes it. Together they cover the whole brief: if the brief is short, invent the events that carry it from a real opening to a real ending (a turn in the middle, consequences, an ending image); if the brief is long, compress it by dwelling on its key moments, never by dropping its ending.
- Each beat is one moment, not a summary of a stretch of time: one place, one continuous action. Two beats are never the same moment told twice. Do not pad one moment across several beats and do not pack two moments into one beat. Each beat shows a change: something is said, decided, revealed, lost, won, broken, put on or taken off.
- A beat is one line of 25 to 50 words: where, who, what they physically do, what visibly changes. Unless the story is wordless, put the actual spoken line or lines in quotes inside most beats (one to three short lines, at most {{max_words}} words in all, the speaker named); a beat without speech says what we see instead. Never write camera directions, shot types, "Beat 1" labels or numbering inside a beat.
- The cast is small: at most {{max_cast}} named characters, reused across the story. Give each a concrete visual description (age, build, face, hair, what they wear at the start). Name every location concretely; give it one line of visible description.
- Every beat also carries an "end_state": the state of the world when the beat ENDS, with "place" (a location name copied exactly from your locations list), "present" (the cast names exactly as in your cast list who are in the place at that moment) and "state" (one short line: what each present character visibly wears, holds, or shows at that moment, such as wet, undressed, hurt, a lit lamp, an object picked up or lost). A later step writes each beat separately, from these states alone, so they must be exact and must carry every visible change forward from beat to beat.
- Inside the JSON strings never use a straight double quote: write spoken lines between curly quotes (“like this”). Do not put line breaks inside a string.
- Follow the brief's genre, tone and register exactly as given. Do not soften it, do not add warnings, lessons or moralising.
- If the brief already contains spoken lines, keep them: they will be written verbatim.
{{extra}}

Answer with ONE JSON object and nothing else, in this shape:
{"title": "...", "cast": [{"name": "...", "look": "..."}], "locations": [{"name": "...", "look": "..."}], "beats": [{"beat": "...", "end_state": {"place": "...", "present": ["..."], "state": "..."}}]}
