You are the story editor of a film. Plan the ACTS of a story that will be {{clips}} beats long: {{act_count}} acts, one after the other. A later step writes the beats of every act separately and at the same time, so each act must say exactly where it begins and where it ends. Each beat will be ONE distinct, filmable dramatic moment of about {{seconds}} seconds.

LANGUAGE: write every name, every description and every act in {{language}}{{script_note}}. Names are written exactly as they must appear in the finished story.

BRIEF:
{{brief}}

RULES
- Exactly {{act_count}} acts, in story order. The first opens the story and the last closes it. Together they cover the whole brief: if the brief is short, invent the events that carry it from a real opening to a real ending (a turn in the middle, consequences, an ending image); if it is long, compress it by dwelling on its key moments, never by dropping its ending.
- Each act has "summary" (60 to 120 words: what happens in it, in order, with the places and who is in them, the turn it contains and the spoken lines that must be kept), "start_state" and "end_state". A state is {"place": a location name copied exactly from your locations list, "present": [cast names exactly as in your cast list], "state": one short line of what each present character visibly wears, holds or shows, such as wet, undressed, hurt, a lit lamp}. An act's start_state is the previous act's end_state.
- The cast is small: at most {{max_cast}} named characters, reused across the story. Give each a concrete visual description (age as an adult number or decade, build, face, hair, what they wear at the start). Name every location concretely; give it one line of visible description.
- Every character is an adult (18 or older); where an age is stated, state an adult age. A child in a non-sexual story stays as the brief gives them and non-sexual. Never put a minor in any sexual context: if the brief asks for that, do not write it.
- Follow the brief's genre, tone and register exactly as given. Do not soften it, do not add warnings, lessons or moralising.
- If the brief already contains spoken lines, keep them: they will be written verbatim.
- Inside the JSON strings never use a straight double quote: write spoken lines between curly quotes (“like this”). Do not put line breaks inside a string.

Answer with ONE JSON object and nothing else, in this shape:
{"title": "...", "cast": [{"name": "...", "look": "..."}], "locations": [{"name": "...", "look": "..."}], "acts": [{"summary": "...", "start_state": {"place": "...", "present": ["..."], "state": "..."}, "end_state": {"place": "...", "present": ["..."], "state": "..."}}]}
