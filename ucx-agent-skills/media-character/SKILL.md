---
name: media-character
description: Image prompt rules for characters - solo portraits, character sheets, multi-pose
  sets and two-person interactions - in the active image model's prompt family.
version: 0.1.0
requires_tools:
  - generate_image
family_sections: true
origin: synthesized
status: active
content_sha256: 2c04d628e32a968803edf6aeaaaa49dda299579428c5cb8092c4e7e7815472a2
approved_by: kennylim (owner approval, 2026-09-28)
approved_at: '2026-09-28T00:00:00Z'
---

# Character Image Prompt Rules

Use this skill when the image is about one or more characters: a solo character, a
character sheet, several poses of the same character, or two people interacting (hugging,
holding hands, fighting). For buildings, rooms and cityscapes load `media-architecture`;
for technical drawings and diagrams load `media-engineering`.

What follows the next heading is written for the prompt family of the image model that is
active now, and only for that family. Follow it as written: it states the grammar the
prompt must use, the order of its parts, and what goes in the negative prompt.

Always write the prompt in English, whatever language the user wrote in. Translate the
user's intent; never copy their words into the prompt.

If the character already has a saved sheet, fetch it with
`character_sheet(action='get', character_id=...)` first and keep its appearance details
identical across every prompt you write for that character.

## family: danbooru

The active model reads English Danbooru tags, separated by commas. No sentences, no
narrative, no non-English text.

### Prompt order for one character

1. **Subject count**: `1girl, solo` or `1boy, solo`.
2. **Appearance and outfit**: hair, eyes, clothing, held items, e.g.
   `long hair, black hair, blue eyes, knight, armor, cape, holding sword`.
3. **One pose and one camera angle**: exactly one posture (`standing`, `sitting`,
   `kneeling`, `running`, `looking back`, `fighting stance`) and one framing
   (`full body`, `cowboy shot`, `upper body`, `from side`, `from below`, `from above`).
   Never list several poses in one prompt, and never rely on a vague pose tag alone.
4. **Background**: `forest, tree, moonlight, light particles, night, fantasy`.
5. **Quality tags last**: `masterpiece, best quality, newest, absurdres`.

### Two people interacting

Attribute bleeding - one person's hair or clothes appearing on the other, or two bodies
fusing - is the main failure. Tag order and wording reduce it; they do not remove it.
Measured on an Illustrious model over five overlapping poses (hug, hug from behind,
piggyback, sitting on lap, arm hug), 60 images per variant, scored blind.
Write the tags in this order:

`[subject count] + [interaction] + [camera angle] + [subject A traits] + [subject B traits] + [background] + [quality]`

1. **Subject count first, always**: `1girl, 1boy`, `2girls` or `2boys`. Never `solo`,
   never omit the count.
2. **Interaction tags** straight after the count:
   - Affection: `hug`, `hug from behind`, `head on another's shoulder`, `closed eyes`,
     `smile`.
   - Companionship: `holding hands`, `interlocked fingers`, `walking`,
     `looking at another`, `eye contact`.
   - Combat: `duel`, `fighting stance`, `facing another`, `sparks`, `holding sword`.
3. **One camera angle** for the whole scene: `full body`, `cowboy shot`, `upper body`,
   `from side`, `wide shot`.
4. **Each subject's traits as one uninterrupted group.** Finish subject A before starting
   subject B; never interleave. Make the two contrast in hair colour, hair length and
   outfit.
5. **Fuse the hair colour into the hairstyle tag.** Write the colour inside a hairstyle
   tag, not as separate colour and length tags: `very long black hair, black hime cut`
   rather than `black hair, very long hair, straight hair`; `short blonde bob, blonde bob
   cut` or `short blonde hair` rather than `blonde hair, short hair, bob cut`. This was
   the one change that helped in every pose.
6. Background, then quality tags.

Do not add role anchors (`maid`, `knight`, `office lady`) to keep people apart: their own
clothing bleeds across like any other tag. A descriptive sentence and `BREAK`-style
splitting did not help either.

#### One girl and one boy

- Count as `1girl, 1boy` and write **the girl's group first**. With the boy's group
  first, the full pass rate halved (40/60 → 21/60).
- Clothing binds to the right person almost always (a dress goes to the girl). What
  swaps is hair colour: the usual failure is a blonde girl and a black-haired boy.
  Fused hair tags took full passes from 5/60 to 40/60.

#### Two girls (two boys not yet tested)

- Fused hair tags fix hair length (16/60 → 59/60 right).
- Outfits are not fixed by any wording tried: for a given seed, each garment lands on
  the same body whatever the prompt says (17/60 right in every variant). Removing
  outfit tags did not help.

#### When it still mixes

Most remaining failures belong to the seed, not the wording: some seeds failed under
every fused-tag variant tested. Regenerate with a different seed rather than rewording.
Region prompts (one prompt per half of the canvas) stop the bleed only when the two people stand apart;
for overlapping poses they draw extra people or break the pose.

Examples:

- Holding hands: `1girl, 1boy, holding hands, looking at another, cowboy shot, very long blonde hair, blonde hime cut, white dress, short black hair, black undercut, suit, outdoors, street, day, masterpiece, best quality, newest`
- Duel: `2girls, duel, fighting stance, facing another, sparks, full body, black ponytail, long black hair, red armor, katana, short blonde bob, blonde bob cut, silver armor, holding sword, ruins, night, masterpiece, best quality, newest`

Prefer a wide aspect ratio (`16:9` or `4:3`) for two people.

### Negative prompt

`worst quality, low quality, bad anatomy, bad hands, missing fingers, extra digits, deformed, blurry, text, signature, watermark`

Add a subject you want excluded only when the scene calls for it - for example `animal`
when a stray animal keeps appearing. It is not a default: it would suppress every
request that includes one.

### Several poses of one character

When asked for several images with different poses ("다양한 포즈 5장", "10 different poses"):

1. Never put several poses in one prompt with `count=N`; the model cannot reconcile them
   in one canvas and produces malformed bodies.
2. Write N complete prompts. Keep the subject count and appearance tags identical in each;
   change only the one pose and the one camera angle.
3. Send them in one call: `generate_image(prompts=[prompt_1, prompt_2, ...])`.

## family: prose

The active model reads descriptive English prose. Write full sentences, not tag lists.
Do not use Danbooru tags - no `1girl`, `solo`, `masterpiece`, `best quality`, `newest` or
`absurdres`; they degrade this model's text encoder. No non-English text.

### What the prompt describes, in this order

1. **The subject**: who they are, their face and expression, hair, clothing and its
   materials, what they hold.
2. **The pose and the camera**: one posture, and the framing and lens, e.g. "a full-length
   shot at eye level, 50mm lens, shallow depth of field".
3. **The setting and the light**: concrete light sources and atmosphere, e.g. "moonlight
   filtering through tall pines, faint glowing motes in the air".

Example: `A young female knight in polished silver plate armor and a long blue cape stands in a moonlit forest, gripping a softly glowing sword in both hands. Her long black hair is tied back and her expression is calm and focused. Full-length shot at eye level with a 50mm lens and shallow depth of field; cold moonlight falls through tall pines and faint glowing motes drift around her.`

### Two people interacting

1. State the number of people and the interaction in the first sentence: "Two people
   hold hands while walking down a city street."
2. Then describe each person in their own sentence, introduced by a distinct noun that
   belongs to them alone ("The man ...", "The woman ...", "The taller knight ...", "The
   swordswoman ..."). Never describe both in one clause, and never share an adjective
   across them.
3. Make the two contrast in hair, clothing and silhouette, and end with the camera and the
   light.

Example: `Two people hold hands while walking down a quiet city street at dusk. The man has short black hair and wears a dark tailored suit. The woman has long blonde hair and wears a flowing white summer dress. Medium-wide shot from the side, warm streetlights and soft evening haze.`

### Negative prompt

Leave the negative prompt empty (`""`). This model family does not use it, and the image
tool discards it.

### Several poses of one character

Write N complete descriptions that keep the character's appearance sentences word for
word identical and change only the pose and camera sentence, then send them in one call:
`generate_image(prompts=[...])`. Never describe several poses in one prompt.

## family: generic

The active model's prompt grammar is not known. Write concise English: the subject as a
noun phrase first, then comma-separated descriptive phrases. Avoid Danbooru-only tags
(`1girl`, `solo`) and long narrative paragraphs alike. No non-English text.

### Prompt pattern

`[subject noun phrase], [appearance and outfit], [one pose], [camera framing], [setting], [lighting], highly detailed`

Example: `A female knight in ornate silver plate armor, long black hair, holding a glowing sword, standing, full-body shot, moonlit forest, dramatic cinematic lighting, highly detailed`

### Two people interacting

Name the count and the interaction first, then each person as a separate phrase with a
noun of its own: `Two people holding hands, a man with short black hair in a dark suit, a woman with long blonde hair in a white dress, walking down a city street, medium-wide shot, warm evening light`.

### Negative prompt

`worst quality, low quality, blurry, deformed, bad anatomy, bad hands, extra limbs, text, signature, watermark`

### Several poses of one character

Write N prompts, one pose and one camera angle each, keeping the appearance phrases
identical, and send them via `generate_image(prompts=[...])`.
