---
name: media-architecture
description: Image prompt rules for buildings, interiors, cityscapes and landscapes with
  architecture, in the active image model's prompt family.
version: 0.1.0
requires_tools:
  - generate_image
family_sections: true
origin: synthesized
status: active
content_sha256: cdf0a7955d6e79ee3f6c5b279c159a1ea0021195e2539a2ee4a1e9d4aa3b4fb0
approved_by: agent:dev-builder (not a human approval)
approved_at: '2026-09-27T19:39:49Z'
---

# Architecture Image Prompt Rules

Use this skill when the image is about a place rather than a person: building exteriors
and facades, groups of buildings, room interiors, streets and cityscapes. For people in a
place, load `media-character` instead - the place is then the background. For technical
drawings, floor plans and cross-sections load `media-engineering`.

What follows the next heading is written for the prompt family of the image model that is
active now, and only for that family. Follow it as written: it states the grammar the
prompt must use, the order of its parts, and what goes in the negative prompt.

Always write the prompt in English, whatever language the user wrote in. Translate the
user's intent; never copy their words into the prompt.

Architecture prompts exclude people unless the user asks for them, and they never carry
body or anatomy terms, in the prompt or in the negative prompt: there is no body in the
frame to correct.

## family: danbooru

The active model reads English Danbooru tags, separated by commas. No sentences, no
non-English text. It is an illustration model: expect an illustrated or anime-background
look rather than an architectural photograph.

### Prompt order

1. **No people**: start with `no humans, scenery`.
2. **Indoors or outdoors, and the kind of place**: `outdoors, building, house, skyscraper,
   city, cityscape, street, bridge, castle, shrine, ruins` or `indoors, living room,
   bedroom, kitchen, classroom, library, cafe, office, hallway`.
3. **Architectural style**: `architecture`, `east asian architecture`,
   `gothic architecture`.
4. **Structure and furnishings**: `window, balcony, stairs, column, rooftop, door, fence,
   wooden floor, stone floor, brick wall, tatami, shouji, couch, table, chair, lamp,
   bookshelf, potted plant, plant`.
5. **View**: exactly one of `wide shot`, `from above`, `from below`, and optionally
   `perspective`, `vanishing point`, `symmetry`.
6. **Light and time**: `sunlight, light rays, day, blue sky, cloud, sunset, night,
   reflection`.
7. **Quality tags last**: `masterpiece, best quality, newest, absurdres`.

Example: `no humans, scenery, indoors, living room, architecture, window, wooden floor, couch, table, bookshelf, potted plant, wide shot, perspective, sunlight, light rays, day, masterpiece, best quality, newest, absurdres`

### Negative prompt

`worst quality, low quality, 1girl, 1boy, multiple girls, multiple boys, blurry, fisheye, dutch angle, text, watermark`

No anatomy tags (`bad anatomy`, `bad hands`, `missing fingers`): there is no person in the
frame.

## family: prose

The active model reads descriptive English prose. Write full sentences in the language of
architectural photography, not tag lists. Do not use Danbooru tags - no `no humans`,
`masterpiece`, `best quality`, `newest` or `absurdres`. No non-English text.

### What the prompt describes, in this order

1. **The building or room and its style**: modern, brutalist, minimalist, traditional
   wooden, glass-facade; what kind of building or room it is.
2. **Structure and materials**: cantilevers, floor-to-ceiling windows, an open plan, a
   double-height ceiling, a patio or balcony; raw concrete, polished hardwood, marble,
   stone, frosted glass, steel beams.
3. **The camera**: "architectural photograph, 24mm wide-angle lens, level horizon,
   corrected verticals", or a tilt-shift view; symmetrical composition where it suits.
4. **Light and surroundings**: ambient sunlight, soft interior lighting, time of day,
   weather, and what surrounds the building.
5. **Say that the space is empty of people** unless the user asked for them.

Example: `An architectural photograph of a minimalist concrete house with a long cantilevered upper floor and floor-to-ceiling windows, set on a grassy hillside. Raw concrete walls meet warm polished hardwood visible through the glass. Shot with a 24mm wide-angle lens at eye level, level horizon and straight verticals, symmetrical composition. Soft late-afternoon sunlight rakes across the facade; the scene is empty of people.`

### Negative prompt

Leave the negative prompt empty (`""`). This model family does not use it, and the image
tool discards it. Exclude people by saying the space is empty, in the prompt itself.

## family: generic

The active model's prompt grammar is not known. Write concise English: the building or
room as a noun phrase first, then comma-separated descriptive phrases. Avoid Danbooru-only
tags and long narrative paragraphs alike. No non-English text.

### Prompt pattern

`[building or room noun phrase], [style], [structure and materials], [camera and lens], [lighting and time of day], no people, highly detailed`

Example: `A modern glass-facade office tower beside a river, brutalist concrete podium, steel beams, architectural photography, 24mm wide angle, level horizon, golden hour sunlight, no people, highly detailed`

### Negative prompt

`worst quality, low quality, people, person, crowd, blurry, tilted horizon, distorted perspective, fisheye, text, watermark`

No anatomy terms (`bad anatomy`, `bad hands`): there is no person in the frame.
