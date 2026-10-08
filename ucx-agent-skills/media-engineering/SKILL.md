---
name: media-engineering
description: Image prompt rules for technical drawings - mechanical assemblies, schematics,
  cross-sections, blueprint and CAD-style diagrams - in the active image model's prompt family.
version: 0.1.0
requires_tools:
  - generate_image
family_sections: true
origin: synthesized
status: active
content_sha256: 35efcaa8812c98bade1a0b8a61038c75a31de7a809fb1b52661df22bd29589c9
approved_by: agent:dev-builder (not a human approval)
approved_at: '2026-09-27T19:39:49Z'
---

# Engineering Drawing Image Prompt Rules

Use this skill when the image is a technical drawing: a mechanical assembly or exploded
view, an electrical or piping schematic, a structural cross-section or cutaway, a
blueprint, or a patent- or CAD-style line drawing. For photographs or illustrations of
buildings load `media-architecture`; for people load `media-character`.

What follows the next heading is written for the prompt family of the image model that is
active now, and only for that family. Follow it as written: it states the grammar the
prompt must use, the order of its parts, and what goes in the negative prompt.

Always write the prompt in English, whatever language the user wrote in. Translate the
user's intent; never copy their words into the prompt.

An image model draws the look of a technical drawing, not a correct one: dimensions,
proportions and connections are approximate, and any text or labels it draws are usually
illegible. Do not promise accuracy. If the user needs readable labels, suggest adding them
afterwards. Engineering prompts never carry body or anatomy terms, in the prompt or in the
negative prompt.

## family: danbooru

### Fit notice - tell the user first

The active model is an anime illustration model trained on Danbooru tags. It is a poor fit
for technical drawings: it produces a stylized, illustrative impression of a diagram, with
approximate geometry and invented, unreadable labels. **Before you call `generate_image`,
tell the user this in their own language**, and say what suits the job better: a
general-purpose photographic or prose-prompted model (a FLUX-class model, for example) for
a cleaner technical look, or a CAD or vector drawing tool for anything that must be
dimensionally correct. Then generate the best approximation this model can give, unless
the user says they would rather not.

### Grammar

English Danbooru tags, separated by commas. No sentences, no non-English text.

### Prompt order

1. **No people**: start with `no humans`.
2. **The object**: `machinery, mechanical parts, gears, pipes, cable, screw, circuit board,
   engine, vehicle focus, mecha` - whatever the drawing is of.
3. **Drawing look**: `diagram, lineart, sketch, isometric, concept art`.
4. **Colour and background**: `monochrome, greyscale, limited palette, high contrast,
   blue theme, white background, simple background, grid background, graph paper`.
5. **Quality tags last**: `masterpiece, best quality, newest, absurdres`.

Example: `no humans, machinery, mechanical parts, gears, engine, diagram, lineart, isometric, monochrome, high contrast, white background, simple background, masterpiece, best quality, newest`

### Negative prompt

`worst quality, low quality, 1girl, 1boy, multiple girls, multiple boys, 3d, realistic, photorealistic, gradient, drop shadow, blurry, watermark`

No anatomy tags (`bad anatomy`, `bad hands`): there is no person in the frame.

## family: prose

The active model reads descriptive English prose. Write full sentences that describe the
drawing as a drawing - its projection, line work and sheet - not as a photographed object.
Do not use Danbooru tags - no `no humans`, `masterpiece`, `best quality`, `newest` or
`absurdres`. No non-English text.

### What the prompt describes, in this order

1. **The kind of drawing**: "a technical drawing", "a blueprint", "a patent-style line
   drawing", "an electrical schematic diagram".
2. **The projection**: isometric, orthographic projection, cross-section, cutaway, or an
   exploded view with parts separated along their assembly axis.
3. **The object and its parts**: what it is and which components are visible.
4. **The rendering**: clean vector lines of even weight, CAD line art, flat fills or no
   fills, no shading, high contrast.
5. **The sheet**: plain white background, blueprint paper with a faint grid, a monochrome
   or single-colour palette.

Example: `A patent-style technical drawing of a single-cylinder engine in exploded isometric view, the piston, connecting rod, crankshaft and cylinder head separated along their assembly axis. Clean black vector lines of even weight on a plain white background, CAD line art with no shading, no gradients and no perspective distortion.`

### Negative prompt

Leave the negative prompt empty (`""`). This model family does not use it, and the image
tool discards it. State what the drawing must not have - shading, gradients, photographic
texture - in the prompt itself.

## family: generic

The active model's prompt grammar is not known. Write concise English: the kind of drawing
and the object first, then comma-separated descriptive phrases. Avoid Danbooru-only tags and
long narrative paragraphs alike. No non-English text.

### Prompt pattern

`[kind of drawing] of [object], [projection], [parts shown], [rendering], [sheet and palette]`

Example: `Technical drawing of a centrifugal water pump, cross-section view, impeller and volute casing visible, clean vector lines, CAD line art, monochrome, white background, high contrast`

### Negative prompt

`worst quality, low quality, 3d render, photorealistic, organic shapes, person, face, messy lines, gradients, blurry lines, drop shadow, painterly, watermark`

No anatomy terms (`bad anatomy`, `bad hands`): there is no person in the frame.
