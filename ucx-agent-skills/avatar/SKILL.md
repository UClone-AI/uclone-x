---
name: avatar
description: "Make or change your own profile picture - draw candidates to choose\
  \ from, or draw one and set it - when the user talks about your avatar, profile\
  \ picture or face (\uC544\uBC14\uD0C0, \uD504\uB85C\uD544 \uC0AC\uC9C4, \uD504\uC0AC\
  )."
version: 0.2.0
requires_tools:
  - set_avatar
origin: synthesized
status: active
content_sha256: d2ebe7f28075489746cb7bddb98f3427a4cbcf52b8299f08db74661d726ced6b
approved_by: agent:dev-builder (not a human approval). Style presets approved by the owner on
  2026-09-27, as relayed to the builder; this line is agent-written
approved_at: '2026-09-28T03:07:57Z'
---

# Your Own Profile Picture

Use this skill when the user asks about your picture: "make me some avatars", "change your
profile picture", "아바타 몇 개 만들어서 보여줘", "프로필 사진 바꿔줘", "프사 새로 그려줘".
It is about **your own** picture only. `set_avatar` changes only yours; if the user wants
another clone's picture changed, draw candidates if you can and tell them to apply one with
**Use as avatar** under the picture, choosing that clone.

Always answer in the language the user wrote in.

## 1. Decide the flow

- **Choose** (the default): the user wants to see options — "make some", "show me a few",
  "몇 개 만들어 보여줘", or no count at all. Draw **4** candidates unless they gave a number.
- **Direct**: the user wants it done — "just make one and set it", "바로 바꿔", "하나 만들어서
  바꿔줘". Draw **1** and set it.

## 2. Describe yourself

Build the subject from your own role and description, plus anything the user asked for
(style, colours, mood, expression). The picture is:

- square, `aspect_ratio: "1:1"`;
- a portrait of one subject's face, facing the viewer;
- on a simple, plain background, so it reads in a small round frame.

These are defaults. **Whatever the user asks for wins over them** — framing, background,
prop, style, anything. When they ask for "full body" or "with a street behind me", draw
that, and drop the default words it contradicts.

Keep it an ordinary, friendly portrait suitable for a profile picture. Before writing the
prompt, load the `media-character` skill and follow its rules for the active image model.

## 3. Choose the style

**A style the user names always wins.** When they name one — a preset below by name
("watercolor", "수채화", "pixel art") or their own words ("like a 90s anime", "oil
painting") — use their style words in place of the default style words. Keep the other
defaults from step 2 that they did not change.

**With no style named, use the default house style, "pastel close-up".** Its template:

`Close-up face portrait of cute chibi <who you are>, wearing/with <one bold silhouette prop that says your role>, simple solid pastel <colour> background, minimalist 3D cute illustration, <expression> smile, bold silhouette, headshot only, no body, no border`

For example, an artist might be "artist girl wearing a big pastel pink beret hat" on a
pastel pink background with a cheerful smile; a guardian "guardian wearing large round
dark-rimmed glasses" on pastel butter yellow with a thoughtful, calm smile.

The template is the default, not a rule: whatever the user asks for replaces the part it
contradicts, and the rest stays. With "full body" and no style, keep the pastel style words
and drop "Close-up face portrait", "headshot only" and "no body" for a full-body
description. A background, prop or expression they name replaces that slot the same way.

Named presets, which the app's style chips refer to by name. Each line is the style
fragment that replaces the default style words; keep your subject and every other default
from step 2 that the user did not change:

- **pastel close-up** (the default): the template above.
- **watercolor**: `soft watercolor painting, gentle washes of colour, visible paper texture, pale plain background`
- **realistic portrait**: `realistic photographic portrait, natural soft studio light, shallow depth of field, plain neutral background`
- **pixel art**: `pixel art portrait, crisp 32-bit style pixels, limited palette, flat plain background`
- **flat vector-look illustration** (also "flat illustration"): `flat vector-style illustration, clean shapes, bold flat colours, no gradients, plain background` — still a raster picture from `generate_image`; never write SVG or draw a picture in code.

Write the final prompt in the grammar `media-character` gives for the active model; for a
tag-based model, turn these words into its tags.

Where the default comes from: the template follows the built-in clone avatars that ship
with the app, which were drawn with the Google Gemini image model current at the time (about
2026-09-22). The exact model is not recorded, and this template is derived from those
prompts rather than a copy of them, so the same words on another model will not give the
same pictures.

## 4. Draw

Make **one** `generate_image` call:

- **Choose**: `prompts: [...]` with one prompt per candidate. All candidates use the same
  style — the one the user named, otherwise the default — and vary within it: with the
  default, change the prop, the background colour and the expression; with a named style,
  change the details that style allows (colours, expression, prop).
- **Direct**: `prompt: "..."` with `count: 1`.

## 5. Present or set

- **Choose**: show every image the call returned, numbered, and say: "Pick one with
  **Use as avatar** under the picture, or tell me which number." When the user names a
  number ("2번으로"), call `set_avatar(image_path=<that image's path>)`.
- **Direct**: call `set_avatar(image_path=<the path generate_image returned>)`, show the
  new picture, and offer to undo it ("되돌리려면 말해줘").

To undo: call `set_avatar(image_path=<previous_path>, undo_of=<change_id>)` with the
`previous_path` and `change_id` the last `set_avatar` returned. When `previous_path` was
null, call `set_avatar(reset=true, undo_of=<change_id>)`, which goes back to your shipped
picture. If the undo is refused because the picture was changed again since, say so plainly
and do not retry it; ask which picture the user wants instead.

## 6. When you cannot draw

If you do not have `generate_image`:

- If you have `a2a_call` and Artist is one of your peers, ask Artist for the candidates
  (your description, the style from step 3, and whatever the user asked for, with the
  step 2 defaults for anything they did not: square, face portrait, simple background),
  then continue from step 5 with
  the image paths Artist returns.
- Otherwise say plainly that you can't draw pictures yourself, and suggest asking Artist in
  a conversation with it. The user can then apply Artist's picture to you with
  **Use as avatar** under the picture.

Never call `set_avatar` with a path no tool gave you in this conversation. A path you make
up names no picture, and nothing is set.

## 7. When drawing fails

If `generate_image` fails, nothing has been set; say so. When it says no image model is
connected (reason `no_image_engine`), tell the user that in plain words — for example
"이미지 모델이 연결되어 있지 않아 그릴 수 없어요." — and offer the two remedies:

1. connect an image model in **Settings › Images**;
2. upload a picture from your profile menu instead.

Give no command-line instructions, and do not repeat error details, engine names or
settings names beyond those two remedies. For any other failure, say which part failed in
plain words; never set a placeholder picture.
