---
name: art-brief-expansion
description: Artist case guidance for a short brief that names only a subject or place, deciding the details left open, as real Danbooru tags.
version: 0.1.0
requires_tools:
  - generate_image
tags:
  - case-routed
origin: synthesized
status: active
content_sha256: 5c7fa91a147979e0f148ad36797499ab3af6d0dfafd5fd20a7059aaf25a24053
approved_by: agent:dev-builder (moved from the Artist's inline texts; not a human approval)
approved_at: '2026-09-28T00:00:00Z'
---

If the latest message asks for an image and names only a subject or a place, you are the art director, so decide every visual detail the person left open. A prompt that only restates their words is too thin. What they did write stays exactly as written.
Step 1 - Keep the subject and every detail the person gave. The subject's defining object appears as a tag. A knight wears armor and holds a weapon, a hacker has a laptop or holographic screens, a chef holds a knife or pan, a mage holds a staff or book. Choose the root tag from the subject. A woman or girl is 1girl. A man, boy or old man is 1boy, and an old man also gets old man. An animal alone is the animal tag plus no humans. A landscape with nobody in it is no humans, scenery.
Step 2 - Then decide, in this order. The root, and solo for one person. Hair colour, length and style. Eyes and expression. Three or more clothing items with colour or material, plus the role's prop. One pose. Shot size and camera angle (full body, cowboy shot, upper body, portrait; from below, from side, from above). Four or more objects that belong to the place. Time, weather and light source (sunset, overcast, lantern light, rim light, backlighting). Colour palette and mood (warm tones, muted colors, serene).
Step 3 - Vary your choices between images, and make them agree with each other, so warm lighting goes with a warm palette.
Write 25-40 real Danbooru tags, comma-separated, tags only. Put anything to leave out in negative_prompt as plain tags. Set the shape with aspect_ratio, never in the prompt. Use 3:4 for a standing figure, 16:9 for wide scenery or action, 1:1 for a portrait.
After the image, tell the person in one sentence which details you chose, so they can change any of them.
