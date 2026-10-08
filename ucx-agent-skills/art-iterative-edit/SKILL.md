---
name: art-iterative-edit
description: Artist case guidance for changing an image already drawn in the conversation, rewriting the previous prompt for the kind of change asked.
version: 0.1.0
requires_tools:
  - generate_image
tags:
  - case-routed
origin: synthesized
status: active
content_sha256: 86044977ec4f7609687763565cc94951f817e7d74d7be521936960d1855c65c7
approved_by: agent:dev-builder (moved from the Artist's inline texts; not a human approval)
approved_at: '2026-09-28T00:00:00Z'
---

If the latest message changes an image you already made in this conversation, start from your previous prompt. Keep the character's identity tags (hair, eyes, body, signature outfit) unless the person changes them. Decide which kind of change it is, then rewrite the prompt rather than appending a word.
- A new place or scene (카페 장면, 공장에서). Replace the old place, pose, framing and lighting. Build the new place from four or more concrete objects, with a pose, light source and mood that fit it, and remove the old place's tags.
- A time or weather change (밤으로, 비 오게). Change the time tag and make every light and palette tag agree, for example night, dark sky, moonlight or street lamps, cool tones, and remove sunlight and golden hour.
- A style or medium change (수채화 느낌). Add the medium tags, for example watercolor (medium), traditional media, and a fitting palette, and keep the content.
- A request for more detail (디테일 채워, 더 화려하게, 더 자세히). Enrich every part the person left open, with clothing materials and accessories, four or more objects in the place, a precise light source, atmosphere (dust, light rays, steam, rain), depth of field and a palette, 30-40 tags in all. Name actual objects, since generic filler such as detailed background adds nothing.
- One attribute (눈은 초록). Change that tag only.
After the image, say in one sentence what you changed.
