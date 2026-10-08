---
name: art-medium-restraint
description: Artist case guidance for a request in a named medium, or for something simple or left out, with the medium tags first and nothing extra.
version: 0.1.0
requires_tools:
  - generate_image
tags:
  - case-routed
origin: synthesized
status: active
content_sha256: 44fa39cc9c576be6e4c6f0997a6fe68e9c88c1e47efc06f6b5c88b71cc249e4b
approved_by: agent:dev-builder (moved from the Artist's inline texts; not a human approval)
approved_at: '2026-09-28T00:00:00Z'
---

If the latest message asks for an image in a named medium, or asks for something simple or for something to be left out, respect that before anything else.
Use these medium tags as written here.
흑백, 모노크롬 = monochrome, greyscale
잉크 = ink (medium), traditional media
스케치 = sketch; 연필 = sketch, graphite (medium)
수채화 = watercolor (medium), traditional media
유화 = oil painting (medium)
선화 = lineart
When the person asks for something simple, an empty background or one thing only (단순하게, 배경 없이, 하나만), use simple background or white background and add nothing beyond what they named. Extra people, props, lighting effects, colours and mood tags stay out.
A monochrome or sketch request also leaves out colour names, glow and cinematic lighting.
What is left out goes into negative_prompt as plain tags, for example color, extra people, detailed background. The prompt itself lists only what is drawn.
The root still follows the subject. 노인 어부 is 1boy, old man; 고양이 한 마리 is cat, no humans. Use 1girl only for a female subject.
Keep the prompt short, 8-15 tags, in this order. The root, the details the person gave, the action, the medium tags, the background tag.
