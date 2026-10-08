---
name: art-literal-spec
description: Artist case guidance for a request that already describes the picture in detail, translating every phrase into tags and inventing nothing.
version: 0.1.0
requires_tools:
  - generate_image
tags:
  - case-routed
origin: synthesized
status: active
content_sha256: f6b175ae66adae88732307ef5443cf53af6be29d07b01b5320f9a29b345dfb95
approved_by: agent:dev-builder (moved from the Artist's inline texts; not a human approval)
approved_at: '2026-09-28T00:00:00Z'
---

If the latest message asks for an image and already describes it in detail, translate it exactly and invent nothing.
Step 1 - Turn every phrase the person wrote into a tag and keep all of them. Colour words are exact.
은발 = silver hair, 백발 = white hair, 금발 = blonde hair, 흑발 = black hair, 갈색 머리 = brown hair, 빨간 머리 = red hair, 분홍 머리 = pink hair
단발 = short hair, bob cut; 장발 = long hair; 트윈테일 = twintails; 포니테일 = ponytail
붉은 눈 = red eyes, 파란 눈 = blue eyes, 초록 눈 = green eyes, 금색 눈 = yellow eyes
Never swap a colour for a nearby one. Silver is not blonde.
Step 2 - Choose the root from the subject. 소녀, 여자 and 여인 are 1girl. 소년, 남자, 아저씨, 노인 and other men are 1boy, and 노인 adds old man, 수염 adds beard. Never write 1girl for a man.
Step 3 - Add only what was left open and cannot conflict, such as shot size and camera angle, and one light tag if the person gave none. Clothing, props, people, colours and background objects stay exactly as the person wrote them, with nothing added. A plain or white background stays plain.
Step 4 - Order the tags as root, character, clothing, pose, framing, place, light.
Real Danbooru tags only, comma-separated. Set the shape with aspect_ratio, never in the prompt.
Leave negative_prompt empty unless the person asked for something to be left out; the tool then uses its own short default.
