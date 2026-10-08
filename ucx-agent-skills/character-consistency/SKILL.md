---
name: character-consistency
description: Character Visual DNA design, persistent character sheet management, and
  multi-character prompt composition.
version: 0.1.0
requires_tools:
  - character_sheet
  - generate_image
origin: synthesized
status: active
content_sha256: b5bc1c64072ec8f065e5efc065b5fec11117d1da0caf109e11e218168eb4f4e1
approved_by: kennylim (owner approval, 2026-09-28)
approved_at: '2026-09-28T00:00:00Z'
---

# Character Consistency & Visual DNA Management

This skill governs the systematic creation, storage, and retrieval of visual characters to ensure rock-solid aesthetic continuity across multiple illustrations and storyboard scenes.

## 1. Defining Character Visual DNA

When establishing a new character, define a deterministic **Visual DNA** consisting of:
1. **Core Anchors (Immutable)**:
   - Specific hair color and hairstyle (e.g. `silver hair, low twin braids, blunt bangs`)
   - Eye color and distinctive gaze (e.g. `amber eyes, sharp gaze`)
   - Distinguishing traits (e.g. `elf ears, tear mole under left eye, scar on cheek`)
2. **Attire & Wardrobe (Contextual)**:
   - Signature armor, dress, or uniform (e.g. `ornate silver plate armor, white tabard, gold trim, navy blue cape`)
   - Key accessories (e.g. `ruby pendant, fingerless leather gloves`)
3. **Prompt Grammar Encoding**:
   - Translate all visual elements into concise, comma-separated Danbooru tags for Illustrious-XL / Anime SDXL models.
   - For prose models (FLUX), translate into high-density descriptive English sentences.

## 2. Preventing Multi-Character Attribute Bleeding

In diffusion models (especially Danbooru SDXL), multiple characters sharing a prompt often suffer from **attribute bleeding** (e.g., hair color of character A transferring to character B).

To minimize bleeding:
1. **Accurate Root Counting**:
   - Never use `solo` when multiple characters appear.
   - Use explicit root tags: `2girls`, `2boys`, `1boy 1girl`, `3girls`, etc.
2. **Character Order & Separation**:
   - Group each character's traits sequentially without interleaving attributes.
   - Use contrastive features (e.g., distinctly different hairstyles, colors, and clothing silhouettes).
   - Fuse each hair colour into a hairstyle tag (`very long black hair, black hime cut`, `short blonde bob, blonde bob cut`) instead of separate colour and length tags.
   - For one girl and one boy, write the girl's traits first. See `media-character` for the measured effect of each rule.
   - Use `character_sheet(action='compose', character_ids=[...])` which formats the prompt and applies segregation rules automatically.
3. **Aspect Ratio Adaptation**:
   - For 2 or more characters, always prefer widescreen framing (`16:9` or `4:3`) to provide sufficient spatial canvas and prevent crowded anatomical overlap.

## 3. Sequential Scene & Expression Consistency

To depict the same character in different poses, emotions, or environments:
1. **Hold Visual DNA Constant**: Keep the hair, eyes, and core outfit tags identical.
2. **Vary Dynamics**:
   - **Expressions**: `smile`, `serious`, `surprised`, `looking at viewer`, `smug`, `blushing`
   - **Poses**: `standing`, `sitting`, `running`, `battle stance`, `holding sword`
   - **Backgrounds**: `ancient ruins, moonlight`, `lively tavern, warm lighting`, `blossom garden, sunlight`
3. **Seed Control**:
   - Retrieve the character's `base_seed` via `character_sheet(action='get', character_id=...)`.
   - Pass `seed_override` close to the `base_seed` (or fix it) when subtle variations in lighting/expression are needed, or leave unpinned when drastic camera angle changes are desired.

## 4. End-to-End Workflow with `character_sheet`

1. **New Character Request**:
   - Formulate tags $\rightarrow$ Call `character_sheet(action='save', character_id='elena', danbooru_tags='...', base_seed=12345)` $\rightarrow$ Call `generate_image(prompt='...', style='anime')`.
2. **Recall Existing Character**:
   - Call `character_sheet(action='get', character_id='elena')` $\rightarrow$ Inject retrieved tags into new prompt $\rightarrow$ Call `generate_image(...)`.
3. **Multi-Character Encounter**:
   - Call `character_sheet(action='compose', character_ids=['elena', 'kaito'], scene_context='conversing in tavern')` $\rightarrow$ Execute `generate_image` with the composed prompt and recommended `16:9` ratio.

## 5. Multi-Pose Storyboard Workflow (Pre-Generation Prompt Planning)

When the user asks for multiple images with different poses, scenes, or camera angles (e.g. "다양한 포즈 5장", "여러 각도에서 4장", "10 different poses"):
1. **Never Cram Multiple Poses into One Prompt**:
   - Do NOT pass `count=N` with a single generic prompt or list multiple poses in one prompt (e.g. writing `"dynamic poses including standing, seated, reclining..."` will produce severe anatomical deformities in diffusion models).
2. **First Plan N Distinct Prompts**:
   - FIRST decompose the request into $N$ distinct, concrete prompts before invoking the tool.
   - For every prompt $i \in [1..N]$:
     `[Character Visual DNA tags] + [Exactly ONE physical posture and camera angle] + [Scene Context/Lighting] + [Quality Tags]`
   - Example 5-pose set:
     - Prompt 1: `... standing gracefully, hands on hips, eye-level full-body shot ...`
     - Prompt 2: `... seated on modern chair, crossed legs, three-quarter medium shot ...`
     - Prompt 3: `... reclining on velvet sofa, relaxed posture, side angle shot ...`
     - Prompt 4: `... stretching arms overhead, dynamic arched back, full-body shot ...`
     - Prompt 5: `... leaning against neon wall, looking back over shoulder, close-up shot ...`
3. **Execute via `prompts` Parameter**:
   - Call `generate_image(prompts=[prompt_1, prompt_2, ...])` in a single tool invocation.
