"""Decoder factory modules for action prediction architectures.

Positional encoding contract:
    All decoder factories follow a unified PE pattern:
    1. ``TransformerInputBuilder`` computes additive PE (spatial 2D sinusoidal,
       temporal learned, flat 1D sinusoidal) and returns
       ``(input_tokens, pos_encodings, padding_mask)``.
    2. Factories always pre-add PE to observation tokens before the transformer:
       ``obs_tokens = obs_tokens + pos_encodings``. This ensures cross-attention
       keys carry absolute position information.
    3. ``positional_encoding_type`` on the transformer controls self-attention PE:
       - ``None``: no internal PE (additive-only from step 2).
       - ``rope``: RoPE applied to Q/K in self-attention layers, on top of the
         pre-added additive PE.
        - ''sinusoidal'' or ''learned'': an extra absolute PE is added inside the transformer.
    4. Cross-attention never applies RoPE. Keys get position info solely from
       the pre-added additive PE. This avoids position-space collisions between
       query and key sequences from different modalities.

When implementing a new decoder factory, follow this contract:
    - Always call ``self.input_sequence_builder(features)`` and pre-add the
      returned ``pos_encodings`` to the tokens.
    - Pass ``positional_encoding_type=self.positional_encoding_type`` to the
      transformer constructor so the user can configure self-attention PE.
"""
