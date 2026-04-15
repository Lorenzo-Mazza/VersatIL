"""Key-Value-Query cache layers in transformers.

- **Generation cache** — stores Key/Value for the main sequence being generated. Grows
  incrementally as new tokens are decoded. Used during autoregressive
  generation where each step produces one new token and reuses the previous ones.

- **Conditioning cache** — stores Key/Value (and optionally Query) for static encoding
  context,  that does not change at generation time (e.g. across AR steps or denoising
  steps). Computed once before generation and reused unchanged across all forward
  calls. The conditioning information does not change between steps.

This distinction is mechanism-agnostic: the same conditioning cache works whether the
decoder consumes the conditioning via one-directional or bidirectional attention (both
streams attend to each other). In the bidirectional case, the queries field is
populated so the conditioning stream's projections also participate.
"""
