"""Section 4 of the paper: the two-level account in Qwen3.8-27B (BF16), template-native renderings only.

Modules (each runnable with python -m selfindex.twolevel.<name>; run labels as recorded in identity.json):
  default_geometry       L  message-content geometry: the role message in the template's default rendering, 150 bodies x 3
                            roles, Q/K/V, FFN and residual at every decoder block, full-vocabulary readout at the body positions
  native_geometry        J  response-onset geometry and self-reference: residual stream at named positions of native
                            conversations, decision-point readout at every layer, option-letter probabilities
  thinking_attribution   N  self-reference with and without thinking (the two official template modes)
  reasoning_ownership    M  reasoning over own records: the same first step as an Assistant or a User message
  role_word_embeddings      the three role words in the input embedding and output unembedding (Qwen3.8-27B)
  role_word_embeddings_gptoss  the same for gpt-oss-20b (token ids from the tokenizer, R_A, R_U and R_S)
base holds what they share: output writers, the 150 bodies in run order, the attribution question, token positions, the
feature collector and the full-vocabulary readout.
"""
