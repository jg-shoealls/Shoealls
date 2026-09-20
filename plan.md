1. Modify `src/models/reasoning_engine.py`:
   - Find the usage of `self.cross_verify(summaries, summaries, summaries)` which currently returns `cross_out, cross_attn_weights`.
   - Update it to `self.cross_verify(summaries, summaries, summaries, need_weights=False)` and rename `cross_attn_weights` to `_`.
   - Add a comment indicating that setting `need_weights=False` prevents unnecessary computation and memory allocation and allows PyTorch to use FlashAttention if available.

2. Modify `src/models/fusion.py`:
   - Find the usage of `self.self_attention(combined, combined, combined)` which currently returns `attn_out, _`.
   - Update it to `self.self_attention(combined, combined, combined, need_weights=False)`.
   - Add a comment about `need_weights=False` and FlashAttention.
   - Find the usage of `self.cross_attn(query, context, context)` which currently returns `attn_out, _`.
   - Update it to `self.cross_attn(query, context, context, need_weights=False)`.
   - Add a comment about `need_weights=False` and FlashAttention.

3. Complete pre commit steps
   - Complete pre commit steps to ensure proper testing, verification, review, and reflection are done.

4. Submit the change
   - Once all tests pass and changes are verified, I will submit the code with an appropriate descriptive commit message.
