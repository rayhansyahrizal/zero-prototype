# Caption Decoding Bug Fix

## 🐛 Problem Identified

### Symptom
- **Baseline method** (BLEU-4: 0.0113) outperformed **Prototype** (0.0061-0.0071) and **Retrieval** (0.0067-0.0070)
- This was unexpected because retrieval-augmented generation should improve results

### Root Cause
The `batch_decode()` function in [src/generation.py](src/generation.py) was decoding **ALL tokens** (prompt + generated text), instead of only the **newly generated tokens**.

**Example of the bug:**
```json
{
  "method": "retrieval",
  "caption": "Based on the context of similar medical images:\n- CT angiography of abdomen...\n- CT angiogram of the chest..."
}
```

The `caption` field contains the **full prompt**, not the actual generated caption!

### Why This Happened

When using BLIP2 with a text prompt:
1. Input: `[prompt tokens] + [image features]`
2. Model generates: `[prompt tokens] + [NEW generated tokens]`
3. Old code: `batch_decode(generated_ids)` → Returns **full sequence** (prompt + generation)
4. **Bug**: Saved the prompt as the "caption"!

### Why Baseline Was Better

- **Baseline**: No prompt → `batch_decode()` correctly returns generated caption ✅
- **Retrieval/Prototype**: Has prompt → `batch_decode()` returns prompt + caption → Mostly just prompt ❌
- **BLEU-4 Evaluation**: Compares corrupted text against ground truth → Very low score!

---

## ✅ Fix Applied

### File: `src/generation.py` (Lines 196-212)

**Before (Buggy):**
```python
# Decode caption
caption = self.processor.batch_decode(
    generated_ids, skip_special_tokens=True
)[0].strip()

return caption
```

**After (Fixed):**
```python
# Decode caption
# IMPORTANT: Only decode the newly generated tokens, not the input prompt
# generated_ids contains [prompt tokens + new tokens], we only want new tokens
if retrieved_captions or context:
    # When using context, skip the prompt tokens
    prompt_length = inputs.input_ids.shape[1]
    generated_tokens_only = generated_ids[:, prompt_length:]
    caption = self.processor.batch_decode(
        generated_tokens_only, skip_special_tokens=True
    )[0].strip()
else:
    # Baseline mode (no prompt), decode everything
    caption = self.processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

return caption
```

### How It Works

1. **With context** (retrieval/prototype):
   - Get prompt length: `prompt_length = inputs.input_ids.shape[1]`
   - Extract only new tokens: `generated_tokens_only = generated_ids[:, prompt_length:]`
   - Decode only the new tokens: `batch_decode(generated_tokens_only)`

2. **Without context** (baseline):
   - Decode everything as before (no prompt to skip)

---

## 🔍 Impact on Other Files

### ✅ `src/main.py`
- **No changes needed**
- This file orchestrates the pipeline but doesn't do decoding itself
- Uses `CaptionGenerator` from `generation.py` which now has the fix

### ✅ `ui/app.py`
- **No changes needed**
- Calls `state['caption_gen'].generate_caption()` which uses the fixed code
- All UI caption generation will automatically use the fix

---

## 📊 Expected Results After Fix

### Before Fix:
```
BLEU-4 Scores:
  baseline:   0.0113 ✅ (working correctly)
  prototype:  0.0067 ❌ (broken - returns prompts)
  retrieval:  0.0069 ❌ (broken - returns prompts)
```

### After Fix (Expected):
```
BLEU-4 Scores:
  baseline:   0.0113 (unchanged)
  prototype:  0.01XX (should improve significantly!)
  retrieval:  0.01XX (should improve significantly!)
```

**Why improvement expected:**
- Now actually comparing **generated captions** vs ground truth
- Retrieved context should help generate better, more specific captions
- METEOR scores might show even more improvement (was 0.155-0.161 before, might go higher)

---

## 🧪 Testing the Fix

### Manual Test
Run the pipeline with any config:
```bash
python src/main.py --config config_prototype_no_tta.yaml
```

Then inspect `results/no_tta/captions.json`:
```python
import json
with open('results/no_tta/captions.json') as f:
    data = json.load(f)

# Check retrieval caption
retrieval_caption = data['retrieval'][0]['caption']
print(retrieval_caption)

# Should be a SHORT caption (e.g., "CT scan showing pneumonia")
# NOT a long prompt starting with "Based on the context..."
```

### Automated Test
A test script is provided: `test_caption_fix.py`

Run it with:
```bash
python test_caption_fix.py
```

This will verify that:
1. Baseline captions are generated correctly
2. Retrieval captions DON'T start with "Based on the context..."
3. Caption lengths are reasonable (<100 chars, not 300+)

---

## 📝 Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Baseline** | ✅ Working | ✅ Working |
| **Retrieval** | ❌ Returns prompts | ✅ Returns captions |
| **Prototype** | ❌ Returns prompts | ✅ Returns captions |
| **BLEU-4** | Baseline wins (0.0113) | All methods should improve |
| **Root cause** | `batch_decode()` includes prompt | Fixed: only decode new tokens |

---

## 🚀 Next Steps

1. **Re-run experiments** with the fix:
   ```bash
   bash scripts/run_experiments.sh
   ```

2. **Compare new results**:
   ```bash
   python scripts/compare_experiments.py
   ```

3. **Expected outcome**:
   - Prototype and Retrieval BLEU-4 scores should **significantly improve**
   - May now **exceed or match baseline** performance
   - METEOR scores should also improve

4. **If prototype still underperforms**:
   - Then it's a real algorithmic issue (not a bug)
   - Investigate: TTA settings, prototype quality, retrieval relevance
   - But at least we'll be comparing apples-to-apples!

---

**Fixed on**: 2025-11-13
**Files modified**: `src/generation.py`
**Files verified**: `src/main.py`, `ui/app.py`
