import json, re, collections

ALL_SUBTASKS = [
    "Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence",
    "IQ_Test", "Jigsaw", "Multi-view_Reasoning", "Object_Localization",
    "Relative_Depth", "Relative_Reflectance", "Semantic_Correspondence",
    "Spatial_Relation", "Visual_Correspondence", "Visual_Similarity",
]
DEFAULT_NUM_CHOICES = {
    "Art_Style": 2, "Counting": 4, "Forensic_Detection": 4,
    "Functional_Correspondence": 4, "IQ_Test": 4, "Jigsaw": 2,
    "Multi-view_Reasoning": 2, "Object_Localization": 2,
    "Relative_Depth": 2, "Relative_Reflectance": 3,
    "Semantic_Correspondence": 4, "Spatial_Relation": 2,
    "Visual_Correspondence": 4, "Visual_Similarity": 2,
}

with open('results/infer/llava_1_5_7b/blink_baseline/results_blink_full_baseline.json') as f:
    data = json.load(f)
if isinstance(data, dict) and 'results' in data:
    data = data['results']

missing_fq = sum(1 for r in data if not r.get('full_question', '').strip())
print(f"Samples missing full_question: {missing_fq} / {len(data)}")

mismatch = collections.Counter()
full_question_sample = {}
for r in data:
    sid = r.get('id','')
    subtask = next((st for st in ALL_SUBTASKS if st in sid), 'unknown')
    fq = r.get('full_question', '') or r.get('original_question', '')
    if fq:
        letters = sorted(set(re.findall(r'\(([A-Z])\)', fq)))
    else:
        n = DEFAULT_NUM_CHOICES.get(subtask, 4)
        letters = [chr(ord('A')+i) for i in range(n)]
    expected = DEFAULT_NUM_CHOICES.get(subtask, 4)
    if len(letters) != expected:
        mismatch[f'{subtask}: expected {expected}, got {len(letters)} ({letters})'] += 1
    if subtask not in full_question_sample:
        full_question_sample[subtask] = fq[:200]

print(f"\nChoice count mismatches:")
for k,v in sorted(mismatch.items()):
    print(f"  {v:4d}x  {k}")

print("\nSample full_question per subtask:")
for st in ALL_SUBTASKS:
    fq = full_question_sample.get(st, '[NOT FOUND]')
    print(f"  {st}: {repr(fq[:150])}")