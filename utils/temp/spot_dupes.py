import json
from pybloom_live import BloomFilter

# Expect ~30M lines
bf = BloomFilter(capacity=30_000_000, error_rate=0.001)

dupes = 0
total = 0

with open("/fsxnew/shyam.pawar/OCR_stuff/02_OCRd/Archive_Multilingual/malayalam.jsonl", "r", buffering=1024*1024) as f:
    for line in f:
        total += 1
        obj = json.loads(line)
        _id = obj["id"]

        if _id in bf:
            dupes += 1
        else:
            bf.add(_id)

print(f"Total lines processed: {total}")
print(f"Approx duplicate count: {dupes}")

# import json
# from collections import defaultdict

# MAX_SAMPLES = 5  # stop after 5 duplicates

# seen = {}                # store first occurrence of each id
# samples = defaultdict(list)  # store duplicates

# with open("/fsxnew/shyam.pawar/OCR_stuff/02_OCRd/Archive_Multilingual/malayalam.jsonl", "r", buffering=1024*1024) as f:
#     for line in f:
#         obj = json.loads(line)
#         _id = obj["id"]

#         if _id in seen:
#             samples[_id].append(obj)
#             if len(samples) >= MAX_SAMPLES:
#                 break
#         else:
#             seen[_id] = obj

# # Print the 5 sample duplicates
# for i, (dup_id, objs) in enumerate(samples.items(), 1):
#     print(f"\n=== Duplicate {i}: id = {dup_id} ===")
#     print("First occurrence:")
#     print(seen[dup_id])
#     print("Duplicate occurrence(s):")
#     for o in objs:
#         print(o)
