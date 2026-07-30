import json
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file) as fin, open(output_file, "w") as fout:
    for line in fin:
        obj = json.loads(line)
        items = json.loads(obj["generated_text"])

        filtered = [
            x for x in items
            if x["category"] not in ("Page-header", "Page-footer")
        ]

        obj["generated_text"] = json.dumps(filtered, ensure_ascii=False)
        fout.write(json.dumps(obj) + "\n")
