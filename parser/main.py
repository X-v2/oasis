import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser.pipeline import parse_floorplan


if __name__ == "__main__":
    result = parse_floorplan("floorplan3.png", debug=True)
    with open("output.json", "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
    print(json.dumps(result, indent=2))
