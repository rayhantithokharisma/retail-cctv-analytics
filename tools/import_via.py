import argparse
import csv
import json
from pathlib import Path
import yaml


def parse_via_csv(path: str) -> list[dict]:
    """Returns [{name, shape_type, points: [(x,y),...]}] — polygon/polyline points as-is,
    rect converted to its 4 corners in the same convention."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            shape_attr_str = row.get("region_shape_attributes", "{}")
            region_attr_str = row.get("region_attributes", "{}")
            if not shape_attr_str or shape_attr_str == "{}":
                continue

            shape_attr = json.loads(shape_attr_str)
            region_attr = json.loads(region_attr_str) if region_attr_str else {}

            name = region_attr.get("name", "unknown")
            shape_name = shape_attr.get("name", "")

            if shape_name in ("polygon", "polyline"):
                xs = shape_attr.get("all_points_x", [])
                ys = shape_attr.get("all_points_y", [])
                points = [(float(x), float(y)) for x, y in zip(xs, ys)]
            elif shape_name == "rect":
                x = float(shape_attr.get("x", 0))
                y = float(shape_attr.get("y", 0))
                w = float(shape_attr.get("width", 0))
                h = float(shape_attr.get("height", 0))
                points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            elif shape_name == "point":
                cx = float(shape_attr.get("cx", 0))
                cy = float(shape_attr.get("cy", 0))
                points = [(cx, cy)]
            else:
                points = []

            results.append({
                "name": name,
                "shape_type": shape_name,
                "points": points
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="Import VIA CSV annotations to YAML fragment")
    parser.add_argument("csv_path", type=str, help="Path to VIA CSV file")
    parser.add_argument("--out", type=str, default=None, help="Output YAML file path")
    args = parser.parse_args()

    parsed = parse_via_csv(args.csv_path)
    output_data = {}
    for item in parsed:
        output_data[item["name"]] = {
            "shape_type": item["shape_type"],
            "points": item["points"]
        }

    yaml_str = yaml.dump(output_data, sort_keys=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(yaml_str)
        print(f"Written to {args.out}")
    else:
        print(yaml_str)


if __name__ == "__main__":
    main()
