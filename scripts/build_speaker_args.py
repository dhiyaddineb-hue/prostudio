#!/usr/bin/env python3
"""Read output/speaker-map.json and emit --speaker-refs / --speaker-map args.

Replaces the fragile inline bash heredoc that broke YAML parsing in dub.yml.
The map file, written by core.py right after rendering, is an array of
{start, end, speaker, ref}; here we turn it into the two CLI flags that
seed_vc_enhance.py accepts so each role keeps its own cloned timbre.
"""
import argparse
import json
import os

REF_FIELD = "ref"
SPK_FIELD = "speaker"
ST_FIELD = "start"
EN_FIELD = "end"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="output/speaker-map.json",
                    help="path to core's speaker-map.json")
    args = ap.parse_args()

    if not os.path.isfile(args.map) or os.path.getsize(args.map) == 0:
        print("NO_SPEAKER_MAP")
        return 0

    try:
        with open(args.map, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        print("NO_SPEAKER_MAP")
        return 0

    refs = {}      # speaker -> reference path
    map_parts = [] # "start-end=speaker"
    if isinstance(data, list):
        for e in data:
            if not isinstance(e, dict):
                continue
            sp = e.get(SPK_FIELD)
            ref = e.get(REF_FIELD)
            if sp and ref:
                refs.setdefault(sp, ref)
            st = e.get(ST_FIELD)
            en = e.get(EN_FIELD)
            if sp is not None and st is not None and en is not None:
                map_parts.append(f"{st}-{en}={sp}")
    else:
        print("NO_SPEAKER_MAP")
        return 0

    if not refs:
        print("NO_SPEAKER_MAP")
        return 0

    # Emit as bash-friendly "KEY=VALUE" chunks on one line each.
    print("REFS=" + ",".join(f"{k}={v}" for k, v in refs.items()))
    print("MAP=" + ",".join(map_parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
