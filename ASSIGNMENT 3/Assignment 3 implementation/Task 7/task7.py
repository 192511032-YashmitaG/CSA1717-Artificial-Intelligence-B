#!/usr/bin/env python3
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import task7
print(json.dumps(task7(), indent=2, default=str))
