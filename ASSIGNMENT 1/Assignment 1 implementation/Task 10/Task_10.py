from pathlib import Path
p=Path(__file__).resolve().parents[1]/'Task_09'/'performance_results.txt'
print('Performance file exists:',p.exists())
if p.exists():
    for line in p.read_text().splitlines()[:8]: print(line)
