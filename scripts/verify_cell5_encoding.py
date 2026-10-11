"""Confirm the Cell 5 code block in KAGGLE_CELLS.md is byte-correct.

The PowerShell console renders the Igbo dotted-i and the accented character
class incorrectly, so we must verify the file itself rather than the terminal
output - the user pastes this text into Kaggle.
"""
import re

md = open("KAGGLE_CELLS.md", encoding="utf-8").read()
blk = md.split("## Cell 5")[1].split("## Cell 6")[0]
src = re.findall(r"```python\n(.*?)```", blk, re.S)[0]

print("Igbo marker U+1EC5 present :", "ị" in src)
print("accented char class intact  :", "àáèéìíòóùúṣọ" in src)
print()
for line in src.splitlines():
    s = line.strip()
    if s.startswith('"Igbo"') or "findall" in s:
        print("  ", repr(s))
print()
# Show codepoints for the two lines that matter, so corruption is visible.
for line in src.splitlines():
    if line.strip().startswith('"Igbo"'):
        print("Igbo marker codepoints:",
              [hex(ord(c)) for c in line.split('"')[3]][:12])
    if "findall" in line:
        i = line.index("r\"")
        print("regex codepoints       :",
              [hex(ord(c)) for c in line[i + 2:i + 24]])