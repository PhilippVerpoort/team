---
title: Examples
---

# Examples

Here is a list of examples of techno-economic assessments made with TEAM. The examples can be downloaded and executed as Jupyter notebooks.

```python exec="true" showcode="false"
from pathlib import Path

for file in (Path(".") / "docs" / "examples").glob("*.ipynb"):
    print(f"* [{file.stem}]({file.stem})")
```
