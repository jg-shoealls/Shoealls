import re

with open("src/data/weargait_dataset.py", "r") as f:
    content = f.read()

content = content.replace("def __init__(self, imu_windows, pressure_windows, labels):", "def __init__(self, imu_windows, pressure_windows, labels, biomarkers=None):\n        self.biomarkers = biomarkers")

content = content.replace('            "label": torch.tensor(self.labels[idx], dtype=torch.long),\n        }', '            "label": torch.tensor(self.labels[idx], dtype=torch.long),\n        }\n        if self.biomarkers is not None:\n            d["biomarkers"] = torch.as_tensor(self.biomarkers[idx], dtype=torch.float32)\n        return d')

content = content.replace('        return {\n', '        d = {\n')

with open("src/data/weargait_dataset.py", "w") as f:
    f.write(content)
