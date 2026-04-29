import shutil
from pathlib import Path
import kagglehub

path = kagglehub.dataset_download("emmarex/plantdisease")
src = Path(path) / "PlantVillage"
dst = Path(__file__).resolve().parent.parent / "data" / "plantvillage_small"

classes = [
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
]

if dst.exists():
    shutil.rmtree(dst)

dst.mkdir(parents=True, exist_ok=True)

for cls in classes:
    shutil.copytree(src / cls, dst / cls)

print("Dataset ready at:", dst)
print("Copied classes:")
for cls in classes:
    print("-", cls)