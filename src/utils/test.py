from src.utils.refine_dataset import RefineDataset

dr = RefineDataset()

#dr.refine()
videos, frames, faces, valences = dr.load_data()

print(len(videos))
print(len(frames))
print(len(faces))
print(len(valences))