from ultralytics import YOLOE

# Load the PyTorch model
model = YOLOE("yoloe-11l-seg.pt")

# Define your specific prompts that you want to bake into the model
names = ["hand", "white cube", "black cube"]

# Set the classes and get text embeddings BEFORE export
model.set_classes(names, model.get_text_pe(names))

# Export model as .onnx format with specified resolution (must be a multiple of 32)
model.export(format="onnx", imgsz=192)
