# Object-s-Color-Shape-Recognition-System-for-Pick-and-Place-Robot
- Utilized YOLOE model for real-time image processing and recognition from the camera. Classified object colors and shapes, then output results via GPIO.
- Transferred processed data to the PLC via GPIO to control pick-and-place actions at predefined positions for each object (triggering gripper signals and position control).
- Optimized performance by selecting an appropriate model size, exporting the model to ONNX, and fine-tuning input resolution, achieving ~12–20 FPS on Raspberry Pi 5 while maintaining stable accuracy in real-world environments.
Technologies: Python, Raspberry Pi 5, Raspberry Pi Camera Module V3, GPIO.
