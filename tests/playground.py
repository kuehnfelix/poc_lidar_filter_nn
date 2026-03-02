import time
import numpy as np
import matplotlib.pyplot as plt

from neural_net.cone_detector import ConeDetector

if __name__ == "__main__":
    
    detector = ConeDetector("model_conv.pt")
    for i in range(30, 200):
        frame = i
        path = f"dataset/track_00/frame_{frame:03d}.npz"
        file = np.load(path)
        data = file["data"]  # (600, 125, 3)
        labels = file["labels"]  # (600, 125)
        print(f"Loaded {data.shape[0]} packets from {path}")    
        
        total_time = 0
        for packet, label in zip(data, labels):
            start = time.time()
            mask = detector.predict(packet)  # (125,) bool array
            total_time += time.time() - start
            plt.scatter(packet[:,0], packet[:,1], c=mask, cmap="coolwarm", s=20)
            
            
        plt.title(f"Cone Detection (Total time: {total_time:.2f}s)")
        plt.axis((-1.2, 1.2, -0.5, 0.5))
        plt.show()
