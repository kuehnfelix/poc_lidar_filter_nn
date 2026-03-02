import time
import numpy as np
import matplotlib.pyplot as plt

from neural_net.inference import ConeDetector

if __name__ == "__main__":
    
   
    detector = ConeDetector('checkpoints/wide_conv_5ch.pt', device='cuda')
    for i in range(200):
        frame = i
        path = f"dataset/track_01/frame_{frame:03d}.npz"
        file = np.load(path)
        data = file["data"]  # (600, 125, 3)
        labels = file["labels"]  # (600, 125)
    
        
        if np.sum(labels) < 30:
            print(f"Skipping frame {frame} (only {np.sum(labels)} cones)")
            #continue
        print(f"Loaded {data.shape[0]} packets from {path}")
        
        
        plt.subplot(1, 2, 2)
        plt.axis((-1.2, 1.2, -0.5, 0.5))
        total_time = 0
        start = time.time()
        mask = detector.predict_batch(data)  # (600, 125) bool array
        total_time += time.time() - start
        plt.scatter(data[:, :, 0].flatten(), data[:, :, 1].flatten(), c=mask.flatten(), cmap="coolwarm", s=20)
    
        plt.title(f"Cone Detection (Total time: {total_time:.2f}s)")
            
        plt.subplot(1, 2, 1)
        plt.title("Ground Truth")
        plt.scatter(data[:,  :, 0].flatten(), data[:, :, 1].flatten(), c=labels.flatten(), cmap="coolwarm", s=20)
        plt.axis((-1.2, 1.2, -0.5, 0.5))

        

        plt.show()