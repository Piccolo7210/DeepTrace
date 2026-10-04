NSG detector - local inference
1. put 256x256_diffusion_uncond.pt and nsg_stats_gbm.joblib (+ parity_samples.npz) into models/
2. pip install torch --index-url https://download.pytorch.org/whl/cpu ; pip install -r requirements.txt
3. python detect.py --parity models/parity_samples.npz
4. python detect.py my_video.mp4
