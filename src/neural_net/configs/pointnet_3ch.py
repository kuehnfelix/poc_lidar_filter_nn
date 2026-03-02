config = {
    "model":             "pointnet",
    "in_channels":       3,
    "features":          ["az", "el", "dist"],

    "pos_weight":        42.0,
    "lr":                1e-3,
    "epochs":            60,
    "batch_size":        64,
    "oversample_factor": 10,
    "threshold":         0.3,
    "val_split":         0.1,

    "save_path":         "checkpoints/pointnet_3ch.pt",
}
