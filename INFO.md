Classification Model/AMM Training:
Input:
    -config file, .json
    -image directory, .png
    -mask directory, .png

corresonding masks and images should have the same name

Output:
    -model.pth: Trained FCResNet model weights
    -loc.npy: Class mean embeddings
    -cov.npy: Class covariance matrices
    -meta_data, .json

Segmentation Model/ M2F Training:
Input:
   




TODO:
- analyze differences in metadata.json: different data normalization, different data sources, different contrast calculation
