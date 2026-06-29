=============================================================================
Project: AI vs. Real Image Classifier using Vision Transformer (ViT)
=============================================================================

Description:
------------
This project fine-tunes a pre-trained Vision Transformer (ViT-B/16) to classify images as either "AI-generated" or "Real". The pipeline is optimized for constrained GPU environments (like a 16GB Nvidia T4) and implements advanced training techniques including:
- Differential Learning Rates (Layer-Wise Learning Rate Decay)
- Automated Mixed Precision (AMP)
- Test-Time Augmentation (TTA)
- Gradient Accumulation

Hardware Requirements:
----------------------
- GPU: Nvidia T4 (16GB VRAM) or equivalent.
- CPU RAM: Minimum 16GB (30GB recommended for large datasets).
- Storage: Sufficient space for the 50k+ image dataset and model checkpoints.

Installation:
-------------
1. Ensure Python 3.8+ is installed.
2. Install the required dependencies using pip:
   pip install -r requirements.txt

Dataset Structure:
------------------
The code expects a directory structure combined into a target folder named 'data_combined' with the following splits:

data_combined/
├── train/
│   ├── ai/
│   └── real/
├── val/
│   ├── ai/
│   └── real/
└── test/
    ├── ai/
    └── real/

Usage:
------
1. Data Preparation: 
   Run the data preparation cells to download, extract, and format the dataset into the structure above.
2. Training: 
   Execute the training loop. The script uses early stopping and saves the best model weights to 'best_vit_v7.pth' based on validation accuracy.
3. Testing & Evaluation: 
   The testing block loads the best weights, performs Test-Time Augmentation (TTA) with a custom probability threshold (e.g., 60% confidence for 'Real'), and outputs the final accuracy.
4. Metrics: 
   The script generates and saves a confusion matrix (confusion_matrix_v9.png), training/validation loss curves (training_curves_v9.png), and a classification report (classification_report_v9.txt).

Notes:
------
- If running on Kaggle or Google Colab, ensure the 'NUM_WORKERS' parameter in the DataLoader is set to 2 to prevent CPU RAM out-of-memory crashes.
- The default batch size is 32 with 4 accumulation steps to simulate an effective batch size of 128 without exceeding 16GB VRAM.