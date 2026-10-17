# IDeal Interactive Demo

An interactive web demo for **IDeal: Interactive Cross-modal Learning for Text-3D Scene Retrieval** (NeurIPS 2025).

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n ideal python=3.8.16
conda activate ideal

# Install PyTorch (GPU recommended)
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

**ScanNet Images (Required for visualization):**  
[Scannet images](https://drive.google.com/file/d/1qmMNOIi4v1unLQ0fLGa9BcEUweFj3BSF/view?usp=drive_link): Download `scenes.zip` unzip it and put it into `IDeal/demo/`

**Model Weights (Already provided):**
- [Pre-trained model](https://drive.google.com/drive/folders/1ol7bwABpCJfb_xmqN7WmkF2Ci_l8smCJ): download `checkpoint/transfer.pth` and put it into `IDeal/runs/scanrefer_train_butd_ESAregion_bigru/`
- [Point cloud data](https://drive.google.com/drive/folders/18QE38xkDeuvgshaH1yiwbnHat7qEktzW): download `pt2vec_200_random_pos_val.npy`, `pt2vec_200_random_val.npy` and put them into `IDeal/lib/data/`
- [Temp-results](https://drive.google.com/drive/folders/1ol7bwABpCJfb_xmqN7WmkF2Ci_l8smCJ): download `temp_results` and put it into `IDeal/`
- [BERT model](https://huggingface.co/google-bert/bert-base-uncased): download and put bert into `IDeal/`

### 3. Launch Demo

```bash
# Start the demo server
python demo/app.py --port 7860

# Access in browser: http://127.0.0.1:7860
```

## 📖 Usage Guide
**Notice**  
The scene images not include `scene0011_00`, `scene0015_00`, `scene0019_00`, `scene0025_00`, `scene0030_00`. 
So the final scene visualization is not available for these cases, but other parts still work.
### Auto Replay

Automatically replay the interactive retrieval process using precomputed results.

**Steps:**
1. **Select Scene** - Choose from 141 validation scenes (e.g., `scene0050_00`)
2. **Select Query** - Choose from 10 available queries per scene
3. **Click "Run Retrieval Replay"** - Watch the interaction unfold:
   - Chat messages appear one by one (initial query → Q&A rounds)
   - Top-5 retrieval results displayed in table
   - Top-1 scene images shown in gallery

**Features:**
- Real-time chat simulation with 3 rounds of interaction
- Adaptive questioning (detail probing vs. exploration divergence)
- Feature-level fusion retrieval (55% original + 35% interactive)
- ScanNet RGB image visualization for retrieved scenes

## 🔧 Troubleshooting

### Common Issues

**Q: "ModuleNotFoundError: No module named 'gradio'"**
```bash
pip install gradio==4.44.1
```

**Q: "Scene images not loading"**
- Ensure images are in: `demo/scenes/{scene_id}/images/*.jpg`
- Check that you've run `extract_scannet_images.py`

**Q: "Model loading slow"**
- First launch downloads BERT model (~400MB)
- Use GPU for faster inference (CUDA required)

**Q: "Out of memory"**
- Reduce batch size in `retrieval_engine.py`
- Close other applications
- Use CPU mode if GPU memory is insufficient

### Performance Tips

- **GPU acceleration**: Ensure CUDA is available (`torch.cuda.is_available()`)
- **Memory optimization**: Close other tabs in browser
- **Fast loading**: Pre-download BERT model before first run

## 📁 Project Structure

```
IDeal/
├── demo/
│   ├── app.py                 # Main Gradio application
│   ├── data_loader.py         # Scene and image loading
│   ├── retrieval_engine.py    # Retrieval model wrapper
│   └── scenes/                # ScanNet RGB images
│       ├── scene0050_00/
│       │   └── images/       # 47 RGB photos
│       └── ...
├── runs/                      # Pre-trained model weights
├── bert-base-uncased/         # BERT tokenizer and model
├── lib/                       # Core library code
├── temp_results/              # Precomputed results
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎯 Key Features

✅ **Interactive Retrieval Simulation** - Watch multi-round Q&A unfold  
✅ **Real-time Results** - See how retrieval improves with each interaction  
✅ **3D Scene Visualization** - View actual ScanNet room images  
✅ **Easy Deployment** - Single command to launch web interface  
✅ **Reproducible** - Uses precomputed results for consistent demos  

## 📚 References

- **Paper**: Feng et al., "Interactive Cross-modal Learning for Text-3D Scene Retrieval", NeurIPS 2025
- **Github**: [IDeal](https://github.com/Yangl1nFeng/IDeal)
- **Dataset**: ScanNet v2 (http://www.scan-net.org/)

---

**Enjoy exploring interactive 3D scene retrieval!** 🎉
