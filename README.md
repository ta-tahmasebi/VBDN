# Malware Classification CLI

This CLI trains a compact ConvNet, pretrained image models, and classical
classifiers based on GLCM texture features. Every evaluation writes summary
metrics, per-class reports, predictions, confusion matrices, and comparison
charts below `results/`.

## Setup

```bash
source ~/Codes/env10/bin/activate
pip install -r requirements.txt
```

BIG2015 conversion also requires the `7z` executable (`p7zip-full` on Ubuntu).

## Commands

```bash
python main.py --help
python main.py convnet --datasets Malimg
python main.py pretrained --datasets Malevis --models VGG16 MobileNetV2
python main.py glcm --datasets Blended
python main.py run-all
python main.py run-all --paper-settings
```

Known Kaggle datasets are downloaded by default. Use `--no-download` to reuse
their newest versions from KaggleHub's default cache at
`~/.cache/kagglehub/datasets`, or pass any ImageFolder-compatible path directly.
MaleVis and Blended use their supplied `train` and `val` directories directly;
validation data is never mixed back into training. Malimg keeps the paper's
8,408 training images; the available 9,339-image archive leaves 931 test images
(Table 1 reports an inconsistent total four images larger). BIG2015 uses a
70/30 stratified split.
Local ImageFolder paths continue to use `--train-split`.

## BIG2015

The source folder must contain `train.7z` and `trainLabels.csv`. Its default is
`/home/amirmahdi/.cache/kaggle/BIG2015/main/`, and it can be overridden:

```bash
python main.py prepare-big2015 \
  --source /path/to/BIG2015/main \
  --workers 6

python main.py convnet \
  --datasets BIG2015 \
  --big2015-path /path/to/BIG2015/main
```

Each `.bytes` member is streamed from the archive into memory, converted to one
PNG, and released. The complete training archive is never extracted. Existing
PNGs are skipped, so interrupted conversion can resume safely. Use
`--big2015-samples-per-class 4` for a balanced smoke subset streamed directly
from `train.7z`, `--big2015-limit` for a global prefix, and
`--rebuild-big2015` to overwrite existing images. The converter never reads or
uses `Processed_Dataset`.

`run-all` runs every CNN and GLCM model on Malimg, Malevis, Blended, and
BIG2015. By default, its custom ConvNet experiments run for 80 epochs and each
pretrained deep model runs for 18 epochs. Override these independently with
`--convnet-epochs` and `--pretrained-epochs`, or set both at once with
`--epochs`. The `--no-download` flag only disables dataset downloads and does
not change the epoch profile. Individual commands default to the original
three datasets.

The `glcm` command runs two explicitly labeled experiments by default:

- `raw`: original training images, without resampling or augmentation
- `balanced-augmented`: index-based oversampling followed by resize, random
  horizontal flip, and random rotation before GLCM extraction

Run only one branch with `--glcm-variants raw` or
`--glcm-variants balanced-augmented`.

Balanced GLCM sampling draws one inverse-frequency weighted epoch with the
original training-set length, matching `BalancedDatasetSampler` instead of
expanding every class to the majority count. GLCM defaults to 256 gray levels,
following the paper's 0-255 image-intensity description; use `--glcm-levels 32`
or `--glcm-levels 8` if memory is constrained.

The main `convnet` command automatically adds a second MaleVis experiment with
the `Other` class removed. Disable it with `--no-malevis-without-other`.

## Paper alignment

Use `--paper-settings` to select the paper's disclosed training hyperparameters:
512-pixel input, three 3x3 convolution layers with 32/64/128 channels,
2x2 pooling, 28x28 global pooling, dense layers of 64 and the class count, 200
epochs, SGD learning rate 0.01, momentum 0.5, and seed 50. ConvNet uses
train/test batches 64/32, while pretrained comparisons use batch size 8.

Without `--paper-settings`, `run-all` uses 80 ConvNet epochs, 18 pretrained
epochs, 224-pixel images, ConvNet train/test batches 32/16, and pretrained batch
size 4. The randomly replaced classifier head uses learning rate 0.01 while the
ImageNet backbone uses 0.0001. This lets short runs learn the new classes without
destabilizing the pretrained feature extractor. Explicit CLI values remain
available; `--paper-settings` only overrides disclosed training parameters and
does not claim to fill in details omitted by the paper.

For a quick full-pipeline check without loading every sample, use:

```bash
python main.py run-all \
  --epochs 1 \
  --max-samples-per-class 4 \
  --big2015-samples-per-class 4 \
  --no-save-models
```

## Result layout

- `results/csv`: pipeline summaries and cross-model comparisons
- `results/reports`: detailed classification reports and JSON metrics
- `results/predictions`: sample-level predictions and class scores
- `results/plots`: class/image and augmentation galleries, GLCM examples,
  class-balance plots, feature importance, learning, confusion, ROC/PR,
  per-class, and comparison charts
- `results/models`: CNN checkpoints
- `results/last_run.json`: command, environment, and reproducibility metadata
