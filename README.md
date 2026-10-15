# Hyperspectral Vegetation Classification & Analysis

## Features
- Upload `.npy`, `.npz`, `.mat`, ENVI `.hdr` with matching binary data when needed, and multiband `.tif/.tiff`.
- Optional ground-truth labels as `.npy`, `.mat`, `.tif/.tiff`.
- False-colour composite preview and click-to-select pixel (with manual row/column fallback).
- Selected pixel coordinates, ground-truth label, predicted class, model confidence and spectral signature.
- SVM classification trained from supplied reference labels.
- Unsupervised spectral clustering when no labels are available (clusters are NOT automatically named river/building/etc.).
- Classification map, mean spectra by class, class pixel counts, estimated area and CSV downloads.

## Run on Windows (PowerShell)
1. Install Python 3.10 or newer.
2. Extract the ZIP and open the folder.
3. In PowerShell, change into the extracted folder:
   `cd path\to\hyperspectral_vegetation_classification`
4. Create a virtual environment:
   `py -m venv .venv`
5. Install dependencies:
   `.\.venv\Scripts\python.exe -m pip install --upgrade pip`
   `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
6. Start the app:
   `.\.venv\Scripts\python.exe -m streamlit run app.py`
7. Open the local URL printed in the terminal, normally `http://localhost:8501`.
8. Keep the terminal open while using the app. To stop it, press `Ctrl+C`.

## File guidance
- `.mat`: the app selects the largest numeric 3D array as the cube. If the file contains multiple cubes, check the variable name reported after loading.
- `.hdr`: upload the header and, if needed, its matching `.img`, `.dat`, `.raw`, `.bil`, `.bip` or `.bsq` file. Keep filenames/relationships consistent with the ENVI header metadata. Some headers reference a particular filename; if loading fails, rename the matching data file to the name specified in the header.
- Cube shape should be height × width × bands. The loader attempts a simple band-first correction, but ambiguous axes may require adjustment for a particular dataset.
- Ground-truth map must have the same height and width as the cube. Nonzero IDs are treated as labelled training pixels; 0 is treated as unlabelled.
- Indian Pines standard class names are shown for IDs 1–16. For other datasets, edit `CLASS_NAMES` in `app.py` to match that dataset's class-ID mapping. Do not use the Indian Pines mapping for unrelated data.

## Important scientific limits
- Ground-truth labels are not a required upload for browsing spectra, but are required for supervised SVM semantic classification.
- Clustering without labels only groups similar spectra; it cannot reliably infer words like river, garden, road or building.
- SVM is trained and tested using a random pixel split, which can overestimate performance due to spatial correlation. Use spatially separated validation for research.
- Confidence is the SVM's maximum class probability estimate, not a guarantee that the class is correct.
- RGB is a false-colour composite selected from approximate band positions; true RGB requires sensor wavelength metadata and appropriate band selection.
