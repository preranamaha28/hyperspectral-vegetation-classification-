from pathlib import Path
import io, csv, warnings, re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import MiniBatchKMeans

try:
    import spectral
except Exception:
    spectral = None
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    streamlit_image_coordinates = None

st.set_page_config(page_title="Hyperspectral Vegetation Classification", page_icon="🌿", layout="wide")
st.markdown("""
<style>
:root { --forest:#123f30; --leaf:#19744f; --ink:#173448; --line:#dce9e4; --paper:#ffffff; }
.stApp {background:#eef5f2;color:var(--ink)}
[data-testid="stHeader"] {background:rgba(238,245,242,.94)}
.block-container {padding-top:.7rem;padding-bottom:1.5rem;max-width:1900px}
.hero {background:linear-gradient(110deg,#103e2f,#1b5a43);color:white;padding:18px 24px;border-radius:13px;margin-bottom:14px;box-shadow:0 3px 14px #163e3014}
.hero h1 {margin:0;color:white;font-size:1.75rem;letter-spacing:-.02em}
.hero p {margin:5px 0 0;color:#dcece4;font-size:.95rem}
.panel {background:white;border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:12px;box-shadow:0 2px 8px #143d2e08}
.panel-title {font-size:1.04rem;font-weight:700;color:#183b4d;margin:0 0 10px;padding:8px 10px;background:#fff;border:1px solid var(--line);border-radius:9px}
div[data-testid="stMetric"] {background:#fff;padding:10px 12px;border-radius:9px;border:1px solid #dfebe5;box-shadow:0 1px 4px #173d2f08}
[data-testid="stSidebar"] {background:#f7fbf9;border-right:1px solid #dce9e4}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {color:#173f32}
.stButton>button[kind="primary"] {background:#16734d;border-color:#16734d}
.stDownloadButton>button {border-radius:8px;border-color:#cbded5}
[data-testid="stDataFrame"] {border:1px solid #e0ebe5;border-radius:8px;overflow:hidden}
img {border-radius:8px}
</style>
<div class="hero"><h1>🌿 Hyperspectral Vegetation Classification and Analysis</h1>
<p>Explore hyperspectral data, classify land cover, inspect pixel spectra, and export analysis reports.</p></div>
""", unsafe_allow_html=True)

CLASS_NAMES = {
  1:"Alfalfa",2:"Corn — notill",3:"Corn — mintill",4:"Corn",
  5:"Grass — pasture",6:"Grass — trees",7:"Grass — pasture mowed",8:"Hay — windrowed",
  9:"Oats",10:"Soybean — notill",11:"Soybean — mintill",12:"Soybean — clean",
  13:"Wheat",14:"Woods",15:"Buildings, grass, trees & drives",16:"Stone-steel towers"
}
PALETTE = np.array([[0,0,0],[44,160,44],[255,215,0],[255,140,0],[210,50,50],[50,180,90],
 [20,120,40],[150,210,80],[180,140,80],[120,90,50],[40,100,220],[30,180,200],
 [170,80,200],[230,100,160],[60,70,90],[180,180,180],[120,80,40]],dtype=np.uint8)

def cube_from_mat(raw):
    mats = {k:v for k,v in raw.items() if not k.startswith("__") and isinstance(v,np.ndarray) and v.ndim >= 2 and np.issubdtype(v.dtype,np.number)}
    if not mats: raise ValueError("No numeric array found in the MAT file.")
    # Prefer 3-D arrays; otherwise let user know this file may not contain a cube.
    candidates = [(k,v) for k,v in mats.items() if v.ndim == 3]
    if not candidates: raise ValueError("No 3D hyperspectral cube found in this MAT file. Available numeric arrays: "+", ".join(f"{k}:{v.shape}" for k,v in mats.items()))
    candidates.sort(key=lambda kv: kv[1].size, reverse=True)
    name, arr = candidates[0]
    return np.asarray(arr), name, mats

def normalize_cube(arr):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 3: raise ValueError(f"Expected a 3D cube; received shape {arr.shape}.")
    # Standard cube shape is H x W x bands. Move the smallest axis to the end when likely band-first.
    if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2] and arr.shape[0] <= 512:
        arr = np.moveaxis(arr, 0, -1)
    elif arr.shape[1] < arr.shape[0] and arr.shape[1] < arr.shape[2] and arr.shape[1] <= 512:
        # H x bands x W is less common
        pass
    arr = arr.astype(np.float32, copy=False)
    arr[~np.isfinite(arr)] = 0
    return arr

def load_cube(upload, hdr_upload=None):
    name = upload.name.lower()
    if name.endswith(".npy"):
        arr = np.load(upload, allow_pickle=False)
        return normalize_cube(arr), {"source":upload.name, "variable":"NumPy array", "wavelengths":None}
    if name.endswith(".npz"):
        z = np.load(upload, allow_pickle=False)
        keys = [k for k in z.files if z[k].ndim == 3]
        if not keys: raise ValueError("NPZ contains no 3D arrays.")
        key = max(keys, key=lambda k:z[k].size)
        return normalize_cube(z[key]), {"source":upload.name, "variable":key, "wavelengths":None}
    if name.endswith(".mat"):
        mats = loadmat(io.BytesIO(upload.getvalue()))
        arr, var, allmats = cube_from_mat(mats)
        wavelengths = None
        for k,v in allmats.items():
            if "wave" in k.lower() and v.size == arr.shape[-1]:
                wavelengths = np.asarray(v).reshape(-1)
        return normalize_cube(arr), {"source":upload.name,"variable":var,"wavelengths":wavelengths}
    if name.endswith(".hdr"):
        if spectral is None: raise ValueError("Install spectral: python -m pip install spectral")
        # Save header and optional paired data file into a temporary folder.
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        hdr_path = Path(tmp.name) / Path(upload.name).name
        header_bytes = upload.getvalue()
        # If a paired binary file was uploaded separately, point the ENVI header
        # at that exact temporary filename instead of relying on the original path.
        if hdr_upload is not None:
            header_text = header_bytes.decode("utf-8", errors="replace")
            data_name = Path(hdr_upload.name).name
            if re.search(r"(?im)^\\s*data\\s*file\\s*=", header_text):
                header_text = re.sub(
                    r"(?im)^(\\s*data\\s*file\\s*=\\s*).*?$",
                    lambda m: m.group(1) + data_name,
                    header_text,
                )
            else:
                header_text = header_text.rstrip() + f"\\ndata file = {{{data_name}}}\\n"
            header_bytes = header_text.encode("utf-8")
        hdr_path.write_bytes(header_bytes)
        if hdr_upload is not None:
            data_path = Path(tmp.name) / Path(hdr_upload.name).name
            data_path.write_bytes(hdr_upload.getvalue())
        img = spectral.open_image(str(hdr_path))
        arr = np.asarray(img.load())
        wavelengths = getattr(img, "metadata", {}).get("wavelength")
        if wavelengths:
            try: wavelengths = np.asarray([float(x) for x in wavelengths])
            except Exception: wavelengths = None
        else: wavelengths = None
        cube = normalize_cube(arr)
        # spectral object lazily reads; array is materialized before temp folder goes away.
        tmp.cleanup()
        return cube, {"source":upload.name,"variable":"ENVI image","wavelengths":wavelengths}
    if name.endswith((".tif",".tiff")):
        import tifffile
        arr = tifffile.imread(io.BytesIO(upload.getvalue()))
        return normalize_cube(arr), {"source":upload.name,"variable":"TIFF stack","wavelengths":None}
    raise ValueError("Unsupported file type. Use .npy, .npz, .mat, .hdr or multiband .tif/.tiff.")

def load_labels(upload):
    if upload is None: return None
    n=upload.name.lower()
    if n.endswith(".npy"): a=np.load(upload,allow_pickle=False)
    elif n.endswith(".mat"):
        m=loadmat(io.BytesIO(upload.getvalue()))
        opts=[(k,v) for k,v in m.items() if not k.startswith("__") and isinstance(v,np.ndarray) and v.ndim==2 and np.issubdtype(v.dtype,np.number)]
        if not opts: raise ValueError("No 2D numeric label map found in MAT file.")
        a=max(opts,key=lambda kv:kv[1].size)[1]
    elif n.endswith((".tif",".tiff")):
        import tifffile
        a=tifffile.imread(io.BytesIO(upload.getvalue()))
    else: raise ValueError("Labels must be .npy, .mat, .tif or .tiff.")
    a=np.squeeze(a)
    if a.ndim!=2: raise ValueError(f"Label map must be 2D; got {a.shape}.")
    return a.astype(np.int32)

def rgb_view(cube):
    b=cube.shape[2]
    # False-colour display bands; these are display choices, not calibrated RGB wavelengths.
    idx=[int(.75*(b-1)),int(.45*(b-1)),int(.15*(b-1))]
    rgb=cube[:,:,idx].astype(np.float32)
    lo,hi=np.percentile(rgb,[2,98])
    if hi<=lo: hi=lo+1
    return (np.clip((rgb-lo)/(hi-lo),0,1)*255).astype(np.uint8)

def plot_spectrum(values, wavelengths=None, title="Selected pixel spectral signature"):
    fig,ax=plt.subplots(figsize=(8,3))
    x=np.asarray(wavelengths) if wavelengths is not None and len(wavelengths)==len(values) else np.arange(1,len(values)+1)
    ax.plot(x,values,linewidth=1.8)
    ax.set_xlabel("Wavelength" if wavelengths is not None and len(wavelengths)==len(values) else "Band number")
    ax.set_ylabel("Spectral value")
    ax.set_title(title); ax.grid(alpha=.25); fig.tight_layout()
    return fig

def class_name(class_id, custom=None):
    if custom and int(class_id) in custom: return custom[int(class_id)]
    return CLASS_NAMES.get(int(class_id), f"Class {int(class_id)}")

def make_class_map(pred, labels=None):
    ids = sorted(int(v) for v in np.unique(pred) if int(v)>=0)
    out=np.zeros((*pred.shape,3),dtype=np.uint8)
    for i,c in enumerate(ids):
        color=PALETTE[(c % (len(PALETTE)-1))+1]
        out[pred==c]=color
    if labels is not None:
        out[labels==0]=[30,30,30]
    return out

with st.sidebar:
    st.header("1. Upload data")
    cube_file=st.file_uploader("Hyperspectral cube",type=["npy","npz","mat","hdr","tif","tiff"])
    hdr_data=st.file_uploader("Matching ENVI data file (only if needed for .hdr)",type=["img","dat","raw","bil","bip","bsq"])
    label_file=st.file_uploader("Optional ground-truth labels (.npy/.mat/.tif)",type=["npy","mat","tif","tiff"])
    pixel_size=st.number_input("Pixel size (metres; if known)",min_value=0.0,value=1.0,step=0.5)
    st.caption("For ENVI .hdr, upload its matching .img/.dat/.raw file only if the header does not already resolve it.")
    st.divider()
    st.header("2. Classification")
    model_choice=st.radio("Method",["Train SVM from ground-truth labels","Unsupervised clustering (no semantic names)"])
    max_samples=st.slider("Maximum training pixels",1000,30000,10000,1000)
    st.caption("SVM requires a label map aligned to the cube. Cluster IDs are not real-world names.")
    train=st.button("Train / Run Classification",type="primary",use_container_width=True)

if "cube" not in st.session_state: st.session_state.cube=None
if cube_file is not None:
    # Reload only when filename changes, so reruns from pixel clicks don't lose the dataset.
    if st.session_state.get("cube_filename") != cube_file.name:
        try:
            cube,meta=load_cube(cube_file,hdr_data)
            st.session_state.cube=cube; st.session_state.meta=meta
            st.session_state.cube_filename=cube_file.name
            st.session_state.pop("pred",None); st.session_state.pop("model",None)
            st.session_state.pop("labels",None)
        except Exception as e:
            st.error(f"Could not load hyperspectral data: {e}")
if st.session_state.cube is None:
    st.info("Upload a hyperspectral cube from the sidebar to begin. Supported formats: .npy, .npz, .mat, ENVI .hdr with its matching binary data, and multiband TIFF.")
    st.markdown("### What is a ground-truth label map?")
    st.write("It is a 2D array the same height and width as the cube. Each pixel stores a known class ID; 0 usually means unlabelled in datasets such as Indian Pines. It is optional for viewing, but required for supervised SVM training.")
    st.stop()

cube=st.session_state.cube; meta=st.session_state.meta
H,W,B=cube.shape
st.caption(f"Loaded **{meta['source']}** · array `{meta['variable']}` · shape **{H} × {W} × {B}** (height × width × bands)")
try:
    labels=load_labels(label_file) if label_file is not None else None
    if labels is not None and labels.shape!=(H,W):
        st.error(f"Label map shape {labels.shape} does not match cube spatial shape {(H,W)}. Use a matching label map.")
        labels=None
    st.session_state.labels=labels
except Exception as e:
    st.error(f"Could not read label map: {e}")
    labels=None

# Train model or create unsupervised map on explicit button click.
if train:
    with st.spinner("Preparing pixels and running classification..."):
        X=cube.reshape(-1,B)
        if model_choice=="Train SVM from ground-truth labels":
            if labels is None:
                st.error("SVM needs a matching ground-truth label map. Upload a matching label map in the sidebar, or choose unsupervised clustering.")
            else:
                y=labels.reshape(-1)
                mask=(y>0)&np.all(np.isfinite(X),axis=1)
                inds=np.flatnonzero(mask)
                if len(inds)<10:
                    st.error("Not enough labelled pixels (nonzero labels) to train. Check your label file.")
                else:
                    if len(inds)>max_samples:
                        rng=np.random.default_rng(42); inds=rng.choice(inds,max_samples,replace=False)
                    Xs=X[inds]; ys=y[inds]
                    if len(np.unique(ys))<2:
                        st.error("SVM training needs at least two distinct labelled classes.")
                    else:
                        try:
                            tr,te=train_test_split(np.arange(len(ys)),test_size=.2,random_state=42,stratify=ys)
                        except Exception:
                            tr,te=train_test_split(np.arange(len(ys)),test_size=.2,random_state=42)
                        model=make_pipeline(StandardScaler(),SVC(kernel="rbf",C=10,gamma="scale",probability=True,random_state=42))
                        model.fit(Xs[tr],ys[tr])
                        predictions=[]; confidences=[]
                        for start_i in range(0,len(X),50000):
                            chunk=X[start_i:start_i+50000]
                            predictions.append(model.predict(chunk))
                            try: confidences.append(model.predict_proba(chunk).max(axis=1))
                            except Exception: confidences.append(np.full(len(chunk),np.nan))
                        st.session_state.pred=np.concatenate(predictions).reshape(H,W)
                        st.session_state.conf=np.concatenate(confidences).reshape(H,W)
                        st.session_state.model=model
                        st.session_state.eval={"accuracy":accuracy_score(ys[te],model.predict(Xs[te])),"y_true":ys[te],"y_pred":model.predict(Xs[te])}
                        st.success("Classification completed.")
        else:
            rng=np.random.default_rng(42)
            take=min(len(X),max_samples)
            sample=X[rng.choice(len(X),take,replace=False)]
            km=MiniBatchKMeans(n_clusters=min(16,max(2,int(st.session_state.get("n_clusters",8)))),random_state=42,batch_size=2048,n_init=3)
            km.fit(sample)
            st.session_state.pred=km.predict(X).reshape(H,W)
            st.session_state.conf=np.full((H,W),np.nan)
            st.session_state.model=None
            st.session_state.eval=None
            st.session_state.cluster_model=km
            st.success("Clustering completed. Cluster IDs are spectral groups, not verified land-cover names.")

pred=st.session_state.get("pred")
conf=st.session_state.get("conf")
if model_choice=="Unsupervised clustering (no semantic names)":
    ncl=st.slider("Number of spectral clusters",2,16,8,key="cluster_slider")
    st.session_state.n_clusters=ncl

# Dashboard header and summary row.
metric1,metric2,metric3,metric4=st.columns(4)
metric1.metric("Image height",f"{H:,} px")
metric2.metric("Image width",f"{W:,} px")
metric3.metric("Spectral bands",f"{B:,}")
metric4.metric("Classification", "Ready" if pred is not None else "Not run")

rgb=rgb_view(cube)
if "selected_pixel" not in st.session_state:
    st.session_state.selected_pixel=(H//2,W//2)
if "manual_row" not in st.session_state: st.session_state.manual_row=H//2
if "manual_col" not in st.session_state: st.session_state.manual_col=W//2

left,right=st.columns([1.65,1],gap="medium")
with left:
    st.markdown('<div class="panel-title">A. Hyperspectral Image Viewer</div>',unsafe_allow_html=True)
    tab_rgb,tab_class,tab_gt=st.tabs(["RGB Composite","Class Map","Ground Truth"])
    with tab_rgb:
        st.caption("Click directly on the image to inspect a pixel. Selected row and column update automatically.")
        show_image=Image.fromarray(rgb)
        try:
            from PIL import ImageDraw
            draw=ImageDraw.Draw(show_image)
            rr,cc=st.session_state.selected_pixel
            rad=max(3,int(min(H,W)*0.012))
            draw.ellipse((cc-rad,rr-rad,cc+rad,rr+rad),outline=(255,255,255),width=2)
            draw.ellipse((cc-2,rr-2,cc+2,rr+2),fill=(255,40,60))
        except Exception: pass
        if streamlit_image_coordinates is not None:
            click_value=streamlit_image_coordinates(show_image,width=min(720,W),key="hyperspectral_click_v2")
            if isinstance(click_value,dict) and click_value.get("x") is not None and click_value.get("y") is not None:
                # The component reports coordinates in the rendered image; scale them to cube coordinates.
                rendered_w=float(click_value.get("width") or min(720,W))
                rendered_h=float(click_value.get("height") or max(1,round(H*rendered_w/W)))
                clicked_c=int(np.clip(round(float(click_value["x"])*W/rendered_w),0,W-1))
                clicked_r=int(np.clip(round(float(click_value["y"])*H/rendered_h),0,H-1))
                new_pixel=(clicked_r,clicked_c)
                if new_pixel != st.session_state.selected_pixel:
                    st.session_state.selected_pixel=new_pixel
                    st.session_state.manual_row=clicked_r
                    st.session_state.manual_col=clicked_c
                    st.rerun()
        else:
            st.image(show_image,use_container_width=True)
            st.warning("Pixel-click support is unavailable. In the project terminal run: python -m pip install streamlit-image-coordinates")
    with tab_class:
        if pred is not None:
            st.image(make_class_map(pred,labels),use_container_width=True,caption="Predicted class map — colours correspond to class IDs.")
        else:
            st.info("Run classification using the button in the left sidebar to create a class map.")
    with tab_gt:
        if labels is not None:
            st.image(make_class_map(labels),use_container_width=True,caption="Ground-truth labels supplied with the dataset.")
        else:
            st.info("No ground-truth label map supplied. Upload a matching .npy, .mat or .tif label file in the sidebar.")

    st.markdown('<div class="panel-title">C. Classification Map &nbsp;&nbsp; · &nbsp;&nbsp; D. Mean Spectral Signature by Class</div>',unsafe_allow_html=True)
    submap,subspec=st.columns([1,1.25],gap="medium")
    with submap:
        if pred is not None:
            st.image(make_class_map(pred,labels),use_container_width=True)
            st.caption("Classification map")
        else:
            st.info("Classification map appears here after running the model.")
    with subspec:
        if pred is not None:
            fig,ax=plt.subplots(figsize=(6,3.3))
            ids=sorted(int(x) for x in np.unique(pred))
            for i in ids:
                m=pred==i
                if np.any(m): ax.plot(np.arange(1,B+1),np.mean(cube[m],axis=0),linewidth=1,label=class_name(i))
            ax.set_title("Mean spectral signature by class"); ax.set_xlabel("Band number"); ax.set_ylabel("Mean reflectance / intensity")
            ax.grid(alpha=.2); ax.legend(ncol=2,fontsize=6); fig.tight_layout(); st.pyplot(fig,use_container_width=True)
        else:
            st.info("Mean spectra will appear after classification.")

with right:
    st.markdown('<div class="panel-title">B. Selected Pixel Information</div>',unsafe_allow_html=True)
    r0,c0=st.session_state.selected_pixel
    gt=int(labels[r0,c0]) if labels is not None else None
    pred_id=int(pred[r0,c0]) if pred is not None else None
    confidence=float(conf[r0,c0]) if conf is not None and np.isfinite(conf[r0,c0]) else None
    q1,q2=st.columns(2)
    with q1: row=int(st.number_input("Row",0,H-1,int(r0),key="manual_row"))
    with q2: col=int(st.number_input("Column",0,W-1,int(c0),key="manual_col"))
    if (row,col)!=(r0,c0):
        st.session_state.selected_pixel=(row,col)
        st.rerun()
    pixel=cube[row,col,:]
    gt=int(labels[row,col]) if labels is not None else None
    pred_id=int(pred[row,col]) if pred is not None else None
    confidence=float(conf[row,col]) if conf is not None and np.isfinite(conf[row,col]) else None
    q1,q2=st.columns(2)
    q1.metric("Predicted class",class_name(pred_id) if pred_id is not None else "Not classified")
    q2.metric("Confidence",f"{confidence*100:.1f}%" if confidence is not None else "N/A")
    st.caption(f"Reference label: **{class_name(gt) if gt is not None and gt>0 else ('Unlabelled' if gt==0 else 'Not supplied')}**")
    st.caption(f"Selected pixel: row **{row}**, column **{col}** · bands: **{B}**")
    fig=plot_spectrum(pixel,meta.get("wavelengths"),title="Spectral Signature (Selected Pixel)")
    st.pyplot(fig,use_container_width=True)
    if pred is not None:
        fig2,ax2=plt.subplots(figsize=(6,2.5))
        ax2.plot(np.arange(1,B+1),pixel,label="Selected pixel",linewidth=1.8)
        if pred_id is not None:
            cmask=pred==pred_id
            if np.any(cmask): ax2.plot(np.arange(1,B+1),np.mean(cube[cmask],axis=0),"--",label=f"Class mean ({class_name(pred_id)})",linewidth=1.5)
        ax2.set_title("Compare with Class Mean Signature"); ax2.set_xlabel("Band number"); ax2.set_ylabel("Reflectance / intensity")
        ax2.grid(alpha=.2); ax2.legend(fontsize=7); fig2.tight_layout(); st.pyplot(fig2,use_container_width=True)

    st.markdown('<div class="panel-title">E. Class Statistics</div>',unsafe_allow_html=True)
    if pred is not None:
        ids=sorted(int(x) for x in np.unique(pred))
        stats=pd.DataFrame([{"Class":class_name(i),"Class ID":i,"Pixels":int(np.sum(pred==i)),"Area (m²)":float(np.sum(pred==i)*pixel_size**2) if pixel_size>0 else np.nan,"%":round(float(np.mean(pred==i)*100),2)} for i in ids])
        st.dataframe(stats,use_container_width=True,hide_index=True,height=230)
        st.download_button("Export all results (CSV)",stats.to_csv(index=False).encode("utf-8"),file_name="class_statistics.csv",mime="text/csv",use_container_width=True)
        spec=pd.DataFrame({"Band":np.arange(1,B+1),"Wavelength":meta.get("wavelengths") if meta.get("wavelengths") is not None and len(meta["wavelengths"])==B else [None]*B,"Spectral value":pixel})
        st.download_button("Download selected pixel spectrum CSV",spec.to_csv(index=False).encode("utf-8"),file_name=f"pixel_row{row}_col{col}_spectrum.csv",mime="text/csv",use_container_width=True)
        if st.checkbox("Include full pixel predictions CSV (may be large)"):
            pixdf=pd.DataFrame({"row":np.repeat(np.arange(H),W),"column":np.tile(np.arange(W),H),"predicted_class":pred.reshape(-1)})
            if labels is not None: pixdf["ground_truth_label"]=labels.reshape(-1)
            if conf is not None: pixdf["confidence"]=conf.reshape(-1)
            st.download_button("Download full pixel predictions CSV",pixdf.to_csv(index=False).encode("utf-8"),file_name="pixel_classification.csv",mime="text/csv",use_container_width=True)
        if st.session_state.get("eval"):
            ev=st.session_state.eval
            st.metric("Holdout accuracy",f"{ev['accuracy']*100:.2f}%")
            with st.expander("Classification report"):
                st.code(classification_report(ev["y_true"],ev["y_pred"],zero_division=0))
    else:
        st.info("Run classification to show class statistics and CSV exports.")

st.markdown("---")
st.caption("Scientific note: the composite is false-colour, not guaranteed true RGB. Supervised class names require matching ground-truth labels; unsupervised cluster IDs are not verified land-cover categories.")
