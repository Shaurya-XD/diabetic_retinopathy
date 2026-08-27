"""EyeZen V4 multi-domain inference API."""
import io, json, logging, os, secrets, socket, uuid
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
import cv2, numpy as np
from PIL import Image
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT=Path(__file__).parent; ART=Path(os.getenv("ARTIFACTS_DIR",ROOT/"artifacts")); OUT=Path(os.getenv("OUTPUT_DIR",ROOT/"runtime"))
for d in (OUT/"images",OUT/"reports"): d.mkdir(parents=True,exist_ok=True)
DEMO_MODE=os.getenv("DEMO_MODE","false").lower()=="true"; INTERNAL_ML_SECRET=os.getenv("INTERNAL_ML_SECRET",""); ALLOW_UNAUTHENTICATED_LOCAL_INFER=os.getenv("ALLOW_UNAUTHENTICATED_LOCAL_INFER","false").lower()=="true"; CLASSES=["No DR","Mild DR","Moderate DR","Severe DR","Proliferative DR"]
REQUIRED_CALIBRATION_KEYS={"grade_temperature","screen_fusion_alpha_binary_head","screen_fusion_alpha_grade_sum","platt_a","platt_b","referral_threshold","threshold_rule"}
logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s"); log=logging.getLogger("eyezen.ml")
classifier=segmenter=grad_model=calibration=deployment=None; status={"classifierLoaded":False,"segmentationLoaded":False,"calibrationLoaded":False}
def artifact(name): return ART/name
def load_v4_artifacts():
 global calibration,deployment
 required=["dr_multidomain_efficientnetb0_v4.keras","calibration_v4.json","deployment_config_v4.json","run_summary_v4.json","idrid_lesion_unet_final.weights.h5"]
 missing=[n for n in required if not artifact(n).is_file()]
 if missing: raise RuntimeError(f"Missing required V4 artifact(s): {', '.join(missing)}")
 try: calibration=json.loads(artifact("calibration_v4.json").read_text(encoding="utf-8"));deployment=json.loads(artifact("deployment_config_v4.json").read_text(encoding="utf-8"))
 except json.JSONDecodeError as exc: raise RuntimeError(f"Invalid V4 JSON artifact: {exc}") from exc
 missing_keys=REQUIRED_CALIBRATION_KEYS-calibration.keys()
 if missing_keys: raise RuntimeError(f"calibration_v4.json is missing keys: {', '.join(sorted(missing_keys))}")
 if float(calibration["grade_temperature"])<=0 or not 0<=float(calibration["referral_threshold"])<=1: raise RuntimeError("calibration_v4.json contains invalid temperature or referral threshold")
 status["calibrationLoaded"]=True
def init_models():
 global classifier,segmenter,grad_model
 load_v4_artifacts()
 if DEMO_MODE: log.warning("Inference mode: DEMO (explicitly enabled)");return
 try:
  import tensorflow as tf
  classifier=tf.keras.models.load_model(artifact("dr_multidomain_efficientnetb0_v4.keras"),compile=False);grad_model=build_gradcam_model(classifier);status["classifierLoaded"]=True
  from model_adapter import build_unet
  segmenter=build_unet();segmenter.load_weights(artifact("idrid_lesion_unet_final.weights.h5"));status["segmentationLoaded"]=True;log.info("REAL V4 classifier and IDRiD U-Net loaded")
 except Exception as exc:
  classifier=segmenter=grad_model=None;status["classifierLoaded"]=status["segmentationLoaded"]=False;log.exception("V4 startup failed");raise RuntimeError("REAL V4 model loading failed; inspect the service log.") from exc
def crop_fundus(image):
 rgb=np.asarray(image.convert("RGB"));foreground=np.max(rgb,axis=2)>10;ys,xs=np.where(foreground)
 if not len(xs): return image.convert("RGB")
 pad=max(2,int(min(rgb.shape[:2])*.01));return Image.fromarray(rgb[max(0,ys.min()-pad):min(rgb.shape[0],ys.max()+pad+1),max(0,xs.min()-pad):min(rgb.shape[1],xs.max()+pad+1)])
def temperature_scale(probabilities):
 logits=np.log(np.clip(np.asarray(probabilities,dtype=np.float64),1e-8,1.0))/float(calibration["grade_temperature"]);logits-=np.max(logits);exp=np.exp(logits);return (exp/exp.sum()).astype(np.float32)
def calibrate_referable_probability(binary_probability,grade_probability):
 fused=float(calibration["screen_fusion_alpha_binary_head"])*binary_probability+float(calibration["screen_fusion_alpha_grade_sum"])*grade_probability;fused=float(np.clip(fused,1e-8,1-1e-8));logit=np.log(fused/(1-fused));return float(1/(1+np.exp(-(float(calibration["platt_a"])*logit+float(calibration["platt_b"]))))),fused
def normalize_classifier_outputs(outputs):
 if isinstance(outputs,dict): grade,referable=outputs.get("grade"),outputs.get("referable")
 elif isinstance(outputs,(list,tuple)) and len(outputs)>=2: grade,referable=outputs[0],outputs[1]
 else: raise RuntimeError("V4 classifier must return both grade and referable outputs")
 if grade is None or referable is None: raise RuntimeError("V4 classifier outputs are missing grade or referable")
 grade=np.asarray(grade,dtype=np.float32).reshape(-1,5)[0];referable=float(np.asarray(referable,dtype=np.float32).reshape(-1)[0])
 if not np.isfinite(grade).all() or not np.isfinite(referable) or not 0<=referable<=1: raise RuntimeError("V4 classifier returned invalid output values")
 return grade,referable
def build_gradcam_model(model):
 import tensorflow as tf
 backbone=model.get_layer("efficientnetb0");extractor=tf.keras.Model(backbone.input,[backbone.get_layer("top_conv").output,backbone.output]);input_=tf.keras.Input(shape=model.input_shape[1:]);conv,features=extractor(input_,training=False);x=model.get_layer("gap")(features);x=model.get_layer("referable_dropout")(x,training=False);logit=model.get_layer("referable_logit")(x);return tf.keras.Model(input_,[conv,logit],name="v4_referable_gradcam")
def generate_gradcam(classifier_input):
 import tensorflow as tf
 with tf.GradientTape() as tape: features,logit=grad_model(classifier_input,training=False);target=tf.reduce_sum(logit)
 gradients=tape.gradient(target,features)
 if gradients is None: raise RuntimeError("Grad-CAM gradients are None")
 features=tf.cast(features,tf.float32);gradients=tf.cast(gradients,tf.float32);weights=tf.reduce_mean(gradients,axis=(1,2));cam=tf.nn.relu(tf.reduce_sum(features*weights[:,None,None,:],axis=-1)[0]);maximum=tf.reduce_max(cam);cam=tf.cond(maximum>0,lambda:cam/maximum,lambda:tf.zeros_like(cam));return np.ascontiguousarray(np.nan_to_num(cam.numpy().astype(np.float32),nan=0.,posinf=0.,neginf=0.))
def save_png(image,path): Image.fromarray(image).save(path,"PNG");return "/images/"+path.name
def fetch_private_image(url):
 parsed=urlparse(url)
 query=parse_qs(parsed.query)
 if parsed.scheme!="https" or not (parsed.hostname or "").endswith(".private.blob.vercel-storage.com") or not query.get("vercel-blob-delegation") or not query.get("vercel-blob-signature"):raise HTTPException(400,"Invalid private image source.")
 try:
  with urlopen(Request(url,headers={"Accept":"image/jpeg,image/png"}),timeout=45) as response:
   status=response.status;content_type=response.headers.get_content_type();data=response.read(12*1024*1024+1)
  if status!=200:log.warning("Private input GET failed: status=%s host=%s",status,parsed.hostname);raise ValueError("unexpected image response")
  if content_type not in ("image/jpeg","image/png") or len(data)>12*1024*1024:log.warning("Private input GET rejected: status=%s content_type=%s host=%s",status,content_type,parsed.hostname);raise ValueError("invalid image response")
  return data
 except HTTPError as exc:log.warning("Private input GET failed: status=%s host=%s",exc.code,parsed.hostname);raise HTTPException(400,"Unable to retrieve the uploaded image.") from exc
 except (socket.timeout,TimeoutError):log.warning("Private input GET timed out: host=%s",parsed.hostname);raise HTTPException(400,"Unable to retrieve the uploaded image.")
 except URLError as exc:log.warning("Private input GET failed: network_error=%s host=%s",type(exc.reason).__name__,parsed.hostname);raise HTTPException(400,"Unable to retrieve the uploaded image.") from exc
 except Exception as exc:raise HTTPException(400,"Unable to retrieve the uploaded image.") from exc
def put_scoped(url,path,content_type):
 try:
  request=Request(url,data=path.read_bytes(),method="PUT",headers={"Content-Type":content_type});response=urlopen(request,timeout=45)
  if not 200<=response.status<300:raise RuntimeError(f"asset upload returned HTTP {response.status}")
 except Exception as exc:raise RuntimeError("Unable to persist generated asset to private storage") from exc
def make_overlay(base,mask):
 layer=np.zeros_like(base);layer[mask]=(220,32,44);return cv2.addWeighted(base,1,layer,.48,0)
def build_inference_result(raw_grade,raw_binary):
 probabilities=temperature_scale(raw_grade);grade=int(np.argmax(probabilities));grade_based=float(probabilities[2:].sum());referable,fused=calibrate_referable_probability(raw_binary,grade_based);threshold=float(calibration["referral_threshold"]);decision="REFER" if referable>=threshold else "NON_REFER"
 result={"modelVersion":"V4 Multi-Domain","grade":grade,"gradeLabel":CLASSES[grade],"gradeConfidence":round(float(probabilities[grade]),5),"gradeProbabilities":{f"grade{i}":round(float(v),5) for i,v in enumerate(probabilities)},"rawBinaryReferableProbability":round(raw_binary,5),"gradeBasedReferableProbability":round(grade_based,5),"referableProbability":round(referable,5),"referralThreshold":threshold,"decision":decision,"lesionOverlayUrl":None,"gradcamUrl":None,"reportUrl":None,"segmentationAvailable":bool(status["segmentationLoaded"]),"explanation":{"lesionLocalization":"U-Net","classifierAttention":"Grad-CAM"},"processing":{"classifierInput":"cropped RGB, 300x300, float32 0-255","segmentationInput":"cropped RGB, 384x384, float32 0-255"}}
 result.update({"predictedGrade":grade,"calibratedConfidence":result["gradeConfidence"],"classProbabilities":list(result["gradeProbabilities"].values()),"referralDecision":decision,"gradCamUrl":None,"modelMode":"demo" if DEMO_MODE else "real","fusedReferableProbability":round(fused,5)})
 return result
def create_report(path,result,lesion_path,gradcam_path,patient):
 c=canvas.Canvas(str(path),pagesize=letter);c.setTitle("EyeZen screening report");width,height=letter
 def heading(text,y): c.setFillColor("#17283b");c.setFont("Helvetica-Bold",12);c.drawString(48,y,text);c.setStrokeColor("#9fb5c8");c.line(48,y-5,width-48,y-5);return y-22
 def row(label,value,y): c.setFillColor("#3b4b5c");c.setFont("Helvetica-Bold",9);c.drawString(52,y,label);c.setFillColor("#101820");c.setFont("Helvetica",9);c.drawString(190,y,str(value if value not in (None,"") else "—"));return y-16
 c.setFillColor("#0f5f8d");c.rect(0,height-82,width,82,fill=1,stroke=0);c.setFillColor("white");c.setFont("Helvetica-Bold",24);c.drawString(48,height-45,"EyeZen");c.setFont("Helvetica",11);c.drawString(48,height-64,"AI-Assisted Diabetic Retinopathy Screening Report")
 y=height-108;y=heading("Patient Information",y)
 for label,value in [("Patient Name",patient.get("name")),("Age",patient.get("age")),("Record ID",patient.get("recordId")),("Screening Date",datetime.now().strftime("%d %b %Y, %H:%M"))]:y=row(label,value,y)
 y-=8;y=heading("Screening Summary",y)
 recommendation="Refer for ophthalmic evaluation." if result["decision"]=="REFER" else "Routine follow-up according to local screening protocol."
 for label,value in [("Predicted DR Grade",f"Grade {result['grade']} - {result['gradeLabel']}"),("Grade Confidence",f"{result['gradeConfidence']:.1%}"),("Referable DR Probability",f"{result['referableProbability']:.1%}"),("Referral Threshold",f"{result['referralThreshold']:.1%}"),("Screening Recommendation",recommendation)]:y=row(label,value,y)
 y-=8;y=heading("Grade Probabilities",y)
 for i,label in enumerate(CLASSES): y=row(f"Grade {i} - {label}",f"{result['gradeProbabilities'].get(f'grade{i}',0):.1%}",y)
 c.setFont("Helvetica",8);c.setFillColor("#526170");c.drawString(48,32,"Screening decision-support prototype only; not a clinical diagnosis.");c.drawRightString(width-48,32,"Page 1 of 2");c.showPage()
 c.setFillColor("#17283b");c.setFont("Helvetica-Bold",18);c.drawString(48,height-52,"EyeZen - Explainable AI")
 c.setFont("Helvetica-Bold",12);c.drawString(48,height-82,"Lesion Localization - U-Net");c.drawImage(ImageReader(str(lesion_path)),48,390,width=235,height=235,preserveAspectRatio=True);c.setFont("Helvetica",9);c.drawString(48,370,"U-Net highlights regions predicted to contain pathological retinal lesions.")
 c.setFont("Helvetica-Bold",12);c.drawString(325,height-82,"Classifier Attention - Grad-CAM");c.drawImage(ImageReader(str(gradcam_path)),325,390,width=235,height=235,preserveAspectRatio=True);c.setFont("Helvetica",9);c.drawString(48,340,"Grad-CAM indicates regions that most influenced the classifier's referable-DR decision and should not be interpreted as") ;c.drawString(48,326,"precise lesion localization.")
 c.setFont("Helvetica-Bold",12);c.setFillColor("#17283b");c.drawString(48,285,"Responsible Use");c.setFont("Helvetica",10);c.setFillColor("#101820");c.drawString(48,266,"Screening decision-support prototype only; not a clinical diagnosis.");c.setFillColor("#526170");c.setFont("Helvetica",8);c.drawRightString(width-48,32,"Page 2 of 2");c.save()
app=FastAPI(title="EyeZen Inference Service");app.mount("/images",StaticFiles(directory=OUT/"images"),name="images");app.mount("/reports",StaticFiles(directory=OUT/"reports"),name="reports")
@app.on_event("startup")
def startup():
 if not INTERNAL_ML_SECRET and not ALLOW_UNAUTHENTICATED_LOCAL_INFER:raise RuntimeError("INTERNAL_ML_SECRET is required; set ALLOW_UNAUTHENTICATED_LOCAL_INFER=true only for explicit local development.")
 init_models()
@app.get("/health")
def health():return {"status":"ok","mode":"DEMO" if DEMO_MODE else "REAL","classifierLoaded":status["classifierLoaded"],"segmentationLoaded":status["segmentationLoaded"],"calibrationLoaded":status["calibrationLoaded"],"storage":"signed-vercel-blob"}
def require_internal_secret(x_eyezen_internal_secret:str|None=Header(None)):
 if ALLOW_UNAUTHENTICATED_LOCAL_INFER:return
 if not INTERNAL_ML_SECRET or not x_eyezen_internal_secret or not secrets.compare_digest(x_eyezen_internal_secret,INTERNAL_ML_SECRET):raise HTTPException(403,"Forbidden")
@app.post("/infer",dependencies=[Depends(require_internal_secret)])
async def infer(image:UploadFile|None=File(None),imageUrl:str=Form(""),assetUploads:str=Form(""),assetPaths:str=Form(""),storageMode:str=Form("local"),patientName:str=Form(""),recordId:str=Form(""),age:str=Form("")):
 if imageUrl:
  payload=fetch_private_image(imageUrl)
 elif image is not None:
  suffix=Path(image.filename or "").suffix.lower();mime=(image.content_type or "").lower()
  if suffix not in (".jpg",".jpeg",".png") and mime not in ("image/jpeg","image/jpg","image/png","image/x-png"):raise HTTPException(400,"Unsupported file type. Upload a JPG or PNG image.")
  payload=await image.read()
 else:raise HTTPException(400,"An uploaded image is required.")
 try:original=Image.open(io.BytesIO(payload)).convert("RGB")
 except Exception as exc:raise HTTPException(400,"Unable to decode the uploaded image.") from exc
 cropped=crop_fundus(original);cls=np.asarray(cropped.resize((300,300)),dtype=np.float32).clip(0,255);seg=np.asarray(cropped.resize((384,384)),dtype=np.float32).clip(0,255)
 if DEMO_MODE:raise HTTPException(503,"DEMO_MODE is enabled; real V4 screening is required for this deployment.")
 try:
  raw_grade,raw_binary=normalize_classifier_outputs(classifier.predict(cls[None],verbose=0));lesion_probability=segmenter.predict(seg[None],verbose=0)[0,:,:,0]
  if not np.isfinite(lesion_probability).all():raise RuntimeError("Segmentation returned invalid values")
  mask=lesion_probability>=float(deployment["segmentation"]["threshold"]);heatmap=generate_gradcam(cls[None])
 except Exception as exc:log.exception("Inference failed");raise HTTPException(500,"Inference failed. See service logs for details.") from exc
 result=build_inference_result(raw_grade,raw_binary);uid=uuid.uuid4().hex;base=seg.astype(np.uint8);lesion=OUT/"images"/f"{uid}-lesion.png";cam=OUT/"images"/f"{uid}-gradcam.png";save_png(make_overlay(base,mask),lesion);heat=cv2.applyColorMap((cv2.resize(heatmap,(384,384)).clip(0,1)*255).astype(np.uint8),cv2.COLORMAP_JET);save_png(cv2.addWeighted(base,.58,heat,.42,0),cam);report=OUT/"reports"/f"{uid}-report.pdf";create_report(report,result,lesion,cam,{"name":patientName,"recordId":recordId,"age":age})
 if storageMode=="vercel-blob":
  try:
   uploads=json.loads(assetUploads);paths=json.loads(assetPaths);put_scoped(uploads["lesionUploadUrl"],lesion,"image/png");put_scoped(uploads["gradcamUploadUrl"],cam,"image/png");put_scoped(uploads["reportUploadUrl"],report,"application/pdf");result.update({"lesionBlobPath":paths["lesionBlobPath"],"gradcamBlobPath":paths["gradcamBlobPath"],"reportBlobPath":paths["reportBlobPath"],"lesionOverlayUrl":None,"gradcamUrl":None,"gradCamUrl":None,"reportUrl":None})
  finally:
   for path in (lesion,cam,report):path.unlink(missing_ok=True)
 else:result["lesionOverlayUrl"]="/images/"+lesion.name;result["gradcamUrl"]=result["gradCamUrl"]="/images/"+cam.name;result["reportUrl"]="/reports/"+report.name
 return result
