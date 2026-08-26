"""RetinaView inference API. Model artifacts remain in this service."""
import io, json, logging, os, uuid
from datetime import datetime
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT=Path(__file__).parent; ART=Path(os.getenv('ARTIFACTS_DIR',ROOT/'artifacts')); OUT=Path(os.getenv('OUTPUT_DIR',ROOT/'runtime'))
for directory in (OUT/'images',OUT/'reports'): directory.mkdir(parents=True,exist_ok=True)
DEMO_MODE=os.getenv('DEMO_MODE','true').lower()=='true'; CLASSES=['No DR','Mild NPDR','Moderate NPDR','Severe NPDR','Proliferative DR']
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s'); log=logging.getLogger('retinaview.ml')
classifier=segmenter=summary=grad_model=None; status={'classifierLoaded':False,'segmenterLoaded':False,'summaryLoaded':False}

def artifact(key): return ART/os.getenv(key,{'CLASSIFIER_PATH':'aptos_efficientnetb0_final.keras','UNET_WEIGHTS_PATH':'idrid_lesion_unet_final.weights.h5','RUN_SUMMARY_PATH':'run_summary.json'}[key])
def load_summary():
 global summary
 with artifact('RUN_SUMMARY_PATH').open(encoding='utf-8') as f: summary=json.load(f)
 _=summary['classifier']['temperature'],summary['classifier']['referral_threshold'],summary['segmentation']['validation_threshold']
 status['summaryLoaded']=True; log.info('run_summary loaded: YES')
def init_models():
 global classifier,segmenter,grad_model
 load_summary()
 if DEMO_MODE: log.info('Inference mode: DEMO'); return
 try:
  import tensorflow as tf
  classifier=tf.keras.models.load_model(artifact('CLASSIFIER_PATH'),compile=False);status['classifierLoaded']=True;grad_model=build_gradcam_model(classifier);log.info('Classifier loaded: YES')
  from model_adapter import build_unet
  segmenter=build_unet();log.info('Segmentation model built: YES');segmenter.load_weights(artifact('UNET_WEIGHTS_PATH'));status['segmenterLoaded']=True;log.info('Segmentation weights loaded: YES');log.info('Inference mode: REAL')
 except Exception as exc:
  classifier=segmenter=grad_model=None;log.exception('REAL inference initialization failed: %s',exc);raise RuntimeError('REAL model loading failed; inspect the service log. Demo fallback was not used.') from exc

def crop_fundus(image):
 rgb=np.asarray(image.convert('RGB'));foreground=np.max(rgb,axis=2)>10;ys,xs=np.where(foreground)
 if not len(xs): return image.convert('RGB')
 pad=max(2,int(min(rgb.shape[:2])*.01));return Image.fromarray(rgb[max(0,ys.min()-pad):min(rgb.shape[0],ys.max()+pad+1),max(0,xs.min()-pad):min(rgb.shape[1],xs.max()+pad+1)])
def softmax(values):
 values=values-np.max(values);exp=np.exp(values);return exp/exp.sum()
def calibrated_probabilities(raw): return softmax(np.log(np.clip(raw,1e-8,1.0))/float(summary['classifier']['temperature']))
def find_backbone(model):
 """Returns the nested EfficientNet model that owns top_conv."""
 try:return model.get_layer('efficientnetb0')
 except ValueError:pass
 for layer in model.layers:
  if hasattr(layer,'layers'):
   try:
    layer.get_layer('top_conv');return layer
   except ValueError:
    found=find_backbone(layer) if layer.layers else None
    if found is not None:return found
 return None
def build_gradcam_model(model):
 """Replays the outer classifier head over a backbone-internal feature graph."""
 import tensorflow as tf
 backbone=find_backbone(model)
 if backbone is None:raise RuntimeError('Could not locate nested EfficientNetB0 backbone for Grad-CAM')
 try:last_conv=backbone.get_layer('top_conv')
 except ValueError as exc:raise RuntimeError('Could not locate top_conv in EfficientNetB0 backbone') from exc
 feature_extractor=tf.keras.Model(inputs=backbone.input,outputs=[last_conv.output,backbone.output],name='gradcam_feature_extractor')
 grad_input=tf.keras.Input(shape=model.input_shape[1:],name='gradcam_input')
 conv_features,backbone_features=feature_extractor(grad_input,training=False)
 x=model.get_layer('gap')(backbone_features)
 x=model.get_layer('head_dropout')(x,training=False)
 predictions=model.get_layer('grade')(x)
 return tf.keras.Model(inputs=grad_input,outputs=[conv_features,predictions],name='referable_gradcam_model')
def gradcam(classifier_input):
 import tensorflow as tf
 if grad_model is None:raise RuntimeError('Grad-CAM model has not been initialized')
 try:
  with tf.GradientTape() as tape:
   features,probabilities=grad_model(classifier_input,training=False);score=tf.reduce_sum(probabilities[:,2:5],axis=1)
  gradients=tape.gradient(score,features)
 except Exception as exc:raise RuntimeError(f'Grad-CAM execution failed: {exc}') from exc
 if gradients is None:raise RuntimeError('Grad-CAM gradients are None')
 weights=tf.reduce_mean(gradients,axis=(1,2));cam=tf.reduce_sum(features*weights[:,None,None,:],axis=-1)[0];cam=tf.nn.relu(cam);maximum=tf.reduce_max(cam)
 cam=tf.cond(maximum>0,lambda:cam/maximum,lambda:tf.zeros_like(cam));heatmap=cam.numpy()
 if not np.isfinite(heatmap).all():raise RuntimeError('Grad-CAM produced non-finite values')
 return heatmap
def demo_prediction(seg_image):
 gray=cv2.cvtColor(seg_image,cv2.COLOR_RGB2GRAY);_,mask=cv2.threshold(gray,70,255,cv2.THRESH_BINARY);mask=cv2.GaussianBlur(mask,(17,17),0)>120;seed=int(seg_image.mean())%5
 return np.roll(np.array([.03,.08,.22,.37,.30]),seed-3),mask,mask.astype('float32')
def save_png(image,path):Image.fromarray(image).save(path,'PNG');return '/images/'+path.name
def make_overlay(base,mask,color=(220,32,44)):
 layer=np.zeros_like(base);layer[mask]=color;return cv2.addWeighted(base,1,layer,.48,0)
def create_report(path,result,lesion_path,gradcam_path):
 c=canvas.Canvas(str(path),pagesize=letter);c.setTitle('RetinaView screening report');c.setFont('Helvetica-Bold',18);c.drawString(50,750,'RetinaView — Screening Review');c.setFont('Helvetica',10)
 lines=[f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}",f"DR severity prediction: Grade {result['predictedGrade']} — {result['gradeLabel']}",f"Calibrated confidence: {result['calibratedConfidence']:.1%}",f"Referable DR probability: {result['referableProbability']:.1%}",f"Referral decision: {result['referralDecision']} (threshold: {result['referralThreshold']:.4f})",'Screening decision-support prototype only; not a clinical diagnosis.','U-Net is lesion localization. Grad-CAM is classifier attention, not lesion localization.']
 y=720
 for text in lines:c.drawString(50,y,text);y-=18
 c.drawImage(ImageReader(str(lesion_path)),55,140,width=220,height=220,preserveAspectRatio=True);c.drawImage(ImageReader(str(gradcam_path)),330,140,width=220,height=220,preserveAspectRatio=True);c.drawString(55,125,'Lesion localization — U-Net');c.drawString(330,125,'Classifier attention — Grad-CAM');c.save()

app=FastAPI(title='RetinaView Inference Service');app.add_middleware(CORSMiddleware,allow_origins=os.getenv('CORS_ORIGINS','*').split(','),allow_methods=['*'],allow_headers=['*']);app.mount('/images',StaticFiles(directory=OUT/'images'),name='images');app.mount('/reports',StaticFiles(directory=OUT/'reports'),name='reports')
@app.on_event('startup')
def startup():init_models()
@app.get('/health')
def health():return {'service':'healthy',**status,'mode':'demo' if DEMO_MODE else 'real'}
@app.post('/infer')
async def infer(image:UploadFile=File(...)):
 filename=Path(image.filename or '').name;extension=Path(filename).suffix.lower();mime=(image.content_type or '').lower();valid_mime=mime in ('image/jpeg','image/jpg','image/png','image/x-png');valid_extension=extension in ('.jpg','.jpeg','.png')
 if not (valid_mime or valid_extension):raise HTTPException(400,'Unsupported file type. Upload a JPG or PNG image.')
 payload=await image.read();log.info('Upload received: filename=%s content_type=%s bytes=%d',filename,mime,len(payload))
 if not payload:raise HTTPException(400,'The uploaded image is empty.')
 try:original=Image.open(io.BytesIO(payload)).convert('RGB')
 except Exception as exc:raise HTTPException(400,'Unable to decode the uploaded image.') from exc
 cropped=crop_fundus(original);cls_image=np.asarray(cropped.resize((300,300)),dtype='float32').clip(0,255);seg_image=np.asarray(cropped.resize((384,384)),dtype='float32').clip(0,255)
 if DEMO_MODE: probabilities,mask,lesion_probability=demo_prediction(seg_image.astype('uint8'));grad_heatmap=lesion_probability
 else:
  raw=classifier.predict(cls_image[None],verbose=0)[0];probabilities=calibrated_probabilities(raw);lesion_probability=segmenter.predict(seg_image[None],verbose=0)[0,:,:,0]
  if not np.isfinite(lesion_probability).all() or lesion_probability.min()<0 or lesion_probability.max()>1:raise HTTPException(500,'Segmentation model returned invalid probabilities')
  mask=lesion_probability>=float(summary['segmentation']['validation_threshold']);grad_heatmap=gradcam(cls_image[None])
 grade=int(np.argmax(probabilities));referable=float(probabilities[2:].sum());referral_threshold=float(summary['classifier']['referral_threshold'])
 result={'predictedGrade':grade,'gradeLabel':CLASSES[grade],'calibratedConfidence':round(float(probabilities[grade]),5),'classProbabilities':[round(float(x),5) for x in probabilities],'referableProbability':round(referable,5),'referralThreshold':referral_threshold,'referralDecision':'REFER' if referable>=referral_threshold else 'NON-REFER','segmentationThreshold':float(summary['segmentation']['validation_threshold']),'modelMode':'demo' if DEMO_MODE else 'real','processing':{'classifierInput':'cropped RGB, 300x300, float32 0-255','segmentationInput':'cropped RGB, 384x384, float32 0-255'}}
 uid=uuid.uuid4().hex;base=seg_image.astype('uint8');lesion_file=OUT/'images'/f'{uid}-lesion.png';mask_file=OUT/'images'/f'{uid}-mask.png';cam_file=OUT/'images'/f'{uid}-gradcam.png'
 result['lesionOverlayUrl']=save_png(make_overlay(base,mask),lesion_file);result['lesionMaskUrl']=save_png(mask.astype('uint8')*255,mask_file);heat=cv2.applyColorMap((cv2.resize(grad_heatmap,(384,384))*255).astype('uint8'),cv2.COLORMAP_JET);result['gradCamUrl']=save_png(cv2.addWeighted(base,.58,heat,.42,0),cam_file)
 report=OUT/'reports'/f'{uid}-report.pdf';create_report(report,result,lesion_file,cam_file);result['reportUrl']='/reports/'+report.name;return result
