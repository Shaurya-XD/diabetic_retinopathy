"""Run only with DEMO_MODE=false; synthetic input validates loading and tensor shapes."""
import os
os.environ['DEMO_MODE']='false'
import numpy as np
import app
def main():
 app.init_models();assert app.status=={'classifierLoaded':True,'segmenterLoaded':True,'summaryLoaded':True};assert tuple(app.classifier.input_shape[1:])==(300,300,3)
 raw=app.classifier.predict(np.zeros((1,300,300,3),dtype='float32'),verbose=0);assert raw.shape==(1,5) and np.isfinite(raw).all();assert np.isclose(app.calibrated_probabilities(raw[0]).sum(),1,atol=1e-5)
 assert tuple(app.segmenter.input_shape[1:])==(384,384,3);seg=app.segmenter.predict(np.zeros((1,384,384,3),dtype='float32'),verbose=0);assert seg.shape==(1,384,384,1) and np.isfinite(seg).all() and seg.min()>=0 and seg.max()<=1
 heatmap=app.gradcam(np.ones((1,300,300,3),dtype='float32')*127);assert heatmap.size>0 and np.isfinite(heatmap).all()
 lesion=app.OUT/'images'/'smoke-lesion.png';cam=app.OUT/'images'/'smoke-cam.png';from PIL import Image;Image.new('RGB',(384,384)).save(lesion);Image.new('RGB',(384,384)).save(cam);report=app.OUT/'reports'/'smoke-test.pdf';app.create_report(report,{'predictedGrade':0,'gradeLabel':'No DR','calibratedConfidence':1.,'referableProbability':0.,'referralDecision':'NON-REFER','referralThreshold':.3},lesion,cam);assert report.exists() and report.stat().st_size>0
 print('REAL SMOKE TEST PASSED')
if __name__=='__main__':main()
