import mongoose from 'mongoose';
const s=new mongoose.Schema({createdBy:{type:mongoose.Schema.Types.ObjectId,ref:'User',required:true,index:true},patient:{name:String,recordId:String,age:Number},imageUrl:String,inputBlobPath:String,lesionBlobPath:String,gradcamBlobPath:String,reportBlobPath:String,result:{type:mongoose.Schema.Types.Mixed,required:true}},{timestamps:true});
export default mongoose.model('Screening',s);
