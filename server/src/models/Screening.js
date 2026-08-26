import mongoose from 'mongoose';
const s=new mongoose.Schema({clinician:{type:mongoose.Schema.Types.ObjectId,ref:'User',required:true},patient:{name:String,recordId:String,age:Number},imageUrl:String,result:{type:mongoose.Schema.Types.Mixed,required:true}},{timestamps:true});
export default mongoose.model('Screening',s);
