import mongoose from 'mongoose';
export default mongoose.model('User',new mongoose.Schema({name:{type:String,required:true},email:{type:String,unique:true,required:true,lowercase:true},password:{type:String,required:true}},{timestamps:true}));
