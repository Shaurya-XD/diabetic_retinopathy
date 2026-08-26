import jwt from 'jsonwebtoken';
export default (req,res,next)=>{const token=req.headers.authorization?.replace('Bearer ','');try{req.user=jwt.verify(token,process.env.JWT_SECRET);next()}catch{res.status(401).json({message:'Sign in required'})}};
