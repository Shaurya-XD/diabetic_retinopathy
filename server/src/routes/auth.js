import {Router} from 'express'; import bcrypt from 'bcryptjs'; import jwt from 'jsonwebtoken'; import User from '../models/User.js';
const r=Router(), token=u=>jwt.sign({id:u._id,name:u.name},process.env.JWT_SECRET,{expiresIn:'7d'});
r.post('/register',async(req,res)=>{try{const {name,email,password}=req.body;if(!name||!email||password?.length<8)throw Error('Use a name, email and password of at least 8 characters');const u=await User.create({name,email,password:await bcrypt.hash(password,12)});res.status(201).json({token:token(u),user:{name:u.name,email:u.email}})}catch(e){res.status(400).json({message:e.code===11000?'Email already registered':e.message})}});
r.post('/login',async(req,res)=>{const u=await User.findOne({email:req.body.email?.toLowerCase()});if(!u||!await bcrypt.compare(req.body.password||'',u.password))return res.status(401).json({message:'Invalid email or password'});res.json({token:token(u),user:{name:u.name,email:u.email}})});
export default r;
