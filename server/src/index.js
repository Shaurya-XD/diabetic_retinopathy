import dotenv from 'dotenv'; import {dirname,join} from 'node:path'; import {fileURLToPath} from 'node:url';
import express from 'express'; import cors from 'cors'; import mongoose from 'mongoose';
import authRoutes from './routes/auth.js'; import screeningRoutes from './routes/screenings.js';
const serverDir=dirname(fileURLToPath(import.meta.url));
dotenv.config({path:join(serverDir,'..','.env')});
const app=express();
app.use(cors({origin:process.env.CLIENT_ORIGIN||'http://localhost:5173'})); app.use(express.json());
app.use('/uploads',express.static('uploads')); app.get('/api/health',(_,res)=>res.json({ok:true}));
app.use('/api/auth',authRoutes); app.use('/api/screenings',screeningRoutes);
const mongoUri=process.env.MONGODB_URI;
if(!mongoUri){console.error('MONGODB_URI is required. Add the MongoDB Atlas connection string to server/.env.');process.exit(1)}
mongoose.connect(mongoUri,{dbName:'retinaview'}).then(()=>{console.log('MongoDB connected successfully');app.listen(process.env.PORT||5000,()=>console.log('API ready'))}).catch(e=>{console.error(`MongoDB connection failed: ${e.name}: ${e.message}`);process.exit(1)});
