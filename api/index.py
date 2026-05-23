import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai

# تهيئة تطبيق FastAPI باسم app (وهو ما تبحث عنه Vercel)
app = FastAPI(title="HETAT AI Productivity Engine")

# إعدادات الـ CORS لتشغيل السيرفر أونلاين بأمان
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# قراءة مفتاح الـ API مباشرة من الـ Secrets الخاصة بـ Vercel
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- نماذج البيانات (Data Models) ---
class TaskItem(BaseModel):
    title: str = Field(..., description="عنوان المهمة باللغة العربية")
    time: str = Field(..., description="وقت المهمة")
    priority: str = Field(..., description="الأولوية: High, Medium, Low")
    category: str = Field(..., description="التصنيف: Work, Personal, Study, Health")

class SmartScheduleResponse(BaseModel):
    summary: str = Field(..., description="ملخص ذكي ومحفز لليوم باللغة العربية")
    tasks: List[TaskItem] = Field(..., description="قائمة المهام المستخرجة")

class UserPromptRequest(BaseModel):
    text: str

# --- نقاط الاتصال (Endpoints) ---
@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to HETAT AI Engine on Vercel"}

@app.post("/api/schedule", response_model=SmartScheduleResponse)
async def generate_smart_schedule(request: UserPromptRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing in Vercel Environment Variables.")
    
    try:
        prompt = f"""
        You are an expert AI productivity assistant. Analyze the user's raw text and transform it into a structured JSON schedule.
        
        User Text: "{request.text}"
        
        Respond ONLY with a valid JSON object matching this structure (Do not include markdown wrappers like ```json):
        {{
            "summary": "A short, motivating 1-sentence summary or advice in Arabic based on their tasks.",
            "tasks": [
                {{
                    "title": "Task title in Arabic",
                    "time": "Specific time or frame like '03:00 PM' or 'Flexible'",
                    "priority": "High or Medium or Low",
                    "category": "Work or Personal or Study or Health"
                }}
            ]
        }}
        """

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = await model.generate_content_async(prompt)
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        return SmartScheduleResponse.model_validate_json(response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
