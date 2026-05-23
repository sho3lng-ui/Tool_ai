import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv

# شحن متغيرات البيئة (الملف الذي يحتوي على مفاتيح الـ API Secret)
load_dotenv()

# تهيئة تطبيق FastAPI
app = FastAPI(
    title="HETAT AI Productivity Engine",
    description="The core AI engine for task scheduling and smart notes"
)

# السماح للـ Frontend بالاتصال بالـ Backend (CORS) دون مشاكل أمنية
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # في البيئة الإنتاجية استبدلها برابط الـ Frontend الخاص بك
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إعداد مفتاح الذكاء الاصطناعي (Gemini كمثال لقوته ومجانيته في التطوير)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables.")

# --- نماذج البيانات (Data Models) ---

# هيكل المهمة الواحدة التي سيستخرجها الذكاء الاصطناعي
class TaskItem(BaseModel):
    title: str = Field(..., description="عنوان المهمة الواضح")
    time: str = Field(..., description="وقت المهمة أو الموعد النهائي، مثال: 02:00 PM أو Tomorrow")
    priority: str = Field(..., description="الأولوية: High, Medium, Low")
    category: str = Field(..., description="التصنيف: Work, Personal, Study, Health")

# هيكل الرد النهائي الذي يعود للمستخدم
class SmartScheduleResponse(BaseModel):
    summary: str = Field(..., description="ملخص ذكي ومحفز لليوم")
    tasks: List[TaskItem] = Field(..., description="قائمة المهام المستخرجة والمنظمة")

# هيكل الطلب القادم من المستخدم
class UserPromptRequest(BaseModel):
    text: str

# --- نقاط الاتصال (Endpoints) ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "Welcome to HETAT AI Engine"}

@app.post("/api/schedule", response_model=SmartScheduleResponse)
async def generate_smart_schedule(request: UserPromptRequest):
    """
    تأخذ هذه النقطة النص العشوائي من المستخدم وتحوله إلى جدول مهام منظم ومصنف بالذكاء الاصطناعي
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="AI Service configuration missing on server.")
    
    try:
        # صياغة الـ Prompt بدقة ليعيد النتيجة كـ JSON متوافق تماماً مع الهيكل المطلوب
        prompt = f"""
        You are an expert AI productivity assistant. Your job is to analyze the user's raw, unstructured text about their day, notes, or tasks, and transform it into a structured JSON schedule.
        
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

        # استدعاء نموذج ذكاء اصطناعي سريع وخفيف (Flash) لضمان استجابة فورية وعدم انهيار السيرفر
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = await model.generate_content_async(prompt)
        
        # تنظيف النص والتأكد من أنه JSON صالح
        response_text = response.text.strip()
        if response_text.startswith("
```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        # تحويل النص إلى كائن مسترجع والتحقق من صحته تلقائياً عبر Pydantic
        return SmartScheduleResponse.model_validate_json(response_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")
