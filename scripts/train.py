from ultralytics import YOLO
import os

def run_training():
    # 1. טעינת המודל - משתמשים בגרסת ה-Pose הקטנה (Nano) כדי לקבל ביצועים מהירים
    # המודל יורד אוטומטית בפעם הראשונה
    model = YOLO('yolov8n-pose.pt')

    # 2. הגדרת נתיב לקובץ הנתונים
    # וודא שהקובץ data.yaml נמצא בנתיב הזה בתוך הפרויקט שלך
    data_path = 'data/freestyle/data.yaml'

    # 3. תחילת תהליך האימון (Fine-tuning)
    # סקריפט זה רלוונטי גם ל-Level 3 - פשוט מחליפים את התוכן של תיקיית הדאטה
    model.train(
        data=data_path,
        epochs=100,              # 100 סבבים מספיקים לשיפור משמעותי ב-Fine-tuning
        imgsz=640,               # גודל תמונה סטנדרטי לאימון YOLO
        batch=16,                # גודל Batch אופטימלי ל-15GB VRAM של ה-Tesla T4
        device=0,                # הוראה מפורשת להשתמש ב-GPU (CUDA:0)
        project='runs/train',    # תיקיית היעד לתוצאות
        name='swimming_finetuning', # שם הניסוי
        save=True,               # שמירת קבצי ה-Weights (best.pt) בסיום
        plots=True,              # יצירת גרפים של Loss ודיוק לצורך הדו"ח
        workers=4                # שימוש ב-Multi-processing לטעינת דאטה מהירה
    )

if __name__ == "__main__":
    run_training()