from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from .views import home, law, logic, predictt, translator , transcribe_audio , summarizer , chiku , chinu , caption_image , carrier , create_presentation , process_responses , text_tone ,  student_tool , user_login , report_card ,  question ,   display_all_marks ,  ai_grading ,   upload_file_ ,  download_resume_template , register  , save_text_file , upload_pdf , ocr_pdf , roadmap , extracted_text_detail , ats

urlpatterns = [
    path('home', home, name='home'),
    path('logic/', logic, name='logic'),
    path('law/', law, name='law'),
    path('translator/', translator, name='translator'),
    path('predict/', predictt, name='predict'),
    path('transcribe_audio/', transcribe_audio, name='transcribe_audio'),
    path('caption_image/', caption_image, name='caption_image'),
    path('carrier/', carrier, name='carrier'),
    path('process_responses/', process_responses, name='process_responses'),
    path('download_resume_template/<int:id>/', download_resume_template, name='download_resume_template'),
    path('register/', register, name='register'),
    path('save_text_file/', save_text_file, name='save_text_file'),
    path('upload_pdf/', upload_pdf, name='upload_pdf'),
    path('extracted_text_detail/<int:id>/', extracted_text_detail, name='extracted_text_detail'),
    path('ats/', ats, name='ats'),
    path('ocr_pdf/', ocr_pdf, name='ocr_pdf'),
    path('roadmap/', roadmap, name='roadmap'),
    path('upload_file/', upload_file_, name='upload_file'),
    path('ai_grading/', ai_grading, name='ai_grading'),
    path('display_all_marks/', display_all_marks, name='display_all_marks'),
    path('question/', question, name='question'),
    path('report_card/<int:id>/', report_card, name='report_card'),
    path('', user_login, name='user_login'),
    path('student_tool/', student_tool, name='student_tool'),
    path('text_tone/', text_tone, name='text_tone'),
    path('summarizer/', summarizer, name='summarizer'),
    path('create_presentation/', create_presentation, name='create_presentation'),
    path('chinu/', chinu, name='chinu'),
    path('chiku/', chiku, name='chiku'),
    
   
    

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)





