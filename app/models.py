from django.db import models

class imagess(models.Model):
    image = models.ImageField(upload_to='images/')


class AudioFiles(models.Model):
    audio_file = models.FileField(upload_to='audio/')
    uploaded_at = models.DateTimeField(auto_now_add=True)



class TextFiles(models.Model):
    text_file = models.FileField(upload_to='text/', null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)




class UserResponse(models.Model):
    response = models.TextField()











class Content(models.Model):
    name = models.CharField(max_length=255)
    content = models.TextField()





class UploadedFiles(models.Model):
    student_no = models.CharField(max_length=255 , default='')
    file_upload = models.FileField(upload_to='documents/')
    upload_name = models.CharField(max_length=255)





class Marks(models.Model):
    name = models.CharField(max_length=255)
    mark = models.CharField(max_length=50)
    remark = models.CharField(max_length=255)
    code = models.CharField(max_length=255 , default='')




class MP4Details(models.Model):
    name = models.CharField(max_length=255)
    video_file = models.FileField(upload_to='videos/')




class PPTXDetails(models.Model):
    pptx_file = models.FileField(upload_to='ppt/')

