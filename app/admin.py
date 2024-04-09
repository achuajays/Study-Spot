from django.contrib import admin
from .models import UserResponse , UploadedFiles , Marks
# Register your models here.


admin.site.register(UserResponse)

admin.site.register(UploadedFiles)

admin.site.register(Marks)