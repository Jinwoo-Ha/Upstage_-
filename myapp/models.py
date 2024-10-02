from django.db import models

class UserInfo(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    country_of_interest = models.CharField(max_length=100)
    interests = models.CharField(max_length=200)
    voice_type = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.name}'s Info"

class Story(models.Model):
    user_info = models.ForeignKey(UserInfo, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    content = models.TextField()
    voice_type = models.CharField(max_length=10)  # 새로 추가된 필드
    created_at = models.DateTimeField(auto_now_add=True)  # 새로 추가된 필드

    def __str__(self):
        return self.title