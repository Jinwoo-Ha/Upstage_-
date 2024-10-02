from django.shortcuts import render, redirect
from .models import UserInfo, Story
from .story_generator import generate_story
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .chatbot import ask_chatbot, scenario

@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        response = ask_chatbot(user_input, scenario)
        return JsonResponse({'response': response})

def landing(request):
    return render(request, 'myapp/landing.html')

def storyinfo(request):
    if request.method == 'POST':
        user_info = UserInfo.objects.create(
            name=request.POST['name'],
            age=request.POST['age'],
            country_of_interest=request.POST['country'],
            interests=request.POST['interests'],
            voice_type=request.POST['voice_type']
        )
        return redirect('loading')
    return render(request, 'myapp/storyinfo.html')

def loading(request):
    user_info = UserInfo.objects.last()
    
    # 동화 생성 시작
    story = Story.objects.create(
        user_info=user_info,
        title=f"{user_info.name}의 {user_info.country_of_interest} 여행",
        content="",  # 초기에는 빈 내용으로 생성
        voice_type=user_info.voice_type
    )
    
    # 비동기적으로 동화 생성 시작
    generate_story_async(story.id, user_info)
    
    return render(request, 'myapp/loading.html', {'story_id': story.id})

def check_story_status(request, story_id):
    story = Story.objects.get(id=story_id)
    status = 'completed' if story.content else 'in_progress'
    return JsonResponse({'status': status})

def story(request, story_id):
    story = Story.objects.get(id=story_id)
    return render(request, 'myapp/story.html', {'story': story})

def end(request):
    return render(request, 'myapp/end.html')

# 비동기 동화 생성 함수
from django.core.cache import cache
from threading import Thread

def generate_story_async(story_id, user_info):
    def task():
        story_content = generate_story(
            name=user_info.name,
            age=user_info.age,
            country=user_info.country_of_interest,
            interests=user_info.interests
        )
        story = Story.objects.get(id=story_id)
        story.content = story_content
        story.save()
    
    Thread(target=task).start()