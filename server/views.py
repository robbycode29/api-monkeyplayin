from django.shortcuts import render
from django.http import JsonResponse

# Create your views here.
info_dict = {
    'team': 'MonkeyPlayin',
    'members': ['Robert', 'Denis', 'Mihail'],
    'technologies': ['Django', 'Angular', 'Scikit', 'Firestore', 'Docker'],
    'description': 'This API\'s purpose is to process and serve Game Recommendation data!',
    'data_format': 'JSON',
    'request_lib': 'axios',
    'request_example': 'axios.get(\'https://api-monkeyplayin.onrender.com/your_endpoint\').then(res => console.log(res.data))',
}

def info(request):
    return JsonResponse(info_dict)