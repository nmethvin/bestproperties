from django.shortcuts import render
from django.http import JsonResponse
from bpaa.scripts.build_model import run_model as return_run_model


def index(request):
    data_list = []

    if request.method == 'POST':
        # If the form is submitted, run the model
        data_list = return_run_model(request)

    context = {'data_list': data_list}

    return render(request, 'index.html', context)


def run_model(request):
    if request.method == 'POST':
        data = return_run_model(request)
        return render(request, 'index.html', {'data_list': [data]})
