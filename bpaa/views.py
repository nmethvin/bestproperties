from django.shortcuts import render
from django.http import JsonResponse
from bpaa.scripts.build_model import run_model as return_run_model


def index(request):
    data_list = []

    if request.method == 'POST':
        data_list = return_run_model(request)

    context = {'data_list': data_list}

    return render(request, 'index.html', context)


def run_model(request):
    if request.method == 'POST':
        max_price = request.POST.get('max_price')
        min_price = request.POST.get('min_price')
        max_distance = request.POST.get('max_distance')
        built_after = request.POST.get('built_after')
        built_before = request.POST.get('built_before')
        max_beds = request.POST.get('max_beds')
        min_beds = request.POST.get('min_beds')
        max_bath = request.POST.get('max_bath')
        min_bath = request.POST.get('min_bath')
        max_sqft = request.POST.get('max_sqft')
        min_sqft = request.POST.get('min_sqft')
        num_of_results = request.POST.get('num_of_results')

        data = return_run_model(
            request,
            max_price=max_price,
            min_price=min_price,
            max_distance=max_distance,
            built_after=built_after,
            built_before=built_before,
            max_beds=max_beds,
            min_beds=min_beds,
            max_bath=max_bath,
            min_bath=min_bath,
            max_sqft=max_sqft,
            min_sqft=min_sqft,
            num_of_results=num_of_results,
        )
        return render(request, 'index.html', {'data_list': data})
