from django.shortcuts import render, redirect

def home(request):
    if request.user.is_authenticated:
        return redirect("tenders:browse")
    return render(request, "core/index.html")

def learn(request):
    return render(request, "core/learn.html")