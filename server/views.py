from aquilify.shortcuts import render

# Define all your views here.

async def homeview(request):
    return await render(request, "index.html")