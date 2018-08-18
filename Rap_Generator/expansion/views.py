from django.shortcuts import render
from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json
from expansion.models import Vocab
from expansion.models import CoOccurrence
from .crawler import Crawler




def generate_coOcurrence(request):
    try:
        strr = request.GET.get('str')
        response = []
        try:
            str_id = Vocab.objects.get(word=strr).id
        except:
            return JsonResponse(response, safe=False)

        mylist = CoOccurrence.objects.filter(word1_id=str_id)
        response = []
        for item in mylist:
            response.append(item.word2_id.word)

        return JsonResponse(response, safe=False)

    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


def generate_crawler(request):
    try:
        key_word = request.GET.get('str')
        num = int(request.GET.get('num'))
        my_crawler = Crawler(key_word, num)
        res = my_crawler.generate_list()
        #res.append(my_crawler.wordnum)
        return JsonResponse(res, safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)









