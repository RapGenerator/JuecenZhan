"""Rap_Generator URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url,include
from django.contrib import admin
from template.views import get_template
from template.views import get_templates

from expansion.views import *



urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^template/(?P<id>[0-9]+)',get_template),
    url(r'^templates/',get_templates),
    url(r'^co_exist/', generate_coOcurrence),
    url(r'^crawler', generate_crawler)


    #(?P<str>[a-zA-Z]+
]
