from django.contrib import admin
from .models import Vocab
from .models import CoOccurrence


class Admin(admin.ModelAdmin):
    list_display = ('word1_id', 'word2_id', 'frequency')


admin.site.register(Vocab)
admin.site.register(CoOccurrence, Admin)