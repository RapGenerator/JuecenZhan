from django.contrib import admin
from .models import Template, Sentence, Verse

# class templateAdmin(admin.ModelAdmin):
#     list_display = ('template_name',)


# class verseAdmin(admin.ModelAdmin):
#     list_display = ('template_name',)
#
#     def template_name(self, instance):
#         return instance.template.name


class sentenceAdmin(admin.ModelAdmin):
    list_display = ('wordCount', 'rhyme_pinyin', 'rhyme_type','verse')


admin.site.register(Template)
admin.site.register(Sentence, sentenceAdmin)
admin.site.register(Verse)