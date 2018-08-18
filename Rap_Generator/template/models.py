from django.db import models

class Template(models.Model):
    name = models.CharField(max_length=30)
    def _str_(self):
        return self.name


class Verse(models.Model):
    template = models.ForeignKey(Template)
    def _str_(self):
        return self.template.name


class Sentence(models.Model):
    wordCount = models.IntegerField()
    rhyme_pinyin = models.CharField(max_length=30)
    rhyme_type = models.CharField(max_length=30)
    verse = models.ForeignKey(Verse)
    def _str_(self):
        return self.verse.name

