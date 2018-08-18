from django.db import models


# Create your models here.
class Vocab(models.Model):
    word = models.CharField(max_length=30)

    def __str__(self):
        return self.word
    class Meta:
        db_table = 'vocabs'



class CoOccurrence(models.Model):
    word1_id = models.ForeignKey(Vocab, related_name='word1')
    word2_id = models.ForeignKey(Vocab, related_name='word2')
    frequency = models.IntegerField()
    class Meta:
        db_table = 'co_occurrences'