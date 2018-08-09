1. Given a keyWord, we can expand this keyword to n keywords based on co-occurrence method.

2. What is our Co-occurrence method?

Given a source file, we create a large dictionary which has all the words in the file as the key (denoted as "wordA", and for each key, we have a co-occurrence dictionary as the value. For the co-occurence dictionary, the key is top n most frequently words that occur in the same sentence containing "wordA".

3. Result

vocabTable.csv: id -> word mapping

co-occurrence_n.csv: (word1_id, word2_id, co-occurrence_count)