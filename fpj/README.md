# LDA

### Fitting

    from corpus import Corpus
    from lda import LDA
    menu_path = 'input/'

    corpus = Corpus()
    corpus.load_ldac(menu_path + 'reuters.ldac')
    model = LDA(n_topic=20)
    model.fit(corpus, n_iter=50)


### Results
    corpus.load_vocabulary(menu_path + 'reuters.tokens')
    corpus.load_context(menu_path + 'reuters.titles')

    topic_word = model.topic_word(n_top_word=10, corpus=corpus)
    print topic_word

    document_topic = model.document_topic(n_top_topic=1, corpus=corpus, limit=10)
    print document_topic


##### Topic words
    ['several', 'us', 'age', 'russia', 'minister', 'born', 'white', 'three', 'teresa', 'bishop']
    ['father', 'off', 'public', 'prime', 'monday', 'news', 'own', 'spokesman', 'say', "n't"]
    ['surgery', 'long', 'ceremony', 'month', 'thursday', 'french', 'head', 'marriage', 'house', 'gave']
    ['son', 'house', 'under', 'surgery', 'next', 'mass', 'percent', 'off', 'known', 'three']
    ['marriage', 'french', 'thursday', 'month', 'ceremony', 'head', 'away', 'reports', 'queen', 'political']
    ['long', 'show', 'operation', 'marriage', 'head', 'french', 'thursday', 'month', 'visit', 'nation']
    ['including', 'film', 'while', 'week', 'left', 'work', 'royal', 'us', 'reuters', 'clinton']
    ['germany', 'show', 'visit', 'operation', 'marriage', 'head', 'french', 'thursday', 'paris', 'began']
    ['paul', 'long', 'ceremony', 'month', 'thursday', 'french', 'head', 'queen', 'best', 'clinton']
    ['million', 'parker', 'died', 'cardinal', 'officials', 'united', 'among', 'newspaper', 'public', 'versace']
    ['thursday', 'french', 'head', 'marriage', 'operation', 'under', 'capital', 'among', 'minister', 'party']
    ['became', 'very', 'say', 'reporters', 'leader', 'roman', 'later', 'diana', 'city', 'police']
    ['son', 'award', 'spokesman', 'own', 'became', 'news', 'days', 'russian', 'says', 'death']
    ['germany', 'days', 'news', 'ago', 'became', 'own', 'spokesman', 'says', 'media', 'versace']
    ['newspaper', 'britain', 'among', 'four', 'funeral', 'whether', 'successor', 'become', 'began', 'local']
    ['own', 'spokesman', 'never', 'head', 'american', 'couple', 'week', 'private', 'took', 'conference']
    ['went', 'national', 'pontiff', 'died', 'support', 'god', 'heart', 'moscow', 'cunanan', 'although']
    ['leader', 'called', 'off', 'political', 'century', 'day', 'held', 'church', 'paris', 'days']
    ['left', 'catholic', 'church', 'russian', 'newspaper', 'wife', 'royal', 'britain', 'women', 'former']
    ['days', 'love', 'news', 'ago', 'became', 'own', 'spokesman', 'says', 'mass', 'italian']

##### Document topic
    ('0 UK: Prince Charles spearheads British royal revolution. LONDON 1996-08-20', (99,))
    ('1 GERMANY: Historic Dresden church rising from WW2 ashes. DRESDEN, Germany 1996-08-21', (22,))
    ("2 INDIA: Mother Teresa's condition said still unstable. CALCUTTA 1996-08-23", (1,))
    ('3 UK: Palace warns British weekly over Charles pictures. LONDON 1996-08-25', (225,))
    ('4 INDIA: Mother Teresa, slightly stronger, blesses nuns. CALCUTTA 1996-08-25', (53,))
    ("5 INDIA: Mother Teresa's condition unchanged, thousands pray. CALCUTTA 1996-08-25", (58,))
    ('6 INDIA: Mother Teresa shows signs of strength, blesses nuns. CALCUTTA 1996-08-26', (93,))
    ("7 INDIA: Mother Teresa's condition improves, many pray. CALCUTTA, India 1996-08-25", (3,))
    ('8 INDIA: Mother Teresa improves, nuns pray for "miracle". CALCUTTA 1996-08-26', (0,))
    ('9 UK: Charles under fire over prospect of Queen Camilla. LONDON 1996-08-26', (0,))
