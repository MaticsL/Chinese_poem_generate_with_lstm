import pypinyin


def init_stop_words():
    stop_words = set()
    f = open('stop_words.txt', 'r')
    for row in f.readlines():
        row = row.replace('\n', '')
        row = row.replace('\r', '')
        if f:
            stop_words.add(row)
    stop_words.add('\r')
    stop_words.add('\n')
    stop_words.add(' ')
    return stop_words


def get_training_data():
    encoding = 'gbk'
    text = open('training_poem.txt', encoding=encoding).read()
    return text


def check_rythm(r, c):
    if r == 0:
        return True
    r = abs(r)
    mapping = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
    }
    x = pypinyin.pinyin(c, style=pypinyin.TONE2)[0]
    x = x[0]
    if '1' in x:
        x = 1
    elif '2' in x:
        x = 2
    elif '3' in x:
        x = 3
    elif '4' in x:
        x = 4
    else:
        return True
    return mapping[x] == r
