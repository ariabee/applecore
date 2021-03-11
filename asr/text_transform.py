'''
Class that maps the data from the Mel Spectrogram to integer labels
'''

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        a 0
        b 1
        c 2
        d 3
        e 4
        f 5
        g 6
        h 7
        i 8
        j 9
        k 10
        l 11
        m 12
        n 13
        o 14
        p 15
        q 16
        r 17
        s 18
        t 19
        u 20
        v 21
        w 22
        x 23
        y 24
        z 25
        ' 26
        , 27
        . 28
        <SPACE> 29
        " 30
        - 31
        ; 32
        ? 33
        ! 34
        : 35
        ( 36
        ) 37
        [ 38
        ] 39
        { 40
        } 41
        — 42
        \ 43
        / 44
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')
