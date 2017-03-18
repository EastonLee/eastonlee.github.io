import string
s = '''
Machine Learning by Stanford University
'''

s = s.strip().translate(None, string.punctuation).lower().replace(' ', '-')
print s
