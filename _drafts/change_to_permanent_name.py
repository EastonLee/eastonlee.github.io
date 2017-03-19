import string
s = '''
FuXi (reasoning engine) internal
'''

s = s.strip().translate(None, string.punctuation).lower().replace(' ', '-')
print s
