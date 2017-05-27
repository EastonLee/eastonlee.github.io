import string
s = raw_input('input your post title:')
s = s.strip().translate(None, string.punctuation).lower().replace(' ', '-')
print s
