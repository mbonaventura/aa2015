# -*- coding: utf-8 -*- 
import re
import sys
import string
import codecs
from nltk import word_tokenize
from nltk.stem import SnowballStemmer


stemmer = SnowballStemmer('spanish')

infile = sys.argv[1]
hfile = codecs.open(infile, encoding='utf-8')
		
#	
line =""
for line in hfile.readlines():
	line = line.strip().lower()
	p = line.split(';')
	
	stemmed_body = ''
	words = p[3].split()
	for w in words:
		stemmed_body+= stemmer.stem(w) + ' '
	#
	stemmed_title = ''
	words = p[4].split()
	for w in words:
		stemmed_title+= stemmer.stem(w) + ' '
	#
	try:
		print p[0], ';', p[1], ';', p[2], ';', stemmed_body, ';', stemmed_title
	except:
		pass
	#print line


