# -*- coding: utf-8 -*- 
import re
import sys
import string
import codecs
reload(sys)

def clean(line):
	line = re.sub(u'\xe1', "a", line)
	line = re.sub(u'\xe9', "e", line)
	line = re.sub(u'\xed', "i", line)
	line = re.sub(u'\xf3', "o", line)
	line = re.sub(u'\xfa', "u", line)
	line = re.sub(u'\xfc', "", line)
	line = re.sub(u'\xf1', "n", line)
	line = re.sub(u'\xab', "", line)
	line = re.sub(u'\xa1', "", line)
	line = re.sub(u'\xbb', "", line)
	line = re.sub(u'\xbf', "", line)
	line = re.sub(u'\xe8', "", line)
	line = re.sub(u'\xb0', "", line)
	line = re.sub(u'\xb4', "", line)
	line = re.sub(u'\xf6', "", line)	
	line = re.sub(u'\xdf', "", line)
	line = re.sub(u'\xef', "", line)
	line = re.sub(u'\xe2', "", line)
	line = re.sub(u'\xe0', "", line)			
	line = re.sub(u'\xec', "", line)	
	line = re.sub(u'\xad', "", line)
	line = re.sub(u'\xba', "", line)
	line = re.sub(u'\xaa', "", line)
	line = re.sub(u'\xea', "", line)
	line = re.sub(u'\xe4', "", line)
	line = re.sub(u'\xf2', "", line)
	line = re.sub(u'\xe7', "", line)
	line = re.sub(u'\xe3', "", line)				
	line = re.sub(u'\xf4', "", line)
	line = re.sub(u'\xb7', "", line)
	line = re.sub(u'\xa8', "", line)
	line = re.sub(u'\xeb', "", line)
	line = re.sub(u'\xfb', "", line)
	line = re.sub(u'\xae', "", line)		
	line = re.sub(u'\xf0', "", line)
	line = re.sub(u'\x81', "", line)	
	line = re.sub(u'\u2013', "", line)
	line = re.sub(u'\u2026', "", line)
	line = re.sub(u'\u2019', "", line)													
	line = re.sub(r'"',' ', line)
	#
	line = re.sub(r',',' ', line)
	line = re.sub(r'\.',' ', line)
	line = re.sub(r'-',' ', line)
	line = re.sub(r'\(',' ', line)
	line = re.sub(r'\)',' ', line)
	line = re.sub(r':',' ', line)
	line = re.sub(r'%',' ', line)
	line = re.sub(r'\'',' ', line)
	line = re.sub(r'\?',' ', line)
	line = re.sub(r'\¿',' ', line)	
	line = re.sub(r'\$',' ', line)	
	line = re.sub(r'&',' ', line)
	line = re.sub(r'  ',' ', line)
	return line

sys.setdefaultencoding('utf-8')
infile = sys.argv[1]
hfile = codecs.open(infile, encoding='utf-8')
		
#	
words = {}
line =""
for line in hfile.readlines():
	#line = decode_utf8(line)
	line = line.decode('utf-8')
	line = line.strip().lower()
	line = clean(line)
	print line
	#
	#for w in line.split():
	#	words[w]=1
	#
	#print clean(line.lower())
	#print mysplit(line)

#for w in words:
#	print w	