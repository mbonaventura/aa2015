# -*- coding: utf-8 -*- 
import re
import sys
import string
import codecs

infile = sys.argv[1]
hfile = codecs.open(infile, encoding='utf-8')
		
#	
stop_words_list = []
swfile = sys.argv[2]
with open(swfile, 'r') as sfile:
    for line in sfile.readlines():
        stop_words_list.append(line.strip())
#
#print stop_words_list
#
line =""
for line in hfile.readlines():
	line = line.strip().lower()
	p = line.split(';')
	
	nosw_body = ''
	words = p[3].split()
	for w in words:
		if w not in stop_words_list:
			nosw_body+= w + ' '
	#
	nosw_title = ''
	words = p[4].split()
	for w in words:
		if w not in stop_words_list:
			nosw_body+= w + ' '
	#
	try:
		print p[0], ';', p[1], ';', p[2], ';', nosw_body, ';', nosw_title
	except:
		pass
	#print line


