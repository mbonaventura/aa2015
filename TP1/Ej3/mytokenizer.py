#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import re, sys, os
import codecs

def mysplit(line):
	list_puntuation  = [',', '!', '\?', '"', ';', '\^', '\*', '\$', '%', '&', "=", '\(', '\)', 'ó']
	list_replacement = [',', '!',  '?', '"', ';', ' ^', ' *', ' $', '%', '&', "=",  '(', ' )', 'a']
	
	ln = 0
	for s in list_puntuation:		
		line = re.sub(s, " "+list_replacement[ln] + " ", line)
		ln+=1
		
	return line.split()

def tokenize(line):
	
	list = mysplit(line)
	
	tmp_tokens = []
	
	for item in list:  
		# Contracciones: can't,  I'll, etc.
		if (re.search('\'', item)):
			if item == "can't":			# Caso: can't
				tmp_tokens.append("can")
				tmp_tokens.append("n't")                
			elif item == "won't":			# Caso: won't
				tmp_tokens.append("will")
				tmp_tokens.append("n't")
			elif re.search("n't",item):		# Casos: wouldn't, shouldn't, etc.
				tmp_tokens.append(item[:-3])
				tmp_tokens.append(item[-3:])				
			elif re.search("'",item):		# Mas palabras con apostrofe ('s, 'll, etc.)
				wordpart = item.split("'")
				tmp_tokens.append(wordpart[0])				
				tmp_tokens.append("'" + wordpart[1])				
			else:
				tmp_tokens.append(item)
		else:			
			tmp_tokens.append(item)
		
	#-------------------------------------------------------------------------------------------------

	tkn = 0
	final_tokens = []
	for item in tmp_tokens:		
		
		# Expresiones regulares para fechas y numeros
		pattern_date = "[0-9]+(\/[0-9]+)+"
		pattern_percent = "([+/-])?[0-9]+()?[0-9]*%"
		pattern_numbers = "(\[0-9]+.?)?+(\.[0-9]+[0-9]+)*"
		
		# Expresiones regulares para abreviaturas		
		pattern_abr1="[A-Za-z]\.([A-Za-z0-9]\.)+"
		pattern_abr2="[A-Z][bcdfghjklmnpqrstvwxyz]+\."
		pattern_abr3="[\.]([,?!:;])"
				
		# Expresiones regulares para direcciones electronicas
		pattern_url = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"		
		pattern_ips = "(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})\.(?:[\d]{1,3})"						
		pattern_email = "[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+"
	
		# Otras RE
		pattern_multiword = "[A-Za-z]-([A-Za-z])+"
			
		if (item == ""):
			break
						
		if (re.search(pattern_email, item)):	# Direccion de mail		
			final_tokens.append(item)		
			
		elif (re.search(pattern_ips, item)):	# Direccion IP
			final_tokens.append(item)		
			
		elif (re.search(pattern_url, item)):	# URL http				
			final_tokens.append(item)
			
		elif (re.search(pattern_multiword, item)):	
			# Multiwords
			item = re.sub('\.', '', item)
			final_tokens.append(item)
			
		elif (re.search('[.]', item)):			
			
			if (re.search(pattern_percent, item)): # Numero con %					
				item = re.sub('%', '', item)
				final_tokens.append(item)
				final_tokens.append("%")
				
			if (re.search(pattern_abr3, item)):	# Abreviaturas			
				final_tokens.append(item)
				
			elif (re.search(pattern_abr2, item)):				
				final_tokens.append(item)
				
			elif (re.search(pattern_abr1, item)):				
				final_tokens.append(item)
				
			elif (re.search('[.]$', item)):
				add_period = 0			
				if (tkn < len(tmp_tokens)-1):
					nexttk = tmp_tokens[tkn+1]				
					if (re.match('[a-z]', nexttk[0])):	# La siguiente es minuscula. Lo interpreto como abreviatura => Le dejo el punto						
						pass
					if (re.match('[A-Z]', nexttk[0])):	# La siguiente es mayuscula. 
						if (re.match('[A-Z]', item[0])): 	
							# Esta es mayuscula => abreviatura (que se "escapo" de las reglas anteriores) => No hago nada
							pass
						else:
							# Lo interpreto como palabra al final de la oracion. Le quito el punto.
							item = re.sub('\.', '', item)
							add_period = 1
					else:
						# Es la ultima palabra de la linea procesada. Lo interpreto como palabra al final de la oracion. Le quito el punto.
						item = re.sub('\.', '', item)
						add_period = 1
				else:
					# No tengo la siguiente palabra => Lo interpreto como "palabra final" (no abreviatura) => Le quito el punto.
					item = re.sub('\.', '', item)
					add_period = 1
								
				final_tokens.append(item)
				if (add_period):
					final_tokens.append(".")
					
				
			else:
				# Es otra cosa que no pude reconocer
				final_tokens.append(item)
				
		else:			
			#item = item.lower()					
			final_tokens.append(item)
			pass

		tkn+=1                
	return final_tokens
#
#
#
if __name__ == '__main__':

	infile = sys.argv[1]
	#hfile = open(infile, 'r')	
	
	hfile = codecs.open(infile, encoding='utf-8')
		




	#------------------------------------------------------------------------------
	line = ""
	xline = "."
	tokens_sum = {}	
	tokens_add = []
	#	
	for line in hfile.readlines():
		line = line.rstrip()
		print "LINE: " + line



		line = line.decode('utf-8')

		print mysplit(line)


		#@sys.exit()
		 

		
		p = line.split(';')

		print '---------------------------------------------------------------------------------'
		print '#1\t', p[0]
		print '#2\t', p[1]
		print '#3\t', p[2]
		print '#4\t', p[3]
		print '#5\t', p[4]
		
		line_txt = p[3]
					
		tmp = tokenize(line_txt)
		
		print tmp

		#sys.exit()
		for w in tmp:
			pass
			#
		#		
		hfile.close()
	#------------------------------------------------------------------------------	
	sys.exit(0)
