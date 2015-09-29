import sys

file = sys.argv[1]
lineno = int(sys.argv[2])
lines = [line.rstrip('\n') for line in open(file)]


this_line = lines[lineno]

print this_line
print '\n'

p = this_line.split(';')

print '---------------------------------------------------------------------------------'
print '#1\t', p[0]
print '#2\t', p[1]
print '#3\t', p[2]
print '#4\t', p[3]
print '#5\t', p[4]
