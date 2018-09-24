#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Student(object):
	def __init__(self,name,score):
		self.name=name		#private variate
		self.__score=score
	
	def get_score(self):
		return self.__score
		
	def set_score(self,score):
		if 0 <= score <= 100:
			self.__score=score
		else:
			raise ValueError('bad score')
	
	def get_grade(self):
		if self.__score >= 90:
			return 'A'
		elif self.__score >= 60:
			return 'B'
		else:
			return 'C'
	
	def print_score(self):
		print('%s: %s' % (self.name,self.__score))
		
bart=Student('Jack',61)
Student.print_score(bart)
bart.city='BJ'
print('%s' % bart.city)
bart.set_score(59)
print('bart.get_score()=',bart.get_score())

print('DO NOT use bart._Student__score:',bart._Student__score)