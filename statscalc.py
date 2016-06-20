class StatsCalculator:
	# Sum array from input 'argA'
	# Return nan if empty
	def Sum(self, argA):
		if len(argA) > 0:
			return sum(argA)
		return float('nan')

	# Find mean of array from input 'argA'
	# Return nan if empty
	def Mean(self, argA):
		if len(argA) > 0:
			return float(sum(argA))/float(len(argA))
		return float('nan')

	# Testing access to class in a different file:
	def Add(self, argA, argB):
		# Import Calculator:
		from calc import Calculator
		pyCalc = Calculator()
		valSum = pyCalc.Add(argA, argB)
		return valSum
