class StatsCalculator:
    # Sum array from input 'argA'
    # Return nan if empty
    def sumarray(self, argA):
        if len(argA) > 0:
            return sum(argA)
        return float('nan')

    # Find mean of array from input 'argA'
    # Return nan if empty
    def mean(self, argA):
        if len(argA) > 0:
            return float(sum(argA))/float(len(argA))
        return float('nan')

    # Testing access to class in a different file:
    def add(self, argA, argB):
        # Import Calculator:
        from calc import Calculator
        pyCalc = Calculator()
        valSum = pyCalc.add(argA, argB)
        return valSum
