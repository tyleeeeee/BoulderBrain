#def getValidHolds(image):
# 1. Call SAM to extract segments
# 2. Label segments as "hold" or "not hold".
#    Option 2a: Train a classifier model (finding data could be hard)
#    Option 2b: Use a formula to determine if a segment is a hold
# 3. For each hold, create a hold object with its location, shape, and color.
# 4. Return the set of holds.