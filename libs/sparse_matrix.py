class SparseMatrix:
  def __init__(self, shape):
    self.shape = shape
    self.height = shape[0]
    self.width = shape[1]
    self.matrix={}
    self.rows_agg_values = {}

  def __check_index(self, i, j):
    if (i >= 0) and (i < self.height) and (j >= 0) and (j < self.width) and\
       (i == int(i)) and (j == int(j)):
       return True
    
    raise(Exception(f'{str(i)}, {str(j)} index is wrong for matrix with shape={str(self.shape)}'))
  
  def __str__(self):
    result = ""

    for key, row in zip(self.matrix.keys(), self.matrix.values()):
       result += str(key)+': '+ str(row)[0:130] + '\n\r'
    
    return result

  def set_value(self, i, j, value):
    self.__check_index(i, j)
    i = str(int(i))
    j = str(int(j))
    if not self.matrix.__contains__(i):
      self.matrix[i] = {}
      self.rows_agg_values[i] = 0
    
    if self.matrix[i].__contains__(j):
      self.rows_agg_values[i] -= self.matrix[i][j]

    self.matrix[i][j] = value
    self.rows_agg_values[i] += value

  def get_value(self, i, j):
    self.__check_index(i, j)
    i = str(int(i))
    j = str(int(j))
    if not self.matrix.__contains__(i):
      return 0
    
    if not self.matrix[i].__contains__(j):
      return 0

    return self.matrix[i][j]
  
  def get_transition_probability(self, i, j):
    cell_value = self.get_value(i, j)
    if cell_value == 0:
      return 0

    i = str(int(i))
    return float(cell_value/self.rows_agg_values[i])