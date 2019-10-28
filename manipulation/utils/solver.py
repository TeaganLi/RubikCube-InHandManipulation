#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from manipulation.utils import py222

hO = np.ones(2186, dtype=np.int) * 12
hP = np.ones(823543, dtype=np.int) * 12

moveStrs = {0: "U", 1: "U'", 2: "U2", 3: "R", 4: "R'", 5: "R2", 6: "F", 7: "F'", 8: "F2"}

# generate pruning table for the piece orientation states
def genOTable(s, d, lm=-3):
  index = py222.indexO(py222.getOP(s))
  if d < hO[index]:
    hO[index] = d
    for m in range(9):
      if int(m / 3) == int(lm / 3):
        continue
      genOTable(py222.doMove(s, m), d + 1, m)

# generate pruning table for the piece permutation states
def genPTable(s, d, lm=-3):
  index = py222.indexP(py222.getOP(s))
  if d < hP[index]:
    hP[index] = d
    for m in range(9):
      if int(m / 3) == int(lm / 3):
        continue
      genPTable(py222.doMove(s, m), d + 1, m)

# IDA* which prints all optimal solutions
def IDAStar(s, d, moves, lm=-3):
  if py222.isSolved(s):
    printMoves(moves)
    return True, np.array([moveStrs[move] for move in moves])
  else:
    sOP = py222.getOP(s)
    if d > 0 and d >= hO[py222.indexO(sOP)] and d >= hP[py222.indexP(sOP)]:
      dOptimal = False
      for m in range(9):
        if int(m / 3) == int(lm / 3):
          continue
        newMoves = moves[:]; newMoves.append(m)
        solved, move_tmp = IDAStar(py222.doMove(s, m), d - 1, newMoves, m)
        if solved and not dOptimal:
          dOptimal = True
          move_seq = move_tmp
      if dOptimal:
        return True, move_seq
  return False, 0

# print a move sequence from an array of move indices
def printMoves(moves):
  moveStr = ""
  for m in moves:
    moveStr += moveStrs[m] + " "
  print(moveStr)

# solve a cube state
def solveCube(s):
  # print cube state
  # py222.printCube(s)

  # FC-normalize stickers
  # print("normalizing stickers...")
  s = py222.normFC(s)

  # generate pruning tables
  # print("generating pruning tables...")
  genOTable(py222.initState(), 0)
  genPTable(py222.initState(), 0)

  # run IDA*
  # print("searching...")
  solved = False
  depth = 1
  while depth <= 11 and not solved:
    # print("depth {}".format(depth))
    solved, moves = IDAStar(s, depth, [])
    depth += 1
  return moves

if __name__ == "__main__":
  # input some scrambled state
  # s = py222.doAlgStr(py222.initState(), "D B L")
  # solve cube
  s = np.array([5, 2, 5, 0, 1, 3, 2, 3, 1, 2, 0, 4, 2, 3,4, 5, 0, 0, 5, 4, 1, 4, 1, 3])
  solveCube(s)

