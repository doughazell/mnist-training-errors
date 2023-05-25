
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd

# 25/5/23 DH: Ctrl-C while in matplotlib loop
import sys
import signal
#signal.signal(signal.SIGINT, signal.SIG_DFL)

# dist is the distribution of birthdays. If it is not known it is assumed uniform
def collect_until_repeat(repeatNumExceed, dist = np.ones(365)/365):
    
  # We'll use this variable as the stopping condition
  repeat = False

  # We'll store the birthdays in this array
  outcomes = np.zeros(dist.shape[0])
  
  # n will count the number of people we meet. Each loop is a new person.
  n = 0
  while not repeat:
    # add one to the counter
    n += 1
           
    # simulate adding a person with a "random birthday" ie a 365 array with {1*1 + 364*0}
    # this {1+364} array is added to 'outcomes' 365 array
    outcomes += np.random.multinomial(1,dist)
    #if n < 3:
    #  print(outcomes)
    
    # check if we got a repeat
    if np.any(outcomes > repeatNumExceed):
      repeat = True
 
  return n

def run_many_simulations(repeatNumExceed, sim, dist = np.ones(365)/365):
  # count stores the result of each simulation in a big array
  count = np.zeros(sim)
  
  printNum = 0
  for idx_sim in range(sim):
    count[idx_sim] = collect_until_repeat(repeatNumExceed, dist)

    if printNum != idx_sim and idx_sim % 1000 == 0:
      print("Sim total:",idx_sim)
      printNum = idx_sim
  
  return count

def printHist(hist, trials, bins, timeout=1):
  plt.clf()
  
  #xAxis = trials/30
  if trials > 90000:
    xAxis = 1200
  else:
    xAxis = 1000

  randHist = plt.hist(hist, bins = np.arange(0,xAxis))
  plt.title("Random histogram after "+str(trials)+" trials on "+str(bins)+" bins")
  plt.xlabel("$n$")
  plt.ylabel("Number of occurences")
  
  #plt.show()
  plt.draw()
  plt.waitforbuttonpress(timeout=timeout)
  

# 25/5/23 DH:
def getRandomHist(bins,sim,printout=False):
  outcomes = np.zeros(bins)

  printNum = 0
  for num in range(sim):
    outcomes += np.random.multinomial(1, np.ones(bins)/bins )

    if printNum != num and num % 1000 == 0 and printout:
      printNum = num

      print(outcomes)
      printHist(outcomes, num, bins)

  return outcomes

def printRandomHist():
  bins = 100
  sim = 10000
  hist = getRandomHist(bins, sim, printout=True)

  printHist(hist,sim, bins, timeout=-1)

def signal_handler(sig, frame):
    print('\nYou pressed Ctrl+C...')
    
    sys.exit(0)

# =========================== MAIN ==============================
signal.signal(signal.SIGINT, signal_handler)

printRandomHist()
sys.exit(0)

#sim = 1000000
sim = 3000
#sim = 5
repeatNumExceed = 5
print("Running",sim,"simulations...")
counts = run_many_simulations(repeatNumExceed,sim)
print("Counts:", type(counts), counts.shape)

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
#repeat_histogram = plt.hist(counts, bins = np.arange(0,100))
repeat_histogram = plt.hist(counts, bins = np.arange(0,1000))

"""
for idx in range(len(repeat_histogram)):
  print(idx,"-",repeat_histogram[idx])
"""

plt.title("The number of additions before "+str(repeatNumExceed+1)+" repeats")
plt.xlabel("$n$")
plt.ylabel("Number of occurences")
plt.show()

"""
print("2 people for a repeat occurred {} times, which is relatively {:.4%}".format(
  repeat_histogram[0][2], repeat_histogram[0][2]/sim
))
"""

rel_dist = plt.hist(counts, bins = np.arange(0,100), density=True)

"""
plt.xlabel("$n$")
plt.ylabel("Probability of $n$")
plt.show()
"""

print("50% of time, no more than {} people were needed for a repeat.".format(
  np.where(np.cumsum(rel_dist[0])>0.5)[0][0]
  )
)

"""
repeatHistVals = repeat_histogram[0]
approxDistVals = rel_dist[0]
repeatHistValsSize = len(repeatHistVals)
for idx in range(repeatHistValsSize):
  print("Repeat Hist:",repeatHistVals[idx],"(",repeatHistVals[idx]/sim,"), Rel Dist:",approxDistVals[idx])
"""
