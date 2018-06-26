import numpy as np
import pandas as pd
#_______________________ File selection box
filename = 'breast10D.csv' # show an "Open" dialog box and return the path to the selected file
Cfilename ='breast10.csv'
#_______________________ Converting csv file to list
df = pd.read_csv(filename)#,index_col=True)
U = df.values.tolist()
U = [[index] + value for index , value in enumerate(U) ]
#________________________ Equivalence partition function
def equivalence_partition( iterable , index ):
    classes = []
    dclasses = {}
    for o in iterable: # for each object
        # find the class it is in
        found = False
        for c in classes:
            indice_ele = next(iter(c))
            element = [iterable[indice_ele][ind] == o[ind] for ind in index]
            if all(element): # is it equivalent to this class?
                c.add( o[0])
                dclasses[o[0]] = c
                found = True
                break
        if not found: # it is in a new class
            classes.append( set([o[0]]))
            dclasses[o[0]] = classes[-1]
    return classes,dclasses
#_________________________ Finding lower approximation and positif region
def lower_appr(B):
    ind_B = equivalence_partition( U , B )[1]
    ind_d = equivalence_partition( U , D )[1]
    lower_appr_set = set()
    for x,ele in enumerate(U):
        if ind_B[x].issubset(ind_d[x]):
            lower_appr_set.add(x)
    return lower_appr_set
#________________________ Finding dependencey of features
def gamma(B):
    return float(len(lower_appr(B)))/float(len(U))
#_________________________ Rough set feature selection quickreduct algorithm
def qreduct(C):
    R = set()
    while True:
        T = R
        for x in C-R:
             if gamma(R.union(set([x]))) > gamma(T):
                T = R.union(set([x]))
             
        R = T
        if gamma(R) == gamma(C):
                break
    return R
#_________________________ Main fuction
decision=len(df.columns)#_________ defining le decision index
D = [decision]           
B = set([ i for i in range(1,decision)]) #__________ defining condition index
Features= qreduct(B)
