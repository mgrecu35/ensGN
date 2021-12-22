using PyCall
unshift!(PyVector(pyimport("sys")["path"]), ".")
p=pyimport(