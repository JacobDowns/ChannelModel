# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:18:34 2016

@author: jake
"""


from dolfin import *
import numpy as np

mesh = UnitSquareMesh(2, 2)
#plot(mesh, interactive = True)

V = FunctionSpace(mesh, FiniteElement("Discontinuous Lagrange Trace", "triangle", 0))



#print V.tabulate_dof_coordinates()

print V.dofmap().num_facet_dofs()
#x = Vector()
print V.dofmap().tabulate_facet_dofs(100)