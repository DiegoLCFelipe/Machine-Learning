import os, sys
import unittest
import numpy as np

os.system('cls')
sys.path.insert(0,os.path.abspath(os.curdir))

from machine_learning.models.analise_fatorial import teste_de_esfericidade_de_bartlett

class TestTesteEsfericidadeBartlett(unittest.TestCase):
    def setUp(self):
        self.dados_nao_esfericos = np.array([
            [5.8, 3.1, 3.1, 10.0],
            [4.0, 3.0, 4.0, 8.0],
            [1.0, 10.0, 4.0, 8.0],
            [6.0, 2.0, 4.0, 8.0]
        ])

    def teste_dados_nao_esfericos(self):
            estatistica, p_valor, gl = teste_de_esfericidade_de_bartlett(self.dados_nao_esfericos, 0.05)
            self.assertAlmostEqual(estatistica,32.35871978882966, places=6)  
            self.assertGreater(p_valor, 0.05)  

    #def teste_dados_esfericos(self):
    #        estatistica, p_valor = teste_de_esfericidade_de_bartlett(self.data2)
    #        self.assertAlmostEqual(estatistica, 10.644138, places=6)  
    #        self.assertGreater(p_valor, 0.05)  


if __name__ == "__main__":
    unittest.main() 