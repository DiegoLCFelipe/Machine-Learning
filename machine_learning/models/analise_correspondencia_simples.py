import prince
import pandas as pd

class AnaliseCorrespondenciaSimples ():
    def __init__(self, dados):
        self.dados = dados
        self.analise_correspondencia_simples = self._analise_correspondencia_simples()

    def _analise_correspondencia_simples(self):
        return prince.CA().fit(self.dados)
    
    def get_autovalores(self):
        return self.analise_correspondencia_simples.eigenvalues_summary