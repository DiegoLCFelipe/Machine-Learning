import prince
import pandas as pd
import numpy as np

class AnaliseCorrespondenciaSimples ():
    def __init__(self, dados:pd.DataFrame):
        self.dados = dados
        self.analise_correspondencia_simples = self._analise_correspondencia_simples()

    def _analise_correspondencia_simples(self) -> pd.DataFrame:
        return prince.CA().fit(self.dados)
    
    def autovalores(self) -> pd.DataFrame:
        return self.analise_correspondencia_simples.eigenvalues_summary
    
    def autovetor_linha(self) -> np.matrix:
        return self.analise_correspondencia_simples.svd_.U
    
    def autovetor_coluna(self) -> np.matrix:
        return self.analise_correspondencia_simples.svd_.V.T
    
    def coordenadas_linhas(self) -> pd.DataFrame:
        return self.analise_correspondencia_simples.row_coordinates(self.dados).reset_index()
    
    def coordenadas_colunas(self) -> pd.DataFrame:
        return self.analise_correspondencia_simples.column_coordinates(self.dados).reset_index()
    