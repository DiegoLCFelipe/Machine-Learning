import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm

class AnaliseDeFrequencias:
    def __init__(self, variavel1: pd.Series, variavel2: pd.Series) -> None:
        self.variavel1 = variavel1
        self.variavel2 = variavel2
        self.tabela_de_frequencias = self._gera_tabela_de_contingencia()
        self.tabela_de_frequencias_parametrizada = self._gera_tabela_parametrizada()

    def get_tabela_de_frequencias(self):
         return self.tabela_de_frequencias
             

    def _gera_tabela_de_contingencia(self) -> pd.DataFrame:
        """Função que gera a tabela de contingência - frequências com que cada combinação
        de categorias aparece em um conjunto de dados.

        Args:
            variavel1 (Series[float]): 
            variavel2 (Series[float]): 

        Returns:
            DataFrame[float]: Tabela de contingência
        """
        return pd.crosstab(self.variavel1, self.variavel2)

    def aplica_chi2_tabela_de_contingecia(self) -> object: 
            """Analisa a significância estatística da associação entre as variáveis (teste qui²)

            Args:
                tabela_contingencia (DataFrame[float]): tabela contendo as fequências observadas

            Returns:
                object: contendo a estatística de teste, p-value e as frequências esperadas 
            """
            return chi2_contingency(self.tabela_de_frequencias)
    
    def _gera_tabela_parametrizada(self) -> sm.stats.Table:
        """Função auxiliar para instanciar a tabela de frequências na biblioteca statsmodels

        Returns:
            sm.stats.Table: objeto que será manipulado pela biblioteca
        """
        return sm.stats.Table(self.tabela_de_frequencias)

   